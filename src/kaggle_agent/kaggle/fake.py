"""Deterministic in-memory Kaggle client for tests and offline work (no network).

It writes real files in ``download`` so the hashes use the same code as the
real client, and sends through the shared ``KaggleClient.submit``, so it
enforces the same contract. Calls are recorded for assertions.
"""

import csv
import hashlib
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .errors import CompetitionNotFoundError, KaggleError, RulesNotAcceptedError
from .interface import DEFAULT_DOWNLOAD_ROOT, KaggleClient, _utc_now
from .models import (
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_PENDING,
    CompetitionInfo,
    DailyLimit,
    DownloadedFile,
    DownloadResult,
    SubmissionRecord,
    SubmissionSpec,
)
from .submission import DailyLimitGuard

# one entry per poll of get_submission: None = pending, a float = complete with
# that public score, "error" = scoring failed; the last entry repeats
ScoreStep = float | str | None


@dataclass(frozen=True)
class FakeCompetition:
    """Everything the fake knows about one competition."""

    files: dict[str, bytes]
    description: str
    rules: str
    evaluation: str
    sample_submission: str
    daily_limit: int
    rules_accepted: bool = True
    data_description: str = ""
    extra_pages: dict[str, str] = field(default_factory=dict)
    score_script: Sequence[ScoreStep] = (None, 0.5)


@dataclass
class _Sent:
    record: SubmissionRecord
    polls: int = 0


class FakeKaggleClient(KaggleClient):
    """Fake client configured with competitions, an injectable clock and sleep."""

    def __init__(
        self,
        competitions: dict[str, FakeCompetition] | None = None,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] | None = None,
        earlier_submissions: dict[str, Sequence[datetime]] | None = None,
    ) -> None:
        super().__init__(clock=clock, sleep=sleep or (lambda seconds: None))
        self.competitions = dict(competitions or {})
        self.calls = []
        self.sent: list[SubmissionRecord] = []
        self._sent: dict[str, _Sent] = {}
        self._guards = {
            name: DailyLimitGuard(
                comp.daily_limit, (earlier_submissions or {}).get(name, ()), self._now
            )
            for name, comp in self.competitions.items()
        }

    def _now(self) -> datetime:
        return self._clock()

    def _competition(self, name: str, needs_rules: bool = True) -> FakeCompetition:
        comp = self.competitions.get(name)
        if comp is None:
            raise CompetitionNotFoundError(f"competition '{name}' not found")
        if needs_rules and not comp.rules_accepted:
            raise RulesNotAcceptedError(name)
        return comp

    def download(self, competition: str, dest_dir: Path | str | None = None) -> DownloadResult:
        self.calls.append(("download", competition))
        comp = self._competition(competition)
        target = Path(dest_dir) if dest_dir is not None else DEFAULT_DOWNLOAD_ROOT / competition
        target.mkdir(parents=True, exist_ok=True)
        files = []
        for name in sorted(comp.files):
            data = comp.files[name]
            (target / name).write_bytes(data)
            files.append(DownloadedFile(name, len(data), hashlib.sha256(data).hexdigest()))
        return DownloadResult(str(target), tuple(files))

    def get_competition_info(self, competition: str) -> CompetitionInfo:
        self.calls.append(("get_competition_info", competition))
        comp = self._competition(competition)
        pages = {
            "description": comp.description,
            "rules": comp.rules,
            "evaluation": comp.evaluation,
        }
        if comp.data_description:
            pages["data-description"] = comp.data_description
        pages.update(comp.extra_pages)
        return CompetitionInfo(
            comp.description, comp.rules, comp.evaluation, comp.data_description, pages
        )

    def get_submission_spec(self, competition: str) -> SubmissionSpec:
        self.calls.append(("get_submission_spec", competition))
        comp = self._competition(competition)
        rows = list(csv.reader(io.StringIO(comp.sample_submission)))
        columns = tuple(rows[0])
        return SubmissionSpec(columns, columns[0], tuple(row[0] for row in rows[1:]))

    def daily_limit(self, competition: str) -> DailyLimit:
        self.calls.append(("daily_limit", competition))
        self._competition(competition, needs_rules=False)
        return self._guards[competition].status()

    def _send(
        self, competition: str, file: Path, message: str, file_sha256: str
    ) -> SubmissionRecord:
        self.calls.append(("send", competition, file_sha256))
        now = self._now()
        record = SubmissionRecord(
            submission_id=f"fake-{len(self.sent) + 1}",
            competition=competition,
            file_sha256=file_sha256,
            message=message,
            submitted_at=now,
            status=STATUS_PENDING,
        )
        self._guards[competition].record(now)
        self.sent.append(record)
        self._sent[record.submission_id] = _Sent(record)
        return record

    def get_submission(self, competition: str, submission_id: str) -> SubmissionRecord:
        self.calls.append(("get_submission", competition, submission_id))
        comp = self._competition(competition)
        entry = self._sent.get(submission_id)
        if entry is None or entry.record.competition != competition:
            raise KaggleError(f"unknown submission '{submission_id}'")
        script = comp.score_script
        step = script[min(entry.polls, len(script) - 1)]
        entry.polls += 1
        base = entry.record
        if step is None:
            return base
        if step == "error":
            return _replace(base, STATUS_ERROR, None)
        return _replace(base, STATUS_COMPLETE, float(step))


def _replace(record: SubmissionRecord, status: str, score: float | None) -> SubmissionRecord:
    return SubmissionRecord(
        record.submission_id,
        record.competition,
        record.file_sha256,
        record.message,
        record.submitted_at,
        status,
        score,
    )
