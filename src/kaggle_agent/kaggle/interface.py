"""Replaceable interface of the Kaggle client (design section 10).

Invariants every implementation keeps:

- Credentials are read only by the client, never logged, never put in a
  result, an exception message or the registry.
- ``submit`` is a template method: it ALWAYS validates the file against the
  sample submission and checks the daily limit, and only then calls the
  ``_send`` hook. An implementation overrides ``_send`` and cannot skip the
  checks. A rejected submission is never sent and never counted.
- Nothing in the code base calls ``submit`` automatically: sending is an
  explicit action of the user after the review.
- Accepting the rules of a competition is a manual step of the human and has
  no operation here; operations on a competition whose rules are not accepted
  raise ``RulesNotAcceptedError``.
- ``download`` writes to ``data/<competition>/`` by default and returns the
  SHA-256 of every file and a combined ``data_hash``, equal for equal bytes
  whatever the directory.
- The daily limit is per UTC day. Its source is Kaggle when the API reports
  it; an implementation that cannot know it needs the limit as a required
  argument, with no default.
- The score is returned and recorded (see ``recording.py``), never used to
  change the evaluation protocol. The client itself never writes to the
  registry.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import time

from .errors import DailyLimitReachedError, ScoreTimeoutError
from .models import (
    STATUS_PENDING,
    CompetitionInfo,
    DailyLimit,
    DownloadResult,
    SubmissionRecord,
    SubmissionSpec,
)
from .submission import validate_submission

DEFAULT_DOWNLOAD_ROOT = Path("data")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_of_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class KaggleClient(ABC):
    """Access to Kaggle: download, rules, submission, score, daily limit."""

    def __init__(
        self,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep

    # -- reads ---------------------------------------------------------------

    @abstractmethod
    def download(self, competition: str, dest_dir: Path | str | None = None) -> DownloadResult:
        """Download the competition files (default ``data/<competition>/``) and hash them."""

    @abstractmethod
    def get_competition_info(self, competition: str) -> CompetitionInfo:
        """Description, rules, evaluation and data description as text, plus every page by name.

        Raises ``KaggleError`` if a required page (description, rules,
        evaluation) is missing; ``data_description`` is empty when the
        competition has no such page.
        """

    @abstractmethod
    def get_submission_spec(self, competition: str) -> SubmissionSpec:
        """Columns, identifier column and expected ids, from the sample submission."""

    @abstractmethod
    def daily_limit(self, competition: str) -> DailyLimit:
        """Limit, submissions used today (UTC), submissions left and reset time."""

    @abstractmethod
    def get_submission(self, competition: str, submission_id: str) -> SubmissionRecord:
        """The current state of one submission, with its score once complete."""

    # -- submission ----------------------------------------------------------

    @abstractmethod
    def _send(
        self, competition: str, file: Path, message: str, file_sha256: str
    ) -> SubmissionRecord:
        """Send an already validated file. Called only by ``submit``."""

    def submit(self, competition: str, file: Path | str, message: str) -> SubmissionRecord:
        """Validate the format, check the daily limit, then send.

        Raises ``SubmissionFormatError`` or ``DailyLimitReachedError`` before
        anything is sent.
        """
        path = Path(file)
        validate_submission(path, self.get_submission_spec(competition))
        status = self.daily_limit(competition)
        if status.remaining <= 0:
            raise DailyLimitReachedError(
                f"daily limit of {status.limit} submissions reached; "
                f"resets at {status.resets_at.isoformat()}"
            )
        return self._send(competition, path, message, sha256_of_file(path))

    def wait_for_score(
        self,
        competition: str,
        submission_id: str,
        timeout: float,
        poll_interval: float,
    ) -> SubmissionRecord:
        """Read the submission until it is no longer pending.

        Returns the final record: complete with a score, or with an error
        status and no score. Raises ``ScoreTimeoutError`` once ``timeout``
        seconds have elapsed on the client clock.
        """
        started = self._clock()
        while True:
            record = self.get_submission(competition, submission_id)
            if record.status != STATUS_PENDING:
                return record
            if (self._clock() - started).total_seconds() >= timeout:
                raise ScoreTimeoutError(
                    f"submission {submission_id} still pending after {timeout} seconds"
                )
            self._sleep(poll_interval)
