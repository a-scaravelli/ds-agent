"""The real submission, after the review, and only on the lead's explicit command.

``submit`` checks, in order: (a) the cycle is STOPPED (the review was accepted) and holds a final
report with a persisted bundle; (b) the persisted folder passes ``load_persisted`` (the hashes of its
code and model are re-checked); then it builds the submission file by running the stored model's
predict stage on the competition's test file through ``pipeline.command.predict`` (JSON, never
pickle), writes it in the format of the sample submission, and checks (c) ``validate_submission``
and (d) ``DailyLimitGuard`` plus the limit reported by Kaggle.

Without ``confirm`` it is a dry run: it prints what it WOULD send and writes no marker and no
registry row. With ``confirm`` it writes ``out/submissions/<cycle>.pending.json`` atomically BEFORE
calling the client and rewrites it as 'recorded' after ``record_submission``. A marker still in the
'sending' state with no matching registry row means a crash between the send and the record: the
command then refuses (exit 24) and never resends by itself; the lead checks the Kaggle submissions
page and passes ``--assume-not-sent`` (or deletes the marker). The wait for the score is bounded and
resumable: a rerun waits again without sending again.

The model is the one fitted on the development data only (the persisted model that was validated and
scored on the holdout); it is not retrained on development plus holdout.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kaggle_agent.kaggle import (
    DailyLimitGuard,
    DailyLimitReachedError,
    KaggleError,
    ScoreTimeoutError,
    SubmissionFormatError,
    SubmissionRecord,
    record_score,
    record_submission,
    validate_submission,
)
from kaggle_agent.kaggle.models import STATUS_COMPLETE, STATUS_PENDING
from kaggle_agent.loop.states import STOPPED
from kaggle_agent.pipeline import FinalReport, PersistedBundleError, command, load_persisted, report_path
from kaggle_agent.registry import NotFoundError, Registry, RegistryError, SqliteRegistry, SubmissionEntry

from . import codes
from .build import Dependencies
from .codes import CommandResult
from .config import Config
from .errors import OrchestratorError, ProjectError, SubmitRefused
from .project import file_sha256, read_project, write_atomic

SUBMISSIONS_DIR = "submissions"
SENDING = "sending"
RECORDED = "recorded"
SECTION_BEGIN = "<!-- submission:begin -->"
SECTION_END = "<!-- submission:end -->"


def submission_file(out: Path, cycle_id: str) -> Path:
    return out / SUBMISSIONS_DIR / f"{cycle_id}.csv"


def marker_path(out: Path, cycle_id: str) -> Path:
    return out / SUBMISSIONS_DIR / f"{cycle_id}.pending.json"


def read_marker(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SubmitRefused(codes.MARKER, f"the marker {path} is unreadable ({error}); check the Kaggle "
                            "submissions page by hand, then delete it or pass --assume-not-sent") from None


def _write_marker(path: Path, data: dict) -> None:
    write_atomic(path, json.dumps(data, indent=2, sort_keys=True))


def _utc(text: str) -> datetime:
    when = datetime.fromisoformat(text)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


# -- the file ------------------------------------------------------------------------------------


def build_submission(persisted_dir: Path, test_csv: Path, spec, target: Path, deps: Dependencies,
                     timeout: float) -> tuple[int, list[str]]:
    """Predict ``test_csv`` with the stored model and write ``target``; returns (rows, columns)."""
    try:
        test = pd.read_csv(test_csv)
    except (OSError, ValueError) as error:
        raise SubmitRefused(codes.INVALID_FILE, f"cannot read the test file {test_csv}: {error}") from None
    if spec.id_column not in test.columns:
        raise SubmitRefused(codes.INVALID_FILE, f"the test file has no id column {spec.id_column!r} "
                            f"(columns: {', '.join(map(str, test.columns))})")
    with tempfile.TemporaryDirectory(prefix="submission-") as work:
        output = Path(work) / "predictions.json"
        result = command.predict(persisted_dir, test_csv, output, deps.sandbox, timeout)
        if result.code != 0:
            raise SubmitRefused(codes.INVALID_FILE, "the stored model could not predict the test file: "
                                + " ".join(result.lines))
        predictions = json.loads(output.read_text(encoding="utf-8"))
    names = list(spec.columns[1:])
    if len(predictions) != len(test):
        raise SubmitRefused(codes.INVALID_FILE, f"{len(predictions)} predictions for {len(test)} test rows")
    matrix = bool(predictions) and isinstance(predictions[0], list)
    width = len(predictions[0]) if matrix else 1
    if width != len(names):
        raise SubmitRefused(
            codes.INVALID_FILE,
            f"the model gives {width} value(s) per row but the sample submission has {len(names)} "
            f"prediction column(s) ({', '.join(names)}); this competition format is not supported")
    frame = pd.DataFrame({spec.id_column: test[spec.id_column].to_numpy()})
    for index, name in enumerate(names):
        frame[name] = [row[index] for row in predictions] if matrix else predictions
    frame = frame[list(spec.columns)]
    write_atomic(target, frame.to_csv(index=False))
    return len(frame), list(frame.columns)


# -- the daily limit -------------------------------------------------------------------------------


def check_daily_limit(deps: Dependencies, slug: str, entries: list[SubmissionEntry]):
    """Kaggle's own count and the cycle's recorded submissions must both leave room."""
    status = deps.kaggle.daily_limit(slug)
    guard = DailyLimitGuard(status.limit, [_utc(e.submitted_at) for e in entries], deps.clock)
    if status.remaining <= 0:
        raise SubmitRefused(codes.DAILY_LIMIT, f"daily limit of {status.limit} submissions reached; "
                            f"resets at {status.resets_at.isoformat()}")
    try:
        mine = guard.check()
    except DailyLimitReachedError as error:
        raise SubmitRefused(codes.DAILY_LIMIT, str(error)) from None
    return min(status.remaining, mine.remaining), status


# -- the report file -------------------------------------------------------------------------------


def refresh_report_file(registry: Registry, out: Path, cycle_id: str) -> None:
    """Rewrite the marked 'Submission' section of the Markdown report from the registry."""
    path = report_path(out, cycle_id)
    entries = registry.list_submissions(cycle_id)
    if not path.exists() or not entries:
        return
    lines = [SECTION_BEGIN, "## Submission", ""]
    for entry in entries:
        score = "pending" if entry.public_score is None else f"{entry.public_score:.5f}"
        lines.append(f"- {entry.submission_id}: {entry.status}, public score {score}, submitted "
                     f"{entry.submitted_at}, file sha256 {entry.file_sha256[:12]}")
    lines.append(SECTION_END)
    section = "\n".join(lines) + "\n"
    text = path.read_text(encoding="utf-8")
    if SECTION_BEGIN in text and SECTION_END in text:
        head, rest = text.split(SECTION_BEGIN, 1)
        tail = rest.split(SECTION_END, 1)[1].lstrip("\n")
        text = head + section + (("\n" + tail) if tail else "")
    else:
        text = text.rstrip("\n") + "\n\n" + section
    path.write_text(text, encoding="utf-8")


# -- the command -----------------------------------------------------------------------------------


def _wait_and_record(registry: Registry, deps: Dependencies, config: Config, cycle_id: str,
                     entry: SubmissionEntry, out: Path) -> CommandResult:
    slug = config.competition.slug
    if entry.status == STATUS_PENDING:
        try:
            final = deps.kaggle.wait_for_score(slug, entry.submission_id, config.runtime.wait_seconds,
                                               config.runtime.poll_interval)
        except ScoreTimeoutError:
            refresh_report_file(registry, out, cycle_id)
            return CommandResult(codes.SCORE_PENDING, (
                f"submission {entry.submission_id} was sent but its score is still pending after "
                f"{config.runtime.wait_seconds:g} seconds.",
                "Run the same command again (with --confirm-submit) to keep waiting: nothing is sent twice."))
        record_score(registry, final, deps.clock())
        refresh_report_file(registry, out, cycle_id)
        if final.status != STATUS_COMPLETE:
            return CommandResult(codes.ERROR, (
                f"submission {entry.submission_id} was sent but Kaggle reports the status "
                f"{final.status!r}, with no score.",))
        return CommandResult(codes.FINISHED, (
            f"submitted: {entry.submission_id}", f"public score: {final.public_score}",
            f"recorded in the registry; report: {report_path(out, cycle_id)}"))
    refresh_report_file(registry, out, cycle_id)
    return CommandResult(codes.FINISHED, (
        f"already submitted: {entry.submission_id} ({entry.status}); nothing is sent again",
        f"public score: {entry.public_score}"))


def _find(entries: list[SubmissionEntry], sha: str) -> SubmissionEntry | None:
    return next((e for e in entries if e.file_sha256 == sha), None)


def submit(config: Config, deps: Dependencies, confirm: bool = False,
           assume_not_sent: bool = False) -> CommandResult:
    """The submit command; ``SubmitRefused`` becomes a result with its exit code."""
    try:
        return _submit(config, deps, confirm, assume_not_sent)
    except SubmitRefused as refusal:
        return CommandResult(refusal.code, (f"refused: {refusal}", "nothing was sent."))


def _submit(config: Config, deps: Dependencies, confirm: bool, assume_not_sent: bool) -> CommandResult:
    paths, slug = config.paths, config.competition.slug
    state = read_project(paths)
    if state is None:
        raise ProjectError(f"no project in {paths.folder}: run start first")
    out = paths.out
    try:
        registry = SqliteRegistry(str(paths.registry), read_only=not confirm)
    except RegistryError as error:
        raise ProjectError(str(error)) from None
    try:
        cycle_id = state.cycle_id
        cycle = registry.get_cycle(cycle_id)
        stored = registry.get_final_report(cycle_id)
        if cycle.state != STOPPED or stored is None:
            raise SubmitRefused(codes.NOT_ACCEPTED, f"the cycle is in state {cycle.state}: the final report "
                                "must be reviewed and accepted first (the lead file's [review] table)")
        report = FinalReport.from_dict(stored.content)
        if report.chosen is None or report.persisted_path is None:
            raise SubmitRefused(codes.NOT_ACCEPTED, "the cycle has no persisted bundle to predict with")
        try:
            persisted = load_persisted(report.persisted_path)
        except PersistedBundleError as error:
            raise SubmitRefused(codes.ERROR, f"the persisted bundle is not valid: {error}") from None
        entries = registry.list_submissions(cycle_id)
        marker_file = marker_path(out, cycle_id)
        marker = read_marker(marker_file)
        if confirm and marker is not None:
            known = _find(entries, marker.get("sha256", ""))
            if known is not None:
                if marker.get("state") != RECORDED:
                    _write_marker(marker_file, {**marker, "state": RECORDED,
                                                "submission_id": known.submission_id})
                return _wait_and_record(registry, deps, config, cycle_id, known, out)
            if marker.get("state") == SENDING and not assume_not_sent:
                raise SubmitRefused(codes.MARKER, (
                    f"a previous submit stopped between sending and recording ({marker_file}, file "
                    f"sha256 {marker.get('sha256', '?')[:12]}). It may have reached Kaggle: check the Kaggle "
                    "submissions page by hand. If it did NOT, delete the marker or run again with "
                    "--assume-not-sent; if it did, record nothing here and do not send again"))

        spec = deps.kaggle.get_submission_spec(slug)
        target = submission_file(out, cycle_id)
        rows, columns = build_submission(persisted.root, paths.data / state.test_file, spec, target, deps,
                                         config.runtime.bundle_timeout)
        try:
            validate_submission(target, spec)
        except SubmissionFormatError as error:
            raise SubmitRefused(codes.INVALID_FILE, f"the submission file is invalid: {error}") from None
        sha = file_sha256(target)
        sent = _find(entries, sha)
        if confirm and sent is not None:  # the same file was already sent: never send it twice
            return _wait_and_record(registry, deps, config, cycle_id, sent, out)
        left, limit = check_daily_limit(deps, slug, entries)
        chosen = report.chosen
        message = (f"kaggle-agent {cycle_id} {chosen['bundle_id']} "
                   f"validation {report.metric} {chosen['validation_mean']:.5f}")
        holdout = report.holdout
        holdout_text = (f"{holdout['score']:.5f}" if holdout.get("status") == "succeeded"
                        else f"n/a ({holdout.get('status')})")
        if not confirm:
            return CommandResult(codes.FINISHED, (
                "DRY RUN: nothing is sent.", f"file: {target}", f"sha256: {sha}", f"rows: {rows}",
                f"columns: {', '.join(columns)}", f"message: {message}",
                f"chosen bundle: {chosen['bundle_id']} (attempt {chosen['attempt_id']}), validation "
                f"{report.metric} {chosen['validation_mean']:.5f}, holdout {holdout_text}",
                f"daily limit: {left} of {limit.limit} submissions left today "
                f"(resets at {limit.resets_at.isoformat()})",
                "To send it, run the same command with --confirm-submit."))

        _write_marker(marker_file, {"state": SENDING, "sha256": sha, "message": message,
                                    "file": str(target), "at": deps.clock().astimezone(timezone.utc).isoformat()})
        try:
            record: SubmissionRecord = deps.kaggle.submit(slug, target, message)
        except SubmissionFormatError as error:  # refused before anything was sent
            marker_file.unlink(missing_ok=True)
            raise SubmitRefused(codes.INVALID_FILE, f"Kaggle client refused the file: {error}") from None
        except DailyLimitReachedError as error:
            marker_file.unlink(missing_ok=True)
            raise SubmitRefused(codes.DAILY_LIMIT, str(error)) from None
        except KaggleError as error:
            raise SubmitRefused(codes.ERROR, f"the send failed ({error}); it may or may not have reached "
                                f"Kaggle: check the submissions page; the marker {marker_file} stays") from None
        entry = record_submission(registry, cycle_id, record)
        _write_marker(marker_file, {"state": RECORDED, "sha256": sha, "message": message, "file": str(target),
                                    "at": deps.clock().astimezone(timezone.utc).isoformat(),
                                    "submission_id": record.submission_id})
        return _wait_and_record(registry, deps, config, cycle_id, entry, out)
    except (NotFoundError, RegistryError) as error:
        raise ProjectError(f"registry: {error}") from None
    finally:
        registry.close()
