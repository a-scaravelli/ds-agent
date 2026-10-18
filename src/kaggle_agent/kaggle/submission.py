"""Fixed-code rules for submissions: format validation and the daily limit (design section 10).

Whether a submission is *worthwhile* and the reserve for the last hours are
policy of later tickets and are not implemented here.
"""

import csv
from collections.abc import Callable, Iterable
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from .errors import DailyLimitReachedError, SubmissionFormatError
from .models import DailyLimit, SubmissionSpec

_SHOWN = 5


def _show(values: Iterable[str]) -> str:
    items = list(values)
    shown = ", ".join(repr(v) for v in items[:_SHOWN])
    return shown + (f" and {len(items) - _SHOWN} more" if len(items) > _SHOWN else "")


def validate_submission(path: Path | str, spec: SubmissionSpec) -> None:
    """Check a submission file against the spec; raise ``SubmissionFormatError`` with the reason.

    Checks, in order: the file is not empty, header (names and order), every
    row has the right number of fields, row count, identifiers (none missing,
    no duplicates, no unknown ones) and that no prediction value is empty.
    """
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError) as exc:
        raise SubmissionFormatError(f"cannot read the submission file: {type(exc).__name__}") from None
    if not rows:
        raise SubmissionFormatError("the submission file is empty")
    header, body = rows[0], rows[1:]
    expected = list(spec.columns)
    if header != expected:
        if sorted(header) == sorted(expected):
            raise SubmissionFormatError(
                f"wrong column order: expected {expected}, got {header}"
            )
        raise SubmissionFormatError(f"wrong columns: expected {expected}, got {header}")
    for number, row in enumerate(body, start=2):
        if len(row) != len(expected):
            raise SubmissionFormatError(
                f"line {number} has {len(row)} fields, expected {len(expected)}"
            )
    wanted = len(spec.expected_ids)
    if len(body) < wanted:
        raise SubmissionFormatError(f"missing rows: expected {wanted}, got {len(body)}")
    if len(body) > wanted:
        raise SubmissionFormatError(f"extra rows: expected {wanted}, got {len(body)}")

    id_index = expected.index(spec.id_column)
    ids = [row[id_index] for row in body]
    if any(not value.strip() for value in ids):
        raise SubmissionFormatError(f"missing identifiers in column '{spec.id_column}'")
    seen, duplicates = set(), []
    for value in ids:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise SubmissionFormatError(f"duplicate identifiers: {_show(duplicates)}")
    unknown = [value for value in ids if value not in set(spec.expected_ids)]
    if unknown:
        raise SubmissionFormatError(f"unknown identifiers: {_show(unknown)}")
    for number, row in enumerate(body, start=2):
        for column, value in zip(expected, row):
            if column != spec.id_column and not value.strip():
                raise SubmissionFormatError(f"empty prediction in column '{column}' at line {number}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DailyLimitGuard:
    """Counts the submissions of the current UTC day against a daily limit.

    ``limit`` is required, with no default: it comes from Kaggle when the API
    reports it, otherwise the caller must supply it. ``clock`` returns an
    aware datetime and is injectable.
    """

    def __init__(
        self,
        limit: int,
        submitted_at: Iterable[datetime] = (),
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("the daily limit must be an integer of at least 1")
        self._limit = limit
        self._submitted = list(submitted_at)
        self._clock = clock

    def record(self, when: datetime | None = None) -> None:
        """Remember a submission made at ``when`` (default: now)."""
        self._submitted.append(when or self._clock())

    def status(self) -> DailyLimit:
        now = self._clock().astimezone(timezone.utc)
        midnight = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
        used = sum(1 for t in self._submitted if t.astimezone(timezone.utc) >= midnight)
        return DailyLimit(
            limit=self._limit,
            used=used,
            remaining=max(self._limit - used, 0),
            resets_at=midnight + timedelta(days=1),
        )

    def check(self) -> DailyLimit:
        """Return the status, or raise ``DailyLimitReachedError`` if nothing is left."""
        status = self.status()
        if status.remaining <= 0:
            raise DailyLimitReachedError(
                f"daily limit of {status.limit} submissions reached; "
                f"resets at {status.resets_at.isoformat()}"
            )
        return status
