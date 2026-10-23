"""Submission format validation and the daily limit guard (design section 10)."""

from datetime import timedelta

import pytest

from kaggle_agent.kaggle import (
    DailyLimitGuard,
    DailyLimitReachedError,
    SubmissionFormatError,
    SubmissionSpec,
    validate_submission,
)
from kaggle_helpers import DAY, IDS, Clock, write_submission

SPEC = SubmissionSpec(("id", "target"), "id", IDS)


def test_valid_submission_is_accepted(tmp_path):
    validate_submission(write_submission(tmp_path / "s.csv"), SPEC)
    validate_submission(str(write_submission(tmp_path / "t.csv", ["c,1", "a,2", "b,3"])), SPEC)


@pytest.mark.parametrize(
    "header, rows, reason",
    [
        ("id,label", None, "wrong columns"),
        ("target,id", None, "wrong column order"),
        ("id,target", ["a,1", "b,1"], "missing rows"),
        ("id,target", ["a,1", "b,1", "c,1", "d,1"], "extra rows"),
        ("id,target", ["a,1", "a,1", "c,1"], "duplicate identifiers"),
        ("id,target", ["a,1", "b,1", "z,1"], "unknown identifiers"),
        ("id,target", ["a,1", "b,", "c,1"], "empty prediction"),
        ("id,target", ["a,1", ",1", "c,1"], "missing identifiers"),
        ("id,target", ["a,1", "b", "c,1"], "fields"),
    ],
)
def test_invalid_submissions_are_rejected_with_a_reason(tmp_path, header, rows, reason):
    path = write_submission(tmp_path / "s.csv", rows, header)
    with pytest.raises(SubmissionFormatError, match=reason):
        validate_submission(path, SPEC)


def test_empty_and_unreadable_files_are_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(SubmissionFormatError, match="empty"):
        validate_submission(empty, SPEC)
    with pytest.raises(SubmissionFormatError, match="cannot read"):
        validate_submission(tmp_path / "missing.csv", SPEC)


def test_guard_counts_the_current_utc_day():
    clock = Clock()
    guard = DailyLimitGuard(2, [DAY - timedelta(days=1)], clock)
    assert guard.status().used == 0
    guard.record()
    guard.check()
    guard.record()
    status = guard.status()
    assert (status.limit, status.used, status.remaining) == (2, 2, 0)
    assert status.resets_at == DAY.replace(hour=0, minute=0) + timedelta(days=1)
    with pytest.raises(DailyLimitReachedError):
        guard.check()
    clock.now += timedelta(days=1)
    assert guard.status().remaining == 2


@pytest.mark.parametrize("bad", [0, -1, True, None, 2.5])
def test_guard_requires_a_valid_limit(bad):
    with pytest.raises(ValueError):
        DailyLimitGuard(bad)
    with pytest.raises(TypeError):
        DailyLimitGuard()  # no default
