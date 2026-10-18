"""Record a submission and its score in the registry (design section 10).

The only module of this subpackage that imports ``registry``. The client
itself never writes to the registry; the caller does, with these functions.
The score is recorded and never used to change the evaluation protocol.
"""

from datetime import datetime, timezone

from kaggle_agent.registry import Registry, SubmissionEntry

from .models import SubmissionRecord


def record_submission(registry: Registry, cycle_id: str, record: SubmissionRecord) -> SubmissionEntry:
    """Store a sent submission under a cycle, as Kaggle reported it (usually pending)."""
    entry = SubmissionEntry(
        cycle_id=cycle_id,
        competition=record.competition,
        submission_id=record.submission_id,
        file_sha256=record.file_sha256,
        message=record.message,
        submitted_at=record.submitted_at.isoformat(),
        status=record.status,
        public_score=record.public_score,
    )
    registry.add_submission(entry)
    return entry


def record_score(
    registry: Registry, record: SubmissionRecord, completed_at: datetime | None = None
) -> None:
    """Record the end of the scoring of an already recorded submission, once."""
    when = completed_at or datetime.now(timezone.utc)
    registry.update_submission_score(
        record.competition,
        record.submission_id,
        record.status,
        record.public_score,
        when.isoformat(),
    )
