"""Experiment registry, deduplication and memory of failures."""

from .errors import DuplicateAttemptError, NotFoundError, RegistryError
from .fingerprint import compute_fingerprint, normalize_config
from .interface import (
    AttemptDetail,
    AttemptSummary,
    FailureRecord,
    MetricPoint,
    Registry,
)
from .records import (
    DEFAULT_MAX_FIX_RETRIES,
    FAILURE_KINDS,
    LLM_OUTCOMES,
    OUTCOMES,
    ComparisonRecord,
    Cycle,
    ExecutionRecord,
    HoldoutRead,
    HoldoutSession,
    LlmCallEntry,
    PartitionMetric,
    SubmissionEntry,
)
from .sqlite import SqliteRegistry

__all__ = [
    "AttemptDetail",
    "AttemptSummary",
    "ComparisonRecord",
    "Cycle",
    "DEFAULT_MAX_FIX_RETRIES",
    "DuplicateAttemptError",
    "ExecutionRecord",
    "FAILURE_KINDS",
    "FailureRecord",
    "HoldoutRead",
    "HoldoutSession",
    "LLM_OUTCOMES",
    "LlmCallEntry",
    "MetricPoint",
    "NotFoundError",
    "OUTCOMES",
    "PartitionMetric",
    "Registry",
    "RegistryError",
    "SqliteRegistry",
    "SubmissionEntry",
    "compute_fingerprint",
    "normalize_config",
]
