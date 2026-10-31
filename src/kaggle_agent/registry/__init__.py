"""Experiment registry, deduplication and memory of failures."""

from .errors import DuplicateAttemptError, NotFoundError, RegistryError
from .fingerprint import compute_fingerprint, normalize_config
from .interface import (
    AttemptDetail,
    AttemptSummary,
    FailureRecord,
    MetricPoint,
    Registry,
    RoundDetail,
    RoundMetrics,
)
from .records import (
    DEFAULT_MAX_FIX_RETRIES,
    FAILURE_KINDS,
    LLM_OUTCOMES,
    OUTCOMES,
    BundleRecord,
    ComparisonRecord,
    Cycle,
    ExecutionRecord,
    HoldoutRead,
    HoldoutSession,
    LlmCallEntry,
    PartitionMetric,
    RoleTrace,
    SubmissionEntry,
)
from .sqlite import SqliteRegistry

__all__ = [
    "AttemptDetail",
    "AttemptSummary",
    "BundleRecord",
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
    "RoleTrace",
    "RoundDetail",
    "RoundMetrics",
    "SqliteRegistry",
    "SubmissionEntry",
    "compute_fingerprint",
    "normalize_config",
]
