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
    OUTCOMES,
    Cycle,
    ExecutionRecord,
    PartitionMetric,
)
from .sqlite import SqliteRegistry

__all__ = [
    "AttemptDetail",
    "AttemptSummary",
    "Cycle",
    "DEFAULT_MAX_FIX_RETRIES",
    "DuplicateAttemptError",
    "ExecutionRecord",
    "FAILURE_KINDS",
    "FailureRecord",
    "MetricPoint",
    "NotFoundError",
    "OUTCOMES",
    "PartitionMetric",
    "Registry",
    "RegistryError",
    "SqliteRegistry",
    "compute_fingerprint",
    "normalize_config",
]
