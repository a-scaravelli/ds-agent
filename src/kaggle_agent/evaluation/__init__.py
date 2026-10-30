"""Frozen metric and evaluation protocol, fixed code the model cannot modify."""

from .candidate import Candidate
from .engine import (
    EvaluationEngine,
    EvaluationResult,
    HoldoutEvaluator,
    HoldoutResult,
)
from .errors import (
    EvaluationError,
    HoldoutSessionError,
    ProtocolHashMismatchError,
    ProtocolNotFrozenError,
)
from .goal import CriterionResult, GoalStatus, evaluate_goal
from .hashing import evaluation_code_hash, protocol_hash
from .metrics import METRICS, MetricSpec, get_metric, register_metric
from .protocol import Protocol
from .schemes import SCHEMES, SplitScheme, derive_seed, get_scheme, register_scheme
from .split import FrozenSplit, HoldoutAccessor, Partition
from .threshold import Comparison, compare, dispersion

__all__ = [
    "Candidate",
    "Comparison",
    "CriterionResult",
    "EvaluationEngine",
    "EvaluationError",
    "EvaluationResult",
    "FrozenSplit",
    "GoalStatus",
    "HoldoutAccessor",
    "HoldoutEvaluator",
    "HoldoutResult",
    "HoldoutSessionError",
    "METRICS",
    "MetricSpec",
    "Partition",
    "Protocol",
    "ProtocolHashMismatchError",
    "ProtocolNotFrozenError",
    "SCHEMES",
    "SplitScheme",
    "compare",
    "derive_seed",
    "dispersion",
    "evaluate_goal",
    "evaluation_code_hash",
    "get_metric",
    "get_scheme",
    "protocol_hash",
    "register_metric",
    "register_scheme",
]
