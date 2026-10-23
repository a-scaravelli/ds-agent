"""Execution record, partition metric and cycle (design section 7)."""

import json
import math
from dataclasses import dataclass

from kaggle_agent.proposals import ShortPlan, ValidationError
from kaggle_agent.proposals.errors import check_keys, loads_object, require_text

OUTCOMES = ("succeeded", "failed", "inconclusive")
FAILURE_KINDS = ("code_error", "hypothesis_falsified", "timeout", "below_threshold")
DEFAULT_MAX_FIX_RETRIES = 3
LLM_OUTCOMES = ("ok", "rate_limited", "quota", "cli_error", "invalid_output")

_METRIC_FIELDS = ("partition", "mean", "dispersion")
_EXEC_FIELDS = (
    "id",
    "attempt_id",
    "config",
    "seed",
    "code_hash",
    "data_hash",
    "protocol_hash",
    "metrics",
    "duration_seconds",
    "resources",
    "outcome",
    "failure_kind",
    "reason",
    "artifacts",
)
_CYCLE_FIELDS = (
    "id",
    "plan",
    "iterations",
    "stagnation_k",
    "max_fix_retries",
    "state",
    "protocol_hash",
)


def _finite(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(field, "must be a number")
    if not math.isfinite(value):
        raise ValidationError(field, "must be finite")
    return float(value)


def _int(field: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(field, "must be an integer")
    if value < minimum:
        raise ValidationError(field, f"must be at least {minimum}")
    return value


def _json_compatible(field: str, value: object) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(field, f"must be JSON-compatible: {exc}") from exc


@dataclass(frozen=True)
class PartitionMetric:
    """Mean and dispersion of the metric on one partition."""

    partition: str
    mean: float
    dispersion: float

    def __post_init__(self) -> None:
        require_text("partition", self.partition)
        object.__setattr__(self, "mean", _finite("mean", self.mean))
        dispersion = _finite("dispersion", self.dispersion)
        if dispersion < 0:
            raise ValidationError("dispersion", "must not be negative")
        object.__setattr__(self, "dispersion", dispersion)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _METRIC_FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "PartitionMetric":
        check_keys("metric", data, _METRIC_FIELDS)
        return cls(**{name: data.get(name) for name in _METRIC_FIELDS})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "PartitionMetric":
        return cls.from_dict(loads_object("metric", text))


@dataclass(frozen=True)
class ExecutionRecord:
    """What happened when an attempt ran, linked to the ``Attempt`` by id.

    An attempt has one or more records: the initial run plus its code-fix
    retries. ``failure_kind`` is required when ``outcome`` is not
    ``succeeded`` and must be None otherwise. On an execution record,
    ``code_error`` only means that this run crashed; at attempt level it means
    the code still failed after the fix-retry cap (enforced by the registry).
    """

    id: str
    attempt_id: str
    config: dict
    seed: int
    code_hash: str
    data_hash: str
    protocol_hash: str
    metrics: tuple[PartitionMetric, ...]
    duration_seconds: float
    outcome: str
    resources: dict | None = None
    failure_kind: str | None = None
    reason: str | None = None
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("id", "attempt_id", "code_hash", "data_hash", "protocol_hash"):
            require_text(name, getattr(self, name))
        if not isinstance(self.config, dict):
            raise ValidationError("config", "must be an object")
        _json_compatible("config", self.config)
        if self.resources is None:
            object.__setattr__(self, "resources", {})
        if not isinstance(self.resources, dict):
            raise ValidationError("resources", "must be an object")
        _json_compatible("resources", self.resources)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValidationError("seed", "must be an integer")
        duration = _finite("duration_seconds", self.duration_seconds)
        if duration < 0:
            raise ValidationError("duration_seconds", "must not be negative")
        object.__setattr__(self, "duration_seconds", duration)
        if not isinstance(self.metrics, (list, tuple)):
            raise ValidationError("metrics", "must be a list")
        metrics = tuple(
            m if isinstance(m, PartitionMetric) else PartitionMetric.from_dict(m)
            for m in self.metrics
        )
        object.__setattr__(self, "metrics", metrics)
        if len({m.partition for m in metrics}) != len(metrics):
            raise ValidationError("metrics", "partitions must be unique")
        if isinstance(self.artifacts, str) or not isinstance(self.artifacts, (list, tuple)):
            raise ValidationError("artifacts", "must be a list of strings")
        for item in self.artifacts:
            require_text("artifacts", item)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.outcome not in OUTCOMES:
            raise ValidationError("outcome", f"must be one of {', '.join(OUTCOMES)}")
        if self.outcome == "succeeded":
            if self.failure_kind is not None:
                raise ValidationError("failure_kind", "must be absent when succeeded")
            if not metrics:
                raise ValidationError("metrics", "must not be empty when succeeded")
            if self.reason is not None:
                require_text("reason", self.reason)
        else:
            if self.failure_kind is None:
                raise ValidationError("failure_kind", "is required unless succeeded")
            if self.failure_kind not in FAILURE_KINDS:
                raise ValidationError(
                    "failure_kind", f"must be one of {', '.join(FAILURE_KINDS)}"
                )
            require_text("reason", self.reason)

    def to_dict(self) -> dict:
        out = {}
        for name in _EXEC_FIELDS:
            value = getattr(self, name)
            if name == "metrics":
                value = [m.to_dict() for m in value]
            elif name == "artifacts":
                value = list(value)
            out[name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionRecord":
        check_keys("execution", data, _EXEC_FIELDS)
        kwargs = {name: data[name] for name in _EXEC_FIELDS if name in data}
        for name in ("id", "attempt_id", "code_hash", "data_hash", "protocol_hash", "outcome"):
            kwargs.setdefault(name, None)
        kwargs.setdefault("config", {})
        kwargs.setdefault("metrics", ())
        for name in ("seed", "duration_seconds"):
            kwargs.setdefault(name, None)
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ExecutionRecord":
        return cls.from_dict(loads_object("execution", text))


@dataclass(frozen=True)
class Cycle:
    """A work cycle: approved plan, parameters and state.

    ``state`` is free non-empty text: the transitions belong to the loop.
    ``protocol_hash`` is the hash of the frozen evaluation protocol, set once
    at approval (None until then).
    """

    id: str
    plan: ShortPlan
    iterations: int
    stagnation_k: int
    state: str
    max_fix_retries: int = DEFAULT_MAX_FIX_RETRIES
    protocol_hash: str | None = None

    def __post_init__(self) -> None:
        require_text("id", self.id)
        require_text("state", self.state)
        if self.protocol_hash is not None:
            require_text("protocol_hash", self.protocol_hash)
        if not isinstance(self.plan, ShortPlan):
            raise ValidationError("plan", "must be a ShortPlan")
        _int("iterations", self.iterations, 1)
        _int("stagnation_k", self.stagnation_k, 1)
        _int("max_fix_retries", self.max_fix_retries, 0)

    def to_dict(self) -> dict:
        out = {name: getattr(self, name) for name in _CYCLE_FIELDS}
        out["plan"] = self.plan.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Cycle":
        check_keys("cycle", data, _CYCLE_FIELDS)
        kwargs = {name: data[name] for name in _CYCLE_FIELDS if name in data}
        for name in ("id", "state", "iterations", "stagnation_k"):
            kwargs.setdefault(name, None)
        plan = kwargs.get("plan")
        kwargs["plan"] = ShortPlan.from_dict(plan) if isinstance(plan, dict) else plan
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Cycle":
        return cls.from_dict(loads_object("cycle", text))


@dataclass(frozen=True)
class HoldoutSession:
    """The one window per cycle in which the holdout can be read (design section 6)."""

    cycle_id: str
    opened_at: str
    closed_at: str | None = None

    def __post_init__(self) -> None:
        require_text("cycle_id", self.cycle_id)
        require_text("opened_at", self.opened_at)
        if self.closed_at is not None:
            require_text("closed_at", self.closed_at)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


@dataclass(frozen=True)
class HoldoutRead:
    """One logged read of the holdout: which attempt, how many rows, when."""

    cycle_id: str
    attempt_id: str
    rows: int
    read_at: str

    def __post_init__(self) -> None:
        require_text("cycle_id", self.cycle_id)
        require_text("attempt_id", self.attempt_id)
        require_text("read_at", self.read_at)
        _int("rows", self.rows, 0)


@dataclass(frozen=True)
class ComparisonRecord:
    """One paired comparison against the significance threshold (design section 6)."""

    cycle_id: str
    candidate_attempt_id: str
    reference_attempt_id: str
    difference: float
    threshold: float
    is_real: bool

    def __post_init__(self) -> None:
        for name in ("cycle_id", "candidate_attempt_id", "reference_attempt_id"):
            require_text(name, getattr(self, name))
        object.__setattr__(self, "difference", _finite("difference", self.difference))
        threshold = _finite("threshold", self.threshold)
        if threshold < 0:
            raise ValidationError("threshold", "must not be negative")
        object.__setattr__(self, "threshold", threshold)
        if not isinstance(self.is_real, bool):
            raise ValidationError("is_real", "must be a boolean")


@dataclass(frozen=True)
class SubmissionEntry:
    """A submission sent to Kaggle and its public score (design sections 7 and 10).

    Recorded when the submission is sent, with ``status`` as Kaggle reports it
    (``pending``), and updated once when the scoring ends: ``completed_at``
    is None until then and ``public_score`` is None unless it completed.
    No credential field exists: the registry never holds one.
    """

    cycle_id: str
    competition: str
    submission_id: str
    file_sha256: str
    message: str
    submitted_at: str
    status: str
    public_score: float | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "cycle_id", "competition", "submission_id", "file_sha256", "submitted_at", "status",
        ):
            require_text(name, getattr(self, name))
        if not isinstance(self.message, str):
            raise ValidationError("message", "must be text")
        if self.public_score is not None:
            object.__setattr__(self, "public_score", _finite("public_score", self.public_score))
        if self.completed_at is not None:
            require_text("completed_at", self.completed_at)


@dataclass(frozen=True)
class LlmCallEntry:
    """Consumption of one invocation of the language model CLI (design sections 7 and 11).

    Holds only what the budget needs to read back after a suspension: no
    prompt, no response and no access token. ``rejection_reason`` is the
    sanitized validation reason of an ``invalid_output`` call. Token counts are None when the
    CLI did not report them; ``model`` is None when it is not known.
    """

    cycle_id: str
    purpose: str
    outcome: str
    started_at: str
    duration_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempt: int = 1
    model: str | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("cycle_id", "purpose", "started_at"):
            require_text(name, getattr(self, name))
        if self.outcome not in LLM_OUTCOMES:
            raise ValidationError("outcome", f"must be one of {', '.join(LLM_OUTCOMES)}")
        duration = _finite("duration_seconds", self.duration_seconds)
        if duration < 0:
            raise ValidationError("duration_seconds", "must not be negative")
        object.__setattr__(self, "duration_seconds", duration)
        for name in ("input_tokens", "output_tokens"):
            if getattr(self, name) is not None:
                _int(name, getattr(self, name), 0)
        _int("attempt", self.attempt, 1)
        if self.model is not None:
            require_text("model", self.model)
        if self.rejection_reason is not None:
            require_text("rejection_reason", self.rejection_reason)
            if self.outcome != "invalid_output":
                raise ValidationError(
                    "rejection_reason", "is only allowed for the invalid_output outcome"
                )
