"""The evaluation protocol: everything the agent proposes with the plan and the code freezes.

Every field is mandatory. There is no default metric, direction or scheme: a
protocol that leaves one out, or names something not registered, is refused.
"""

import json
import math
from dataclasses import dataclass

from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import check_keys, loads_object, require_text

from .metrics import DIRECTIONS, get_metric
from .schemes import get_scheme

_FIELDS = (
    "metric",
    "metric_params",
    "direction",
    "target",
    "validation_scheme",
    "validation_params",
    "holdout_scheme",
    "holdout_params",
    "holdout_share",
    "seed",
    "repetitions",
    "threshold_k",
)
MIN_PARTITIONS = 2


def _json_object(field: str, value: object) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(field, "must be an object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(field, f"must be JSON-compatible: {exc}") from exc
    return value


def _number(field: str, value: object) -> float:
    if value is None:
        raise ValidationError(field, "is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(field, "must be a number")
    if not math.isfinite(value):
        raise ValidationError(field, "must be finite")
    return value


@dataclass(frozen=True)
class Protocol:
    """Metric, validation scheme, holdout, seed, repetitions and threshold factor.

    ``threshold_k`` is the factor of the significance threshold: an improvement
    is real only if it exceeds ``threshold_k`` times the dispersion of the
    baseline across partitions (design section 6).
    """

    metric: str
    metric_params: dict
    direction: str
    target: str
    validation_scheme: str
    validation_params: dict
    holdout_scheme: str
    holdout_params: dict
    holdout_share: float
    seed: int
    repetitions: int
    threshold_k: float

    def __post_init__(self) -> None:
        for name in ("metric", "direction", "target", "validation_scheme", "holdout_scheme"):
            value = getattr(self, name)
            if value is None:
                raise ValidationError(name, "is required")
            require_text(name, value)
        for name in ("metric_params", "validation_params", "holdout_params"):
            _json_object(name, getattr(self, name))
        spec = get_metric(self.metric)
        spec.validate_params(self.metric_params)
        if self.direction not in DIRECTIONS:
            raise ValidationError("direction", f"must be one of {', '.join(DIRECTIONS)}")
        if self.direction != spec.direction:
            raise ValidationError(
                "direction",
                f"metric {self.metric!r} is naturally {spec.direction}; "
                f"got {self.direction!r} (register a separate metric for a different direction)",
            )
        validation = get_scheme(self.validation_scheme, "validation_scheme")
        validation.validate_params(self.validation_params, "validation")
        get_scheme(self.holdout_scheme, "holdout_scheme").validate_params(
            self.holdout_params, "holdout"
        )
        share = _number("holdout_share", self.holdout_share)
        if not 0 < share < 1:
            raise ValidationError("holdout_share", "must be between 0 and 1, exclusive")
        for name, minimum in (("seed", 0), ("repetitions", 1)):
            value = getattr(self, name)
            if value is None:
                raise ValidationError(name, "is required")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(name, "must be an integer")
            if value < minimum:
                raise ValidationError(name, f"must be at least {minimum}")
        if _number("threshold_k", self.threshold_k) < 0:
            raise ValidationError("threshold_k", "must not be negative")
        total = self.repetitions * validation.n_folds(self.validation_params)
        if total < MIN_PARTITIONS:
            raise ValidationError(
                "repetitions",
                f"repetitions x folds gives {total} partitions; the dispersion of the "
                f"baseline needs at least {MIN_PARTITIONS}",
            )

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "Protocol":
        check_keys("protocol", data, _FIELDS)
        return cls(**{name: data.get(name) for name in _FIELDS})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Protocol":
        return cls.from_dict(loads_object("protocol", text))
