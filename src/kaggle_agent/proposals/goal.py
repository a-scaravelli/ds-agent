"""Goal criteria: named thresholds on metrics, used as a stop condition (design section 11)."""

import json
import math
from dataclasses import dataclass

from .errors import ValidationError, check_keys, loads_object, require_text

DIRECTIONS = ("maximize", "minimize")
_CRITERION_FIELDS = ("metric", "threshold", "direction")
_CRITERIA_FIELDS = ("criteria",)


@dataclass(frozen=True)
class GoalCriterion:
    """One threshold on a metric: ``maximize`` means value >= threshold, ``minimize`` value <= threshold.

    The name is any metric name; the proposals package does not know which
    metrics exist, the evaluation engine checks that.
    """

    metric: str
    threshold: float
    direction: str

    def __post_init__(self) -> None:
        require_text("metric", self.metric)
        value = self.threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError("threshold", "must be a number")
        if not math.isfinite(value):
            raise ValidationError("threshold", "must be finite")
        object.__setattr__(self, "threshold", float(value))
        if self.direction not in DIRECTIONS:
            raise ValidationError("direction", f"must be one of {', '.join(DIRECTIONS)}")

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _CRITERION_FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "GoalCriterion":
        check_keys("criterion", data, _CRITERION_FIELDS)
        return cls(**{name: data.get(name) for name in _CRITERION_FIELDS})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "GoalCriterion":
        return cls.from_dict(loads_object("criterion", text))


@dataclass(frozen=True)
class GoalCriteria:
    """One or more criteria on distinct metrics, in the order given."""

    criteria: tuple[GoalCriterion, ...]

    def __post_init__(self) -> None:
        value = self.criteria
        if value is None:
            raise ValidationError("criteria", "is required")
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValidationError("criteria", "must be a list")
        if not value:
            raise ValidationError("criteria", "must not be empty")
        items = tuple(
            c if isinstance(c, GoalCriterion) else _criterion_from(c) for c in value
        )
        names = [c.metric for c in items]
        if len(set(names)) != len(names):
            raise ValidationError("criteria", "metrics must be unique")
        object.__setattr__(self, "criteria", items)

    @property
    def metrics(self) -> tuple[str, ...]:
        return tuple(c.metric for c in self.criteria)

    def to_dict(self) -> dict:
        return {"criteria": [c.to_dict() for c in self.criteria]}

    @classmethod
    def from_dict(cls, data: dict) -> "GoalCriteria":
        check_keys("goal", data, _CRITERIA_FIELDS)
        return cls(criteria=data.get("criteria"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "GoalCriteria":
        return cls.from_dict(loads_object("goal", text))


def _criterion_from(value: object) -> GoalCriterion:
    if not isinstance(value, dict):
        raise ValidationError("criteria", "each criterion must be an object")
    return GoalCriterion.from_dict(value)
