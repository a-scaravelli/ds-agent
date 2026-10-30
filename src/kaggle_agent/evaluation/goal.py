"""Goal criteria met or not met, from observed metric values (design sections 11 and 14).

A pure function: it reads numbers, never a candidate, so the same observed
values always give the same status.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from kaggle_agent.proposals import GoalCriteria, ValidationError


@dataclass(frozen=True)
class CriterionResult:
    """One criterion against its observed value."""

    metric: str
    value: float
    threshold: float
    direction: str
    met: bool


@dataclass(frozen=True)
class GoalStatus:
    """``met`` is true only when every criterion is met."""

    met: bool
    results: tuple[CriterionResult, ...]


def evaluate_goal(goal: GoalCriteria, observed: Mapping[str, float]) -> GoalStatus:
    """Compare each criterion with the observed value of its metric.

    ``maximize`` needs value >= threshold and ``minimize`` value <= threshold,
    so a value equal to the threshold is met. A metric missing from
    ``observed`` raises ``ValidationError``.
    """
    results = []
    for criterion in goal.criteria:
        if criterion.metric not in observed:
            raise ValidationError(
                "observed", f"no value for goal metric {criterion.metric!r}"
            )
        value = float(observed[criterion.metric])
        if criterion.direction == "maximize":
            met = value >= criterion.threshold
        else:
            met = value <= criterion.threshold
        results.append(
            CriterionResult(criterion.metric, value, criterion.threshold, criterion.direction, met)
        )
    return GoalStatus(all(r.met for r in results), tuple(results))
