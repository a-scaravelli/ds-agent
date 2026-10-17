"""Registry of metrics, identified by name and never supplied as callables (design section 5).

Each metric declares its natural direction. The protocol must state its own
direction explicitly; the registry only checks that the two agree. There is no
default metric: the agent derives it from the competition and freezes it.
"""

from dataclasses import dataclass
from typing import Callable

from sklearn import metrics as skm

from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import require_text

DIRECTIONS = ("maximize", "minimize")
INPUT_KINDS = ("values", "labels", "scores", "probabilities")


@dataclass(frozen=True)
class MetricSpec:
    """A metric building block: function, expected prediction kind and natural direction."""

    name: str
    function: Callable
    input_kind: str
    direction: str
    allowed_params: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text("name", self.name)
        if not callable(self.function):
            raise ValidationError("function", "must be callable")
        if self.input_kind not in INPUT_KINDS:
            raise ValidationError("input_kind", f"must be one of {', '.join(INPUT_KINDS)}")
        if self.direction not in DIRECTIONS:
            raise ValidationError(
                "direction", f"is required and must be one of {', '.join(DIRECTIONS)}"
            )

    def validate_params(self, params: object) -> dict:
        if not isinstance(params, dict):
            raise ValidationError("metric_params", "must be an object")
        for key in params:
            if key not in self.allowed_params:
                allowed = ", ".join(self.allowed_params) or "none"
                raise ValidationError(
                    "metric_params", f"unknown parameter {key!r} for {self.name} (allowed: {allowed})"
                )
        return params

    def score(self, y_true, y_pred, params: dict) -> float:
        return float(self.function(y_true, y_pred, **params))


METRICS: dict[str, MetricSpec] = {}


def register_metric(spec: MetricSpec) -> None:
    """Add a metric to the registry; a spec without a valid direction is refused."""
    if not isinstance(spec, MetricSpec):
        raise ValidationError("metric", "must be a MetricSpec")
    if spec.name in METRICS:
        raise ValidationError("metric", f"{spec.name!r} is already registered")
    METRICS[spec.name] = spec


def get_metric(name: object) -> MetricSpec:
    if not isinstance(name, str):
        raise ValidationError("metric", "must be the name of a registered metric")
    if name not in METRICS:
        raise ValidationError(
            "metric", f"unknown metric {name!r} (registered: {', '.join(sorted(METRICS))})"
        )
    return METRICS[name]


def _f1(y_true, y_pred, average="binary"):
    return skm.f1_score(y_true, y_pred, average=average)


for _spec in (
    MetricSpec("rmse", skm.root_mean_squared_error, "values", "minimize"),
    MetricSpec("mae", skm.mean_absolute_error, "values", "minimize"),
    MetricSpec("r2", skm.r2_score, "values", "maximize"),
    MetricSpec("accuracy", skm.accuracy_score, "labels", "maximize"),
    MetricSpec("balanced_accuracy", skm.balanced_accuracy_score, "labels", "maximize"),
    MetricSpec("f1", _f1, "labels", "maximize", ("average",)),
    MetricSpec("log_loss", skm.log_loss, "probabilities", "minimize"),
    MetricSpec("roc_auc", skm.roc_auc_score, "scores", "maximize"),
    MetricSpec("average_precision", skm.average_precision_score, "scores", "maximize"),
):
    register_metric(_spec)
