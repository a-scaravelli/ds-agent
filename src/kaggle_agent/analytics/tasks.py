"""Which kind of task a metric is, and which panels its predictions allow."""

from kaggle_agent.evaluation.metrics import get_metric

MAX_CLASSES = 10


def task_of(metric: str) -> dict:
    """``task`` (classification or regression) and ``input_kind`` of a registered metric."""
    kind = get_metric(metric).input_kind
    return {"task": "regression" if kind == "values" else "classification", "input_kind": kind}


def panel_support(input_kind: str) -> dict:
    """Which classification panels the prediction kind allows (True) or why not (a text)."""
    if input_kind == "labels":
        no = "the metric scores labels only: there are no probabilities or scores"
        return {"confusion": True, "roc": no, "pr": no, "calibration": no, "log_loss": no, "sweep": no}
    if input_kind == "scores":
        no = "the predictions are scores, not probabilities"
        return {"confusion": "the predictions are scores, not labels", "roc": True, "pr": True,
                "calibration": no, "log_loss": no, "sweep": True}
    return {"confusion": True, "roc": True, "pr": True, "calibration": True, "log_loss": True, "sweep": True}
