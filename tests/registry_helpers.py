"""Builders shared by the registry tests."""

from kaggle_agent.proposals import Attempt, ShortPlan
from kaggle_agent.registry import Cycle, ExecutionRecord, PartitionMetric

PLAN = ShortPlan(
    metric="m",
    validation="v",
    feature_engineering=["f"],
    models=["x"],
    baseline="b",
)


def make_cycle(cycle_id="C1", **kwargs):
    params = {"iterations": 10, "stagnation_k": 3, "state": "ITERATING", **kwargs}
    return Cycle(id=cycle_id, plan=PLAN, **params)


def make_attempt(attempt_id="A1", **kwargs):
    data = {
        "id": attempt_id,
        "phase": "iteration",
        "hypothesis": "h",
        "rationale": "r",
        "what_changed": f"change {attempt_id}",
        "measure": "m",
        "exploratory": True,
        **kwargs,
    }
    return Attempt(**data)


def make_execution(attempt_id="A1", exec_id=None, outcome="succeeded", **kwargs):
    data = {
        "id": exec_id or f"{attempt_id}-E",
        "attempt_id": attempt_id,
        "config": {"lr": 0.1},
        "seed": 7,
        "code_hash": "code1",
        "data_hash": "data1",
        "protocol_hash": "proto1",
        "metrics": [PartitionMetric("validation", 0.8, 0.02)],
        "duration_seconds": 1.5,
        "outcome": outcome,
        **kwargs,
    }
    if outcome != "succeeded":
        data["metrics"] = kwargs.get("metrics", ())
        data.setdefault("failure_kind", "code_error")
        data.setdefault("reason", "Traceback: boom")
    return ExecutionRecord(**data)


def crash(attempt_id="A1", n=1):
    """A run that crashed, with a unique id."""
    return make_execution(attempt_id, f"{attempt_id}-crash{n}", "failed", code_hash=f"fix{n}")


def new_attempt(reg, attempt_id, cycle_id="C1", fingerprint=None, **kwargs):
    reg.add_attempt(cycle_id, make_attempt(attempt_id, **kwargs), fingerprint or f"fp-{attempt_id}")


def close_after_cap(reg, attempt_id, cap=3):
    """Run the attempt until it crashed 1 + cap times, then close it."""
    for n in range(1 + cap):
        reg.add_execution(crash(attempt_id, n))
    reg.close_attempt(attempt_id, "gave up")
