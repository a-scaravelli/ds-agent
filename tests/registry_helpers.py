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


T0 = "2026-10-30T10:00:00+00:00"


def make_trace(cycle_id="C1", role="modeling", round_id="R1", **kwargs):
    from kaggle_agent.registry import RoleTrace

    data = {
        "cycle_id": cycle_id,
        "role": role,
        "round_id": round_id,
        "input_type": "Round",
        "input": {"tasks": ["try a tree model"]},
        "output_type": "Attempt",
        "output": {"id": "A1"},
        "outcome": "ok",
        "created_at": T0,
        **kwargs,
    }
    return RoleTrace(**data)


def add_round(reg, round_id="R1", cycle_id="C1", tasks=("task",), candidates=()):
    from kaggle_agent.proposals import Round

    return reg.add_round(cycle_id, Round(round_id, tuple(tasks), tuple(candidates)))
