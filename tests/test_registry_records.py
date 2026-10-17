"""Records: execution record, partition metric and cycle (no network)."""

import dataclasses

import pytest
from registry_helpers import PLAN, make_cycle, make_execution

from kaggle_agent.proposals import Attempt, ValidationError
from kaggle_agent.registry import (
    FAILURE_KINDS,
    Cycle,
    ExecutionRecord,
    NotFoundError,
    PartitionMetric,
    SqliteRegistry,
)


def test_execution_record_is_separate_type_linked_by_id():
    fields = {f.name for f in dataclasses.fields(Attempt)}
    for name in ("config", "seed", "code_hash", "data_hash", "protocol_hash", "metrics"):
        assert name not in fields
    assert make_execution().attempt_id == "A1"
    with SqliteRegistry() as reg, pytest.raises(NotFoundError):
        reg.add_execution(make_execution("missing"))


def test_failure_kind_required_unless_succeeded():
    for outcome in ("failed", "inconclusive"):
        with pytest.raises(ValidationError) as exc:
            make_execution(outcome=outcome, failure_kind=None)
        assert exc.value.field == "failure_kind"
    with pytest.raises(ValidationError) as exc:
        make_execution(failure_kind="code_error")
    assert exc.value.field == "failure_kind"
    assert make_execution().failure_kind is None


def test_failure_kind_values_and_reason():
    for bad in ({"failure_kind": "oops"}, {"reason": ""}, {"reason": None}):
        with pytest.raises(ValidationError) as exc:
            make_execution(outcome="failed", **bad)
        assert exc.value.field in ("failure_kind", "reason")
        assert exc.value.field in bad
    assert FAILURE_KINDS == ("code_error", "hypothesis_falsified", "timeout", "below_threshold")
    for kind in FAILURE_KINDS:
        record = make_execution(outcome="failed", failure_kind=kind, reason=f"why {kind}")
        again = ExecutionRecord.from_json(record.to_json())
        assert (again.failure_kind, again.reason) == (kind, f"why {kind}")


def test_invalid_execution_data_rejected():
    bad = [
        ({"outcome": "done"}, "outcome"),
        ({"duration_seconds": -1}, "duration_seconds"),
        ({"metrics": [PartitionMetric("v", 0.5, 0.1)], "duration_seconds": float("nan")}, "duration_seconds"),
        ({"code_hash": ""}, "code_hash"),
        ({"data_hash": " "}, "data_hash"),
        ({"protocol_hash": None}, "protocol_hash"),
        ({"metrics": ()}, "metrics"),
        ({"seed": "7"}, "seed"),
        ({"config": {"x": object()}}, "config"),
    ]
    for kwargs, field in bad:
        with pytest.raises(ValidationError) as exc:
            make_execution(**kwargs)
        assert exc.value.field == field, kwargs
    for mean in (float("nan"), float("inf")):
        with pytest.raises(ValidationError) as exc:
            PartitionMetric("validation", mean, 0.1)
        assert exc.value.field == "mean"
    with pytest.raises(ValidationError) as exc:
        PartitionMetric("validation", 0.5, -0.1)
    assert exc.value.field == "dispersion"


def test_records_serialization_round_trip():
    record = make_execution(resources={"cpu": 2}, artifacts=["diff.patch"], reason="ok")
    metric = PartitionMetric("holdout", 0.7, 0.01)
    cycle = make_cycle(max_fix_retries=2)
    for obj in (record, metric, cycle):
        assert type(obj).from_json(obj.to_json()) == obj
        assert type(obj).from_dict(obj.to_dict()) == obj
    assert cycle.to_dict()["plan"] == PLAN.to_dict()
    for cls, obj in ((ExecutionRecord, record), (PartitionMetric, metric), (Cycle, cycle)):
        with pytest.raises(ValidationError):
            cls.from_dict({**obj.to_dict(), "extra": 1})
        with pytest.raises(ValidationError):
            cls.from_json("{not json")


def test_cycle_parameters_validated():
    assert make_cycle().max_fix_retries == 3
    for kwargs, field in (
        ({"iterations": 0}, "iterations"),
        ({"stagnation_k": 0}, "stagnation_k"),
        ({"max_fix_retries": -1}, "max_fix_retries"),
        ({"max_fix_retries": 1.5}, "max_fix_retries"),
        ({"state": ""}, "state"),
    ):
        with pytest.raises(ValidationError) as exc:
            make_cycle(**kwargs)
        assert exc.value.field == field
