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


def test_cycle_protocol_hash_optional_and_serialized():
    assert make_cycle().protocol_hash is None
    cycle = make_cycle(protocol_hash="abc")
    assert Cycle.from_json(cycle.to_json()) == cycle
    assert cycle.to_dict()["protocol_hash"] == "abc"
    with pytest.raises(ValidationError):
        make_cycle(protocol_hash="")


def test_holdout_and_comparison_records_validated():
    from kaggle_agent.registry import ComparisonRecord, HoldoutRead, HoldoutSession

    assert HoldoutSession("C1", "t0").is_open
    assert not HoldoutSession("C1", "t0", "t1").is_open
    assert HoldoutRead("C1", "A1", 10, "t").rows == 10
    with pytest.raises(ValidationError):
        HoldoutRead("C1", "A1", -1, "t")
    with pytest.raises(ValidationError):
        ComparisonRecord("C1", "A1", "A0", 0.1, -1.0, True)
    with pytest.raises(ValidationError):
        ComparisonRecord("C1", "A1", "A0", 0.1, 0.0, "yes")


def test_submission_entry_validation_and_no_credential_field():
    from kaggle_agent.registry import SubmissionEntry

    entry = SubmissionEntry("C1", "comp", "s1", "h" * 64, "msg", "2026-03-10T12:00:00+00:00", "pending")
    assert entry.public_score is None and entry.completed_at is None
    fields = {f.name for f in dataclasses.fields(SubmissionEntry)}
    assert not {f for f in fields if any(w in f for w in ("token", "key", "secret", "credential"))}
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.status = "complete"
    for bad in ({"cycle_id": ""}, {"status": " "}, {"public_score": float("nan")}, {"message": None}):
        data = {**dataclasses.asdict(entry), **bad}
        with pytest.raises(ValidationError):
            SubmissionEntry(**data)


def test_llm_call_entry_validates_and_has_no_secret_fields():
    from kaggle_agent.registry import LLM_OUTCOMES, LlmCallEntry

    entry = LlmCallEntry("C1", "plan", "ok", "2026-01-01T00:00:00+00:00", 1.0)
    assert entry.attempt == 1 and entry.model is None and entry.input_tokens is None
    assert set(LLM_OUTCOMES) == {"ok", "rate_limited", "quota", "cli_error", "invalid_output"}
    names = {f.name for f in dataclasses.fields(LlmCallEntry)}
    assert not names & {"prompt", "response", "text", "token", "access_token"}
    with pytest.raises(ValidationError):
        LlmCallEntry("C1", "plan", "bogus", "t", 1.0)
    with pytest.raises(ValidationError):
        LlmCallEntry("C1", "plan", "ok", "t", float("nan"))


def test_llm_call_entry_rejection_reason_validation():
    from kaggle_agent.registry import LlmCallEntry

    def make(outcome, reason):
        return LlmCallEntry("C1", "plan", outcome, "t", 1.0, rejection_reason=reason)

    assert make("invalid_output", "models: must not be empty").rejection_reason
    assert make("ok", None).rejection_reason is None
    for outcome, reason in (("invalid_output", ""), ("invalid_output", "  "), ("ok", "x: y")):
        with pytest.raises(ValidationError):
            make(outcome, reason)
