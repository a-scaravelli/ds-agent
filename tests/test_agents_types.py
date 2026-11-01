"""New summary, input and output types validate and round-trip (case 20)."""

import json

import pytest

from kaggle_agent.agents import (
    AttemptDigest,
    CandidateResult,
    ColumnSummary,
    CoordinatorInput,
    CoordinatorOutput,
    DataFindings,
    DataInput,
    DataSummary,
    FeatureEngineeringInput,
    MetricValue,
    ModelingInput,
    RegistrySummary,
    ReviewerInput,
    ReviewerNotes,
)
from kaggle_agent.proposals import ValidationError
import agents_helpers as h

ROUND_TRIP = [
    h.data_summary(),
    h.history(),
    h.findings(),
    ColumnSummary("a", "float64", 0.0, 3),
    AttemptDigest("A1", "h", "failed"),
    MetricValue("accuracy", 0.5),
    h.coordinator_input(),
    h.data_input(),
    h.feature_engineering_input(),
    h.modeling_input(),
    h.reviewer_input(),
    CandidateResult("A1", "h", "ok"),
    CoordinatorOutput.from_json(h.coordinator_json()),
    ReviewerNotes("text"),
]


@pytest.mark.parametrize("value", ROUND_TRIP, ids=lambda v: type(v).__name__)
def test_types_round_trip_identically(value):
    cls = type(value)
    assert cls.from_json(value.to_json()) == value
    assert cls.from_dict(value.to_dict()) == value
    assert json.loads(value.to_json()) == json.loads(json.dumps(value.to_dict()))


@pytest.mark.parametrize(
    "build",
    [
        lambda: ColumnSummary("", "int", 0.0, 1),
        lambda: ColumnSummary("a", "int", 1.5, 1),
        lambda: ColumnSummary("a", "int", 0.0, -1),
        lambda: ColumnSummary("a", "int", True, 1),
        lambda: DataSummary(-1, "y", ()),
        lambda: DataSummary(1, "y", ("not a column",)),
        lambda: DataSummary(1, "y", (), ("",)),
        lambda: AttemptDigest("A", "h", "ok", float("nan")),
        lambda: RegistrySummary(best=("x",)),
        lambda: DataFindings(""),
        lambda: DataFindings("p", quality_issues=("",)),
        lambda: ReviewerNotes(""),
        lambda: ReviewerNotes.from_dict({"text": "t", "accept": True}),
        lambda: ReviewerNotes.from_json("not json"),
        lambda: ReviewerNotes.from_json("[]"),
        lambda: DataFindings.from_dict({"profile": "p", "labels": []}),
        lambda: CoordinatorOutput.from_dict({"tasks": ["t"]}),
        lambda: CoordinatorOutput.from_dict({"plan": h.plan().to_dict(), "tasks": []}),
        lambda: CoordinatorOutput.from_dict({"plan": h.plan().to_dict(), "tasks": ["t"], "x": 1}),
        lambda: DataInput.from_dict({"summary": h.data_summary().to_dict(), "task": ""}),
        lambda: DataInput.from_dict({"task": "t"}),
        lambda: FeatureEngineeringInput.from_dict({**h.feature_engineering_input().to_dict(), "candidate_id": ""}),
        lambda: ModelingInput.from_dict({**h.modeling_input().to_dict(), "previous_attempt": None}),
        lambda: CoordinatorInput.from_dict({**h.coordinator_input().to_dict(), "metric": " "}),
        lambda: ReviewerInput.from_dict({**h.reviewer_input().to_dict(), "round_id": ""}),
        lambda: CandidateResult("A", "h", "ok", (1,)),
        lambda: MetricValue("m", "x"),
    ],
)
def test_invalid_values_raise_validation_error(build):
    with pytest.raises(ValidationError):
        build()
