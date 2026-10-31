"""Builders shared by the agents tests: valid inputs and outputs, registry, permissions, clock."""

import json
from datetime import datetime, timezone

from kaggle_agent.agents import (
    AttemptDigest,
    CandidateResult,
    ColumnSummary,
    CoordinatorInput,
    DataFindings,
    DataInput,
    DataSummary,
    FeatureEngineeringInput,
    MetricValue,
    ModelingInput,
    RegistrySummary,
    ReviewerInput,
    TraceContext,
    run_coordinator,
    run_data,
    run_feature_engineering,
    run_modeling,
    run_reviewer,
)
from kaggle_agent.llm import ToolPermissions
from kaggle_agent.llm.fake import valid_attempt_json, valid_plan_json
from kaggle_agent.proposals import (
    Attempt,
    GoalCriteria,
    GoalCriterion,
    Guidance,
    ReviewerReport,
    ShortPlan,
)
from kaggle_agent.registry import SqliteRegistry
from registry_helpers import add_round, make_cycle

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
CANDIDATE_ID = "A7"
REPORT_ID = "REP1"
ROLES = ("coordinator", "data", "feature_engineering", "modeling", "reviewer")


def fixed_now():
    return NOW


class RecordingRegistry(SqliteRegistry):
    """In-memory registry that remembers what ``add_llm_call`` returned."""

    def __init__(self) -> None:
        super().__init__(":memory:")
        self.llm_call_ids: list[int] = []

    def add_llm_call(self, entry):
        row_id = super().add_llm_call(entry)
        self.llm_call_ids.append(row_id)
        return row_id


def fresh_registry():
    """A new registry with cycle C1 and round R1: ``llm_calls`` rowids start at 1."""
    registry = RecordingRegistry()
    registry.add_cycle(make_cycle("C1"))
    add_round(registry, "R1")
    return registry


def context():
    return TraceContext("C1", "R1")


def permissions(tmp_path):
    return ToolPermissions(working_dir=tmp_path)


def plan():
    return ShortPlan.from_json(valid_plan_json())


def goal():
    return GoalCriteria((GoalCriterion("accuracy", 0.9, "maximize"),))


def data_summary():
    return DataSummary(
        rows=100,
        target="y",
        columns=(ColumnSummary("age", "int64", 0.1, 40), ColumnSummary("y", "int64", 0.0, 2)),
        notes=("target is balanced",),
    )


def history():
    return RegistrySummary(
        best=(AttemptDigest("A1", "scaling helps", "succeeded", 0.81),),
        recent_failures=(AttemptDigest("A2", "deep trees", "failed"),),
    )


def findings():
    return DataFindings("100 rows, 2 columns", ("10% missing age",), (), ("impute age",))


def attempt(attempt_id="A1", **overrides):
    return Attempt.from_json(valid_attempt_json(id=attempt_id, **overrides))


def coordinator_input():
    report = ReviewerReport("REP0", "R0", "first round looked fine", "2026-10-29T10:00:00+00:00")
    lead = Guidance("G1", "R0", "focus on trees", "2026-10-29T11:00:00+00:00")
    return CoordinatorInput(
        data=data_summary(), metric="accuracy", history=history(),
        reports=(report,), guidance=(lead.text,),
    )


def data_input():
    return DataInput(data_summary(), "profile the data")


def feature_engineering_input():
    return FeatureEngineeringInput(CANDIDATE_ID, plan(), findings(), "add features", history())


def modeling_input(previous=None):
    return ModelingInput(
        CANDIDATE_ID, plan(), findings(), "pick a model", history(),
        previous if previous is not None else attempt("FE-1", what_changed="add ratio features"),
    )


def reviewer_input():
    return ReviewerInput(
        "R1", plan(), goal(),
        (CandidateResult("A1", "scaling helps", "succeeded", (MetricValue("accuracy", 0.8),)),),
    )


def coordinator_json():
    return json.dumps(
        {
            "plan": json.loads(valid_plan_json()),
            "tasks": ["try tree models"],
            "goal": {"criteria": [{"metric": "accuracy", "threshold": 0.9, "direction": "maximize"}]},
        }
    )


def data_json():
    return json.dumps(
        {
            "profile": "100 rows, 2 columns",
            "quality_issues": ["10% missing age"],
            "leakage_hints": ["none seen"],
            "proposed_cleaning": ["impute age"],
        }
    )


def reviewer_json(text="the first candidate improved accuracy"):
    return json.dumps({"text": text})


# role -> (input builder, valid reply text)
VALID = {
    "coordinator": (coordinator_input, coordinator_json),
    "data": (data_input, data_json),
    "feature_engineering": (feature_engineering_input, lambda: valid_attempt_json(id="MODEL-ID")),
    "modeling": (modeling_input, lambda: valid_attempt_json(id="MODEL-ID")),
    "reviewer": (reviewer_input, reviewer_json),
}


def valid_reply(role):
    return VALID[role][1]()


def run_role(role, client, registry, tmp_path, model=None):
    """Run one role with its default valid input; returns what ``run_<role>`` returns."""
    role_input = VALID[role][0]()
    args = (client, registry, context(), role_input, permissions(tmp_path))
    kwargs = {"model": model, "now": fixed_now}
    if role == "coordinator":
        return run_coordinator(*args, **kwargs)
    if role == "data":
        return run_data(*args, **kwargs)
    if role == "feature_engineering":
        return run_feature_engineering(*args, **kwargs)
    if role == "modeling":
        return run_modeling(*args, **kwargs)
    return run_reviewer(*args, report_id=REPORT_ID, **kwargs)


def role_input_for(role):
    return VALID[role][0]()


CANDIDATE_ROLES = ("feature_engineering", "modeling")


def settle(role, result, registry):
    """Finish a successful call: candidate roles register the attempt, then write the trace.

    Returns the typed output of the role.
    """
    from kaggle_agent.agents import record_trace
    from registry_helpers import new_attempt

    if role not in CANDIDATE_ROLES:
        return result
    attempt_out = result.output
    new_attempt(registry, attempt_out.id, what_changed=attempt_out.what_changed)
    record_trace(registry, result.trace, candidate_id=attempt_out.id)
    return attempt_out
