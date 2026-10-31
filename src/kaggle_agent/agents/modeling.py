"""Modeling role: proposes the model and hyperparameters of a candidate as an ``Attempt``.

Like feature engineering, it has a candidate: the result carries a
``PendingTrace`` for the caller to write with ``record_trace`` after the
attempt is registered. It receives the previous role's ``Attempt`` as plain
data passed by the caller; it does not import or call that role.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from kaggle_agent.llm import LlmClient, ToolPermissions
from kaggle_agent.proposals import Attempt, ShortPlan
from kaggle_agent.proposals.errors import check_keys, require_text
from kaggle_agent.registry import Registry

from .call import PURPOSES, CandidateRoleResult, TraceContext, _utc_now, invoke_role
from .prompting import render_prompt
from .summaries import DataFindings, JsonObject, RegistrySummary, object_from

ROLE = "modeling"

INSTRUCTIONS = """\
You are the modeling specialist of a data science team. From the plan, the
findings of the data role, the attempt of the feature engineering role, the task
of the round and the history of the attempts, propose one falsifiable hypothesis
about the model and its hyperparameters: what you change, why, the evidence you
have (or mark the attempt exploratory) and how to measure it. Leave bundle_id
out. The id of the attempt is assigned by the fixed code: any id you write is
replaced. You only see summaries, never rows, labels or holdout data."""


@dataclass(frozen=True)
class ModelingInput(JsonObject):
    """Plan, data findings, the feature engineering attempt, task, history and candidate id."""

    LABEL: ClassVar[str] = "modeling input"
    candidate_id: str
    plan: ShortPlan
    findings: DataFindings
    task: str
    history: RegistrySummary
    previous_attempt: Attempt

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "candidate_id", "plan", "findings", "task", "history", "previous_attempt",
    )

    def __post_init__(self) -> None:
        require_text("candidate_id", self.candidate_id)
        if not isinstance(self.plan, ShortPlan):
            raise ValueError("plan must be a ShortPlan")
        if not isinstance(self.findings, DataFindings):
            raise ValueError("findings must be DataFindings")
        require_text("task", self.task)
        if not isinstance(self.history, RegistrySummary):
            raise ValueError("history must be a RegistrySummary")
        if not isinstance(self.previous_attempt, Attempt):
            raise ValueError("previous_attempt must be an Attempt")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "plan": self.plan.to_dict(),
            "findings": self.findings.to_dict(),
            "task": self.task,
            "history": self.history.to_dict(),
            "previous_attempt": self.previous_attempt.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelingInput":
        check_keys("modeling input", data, cls._FIELDS)
        return cls(
            candidate_id=data.get("candidate_id"),
            plan=object_from("plan", data.get("plan"), ShortPlan),
            findings=object_from("findings", data.get("findings"), DataFindings),
            task=data.get("task"),
            history=object_from("history", data.get("history"), RegistrySummary),
            previous_attempt=object_from("previous_attempt", data.get("previous_attempt"), Attempt),
        )


def build_modeling_prompt(role_input: ModelingInput) -> str:
    """The prompt of the modeling role: instructions and the serialized input."""
    return render_prompt(INSTRUCTIONS, role_input)


def run_modeling(
    client: LlmClient,
    registry: Registry,
    context: TraceContext,
    role_input: ModelingInput,
    permissions: ToolPermissions,
    model: str | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> CandidateRoleResult:
    """Ask the role for an ``Attempt`` whose id is ``role_input.candidate_id``.

    The trace is NOT written: register the attempt, then call
    ``record_trace(registry, result.trace, candidate_id=attempt.id)``. A failed
    or rejected call is already traced (no candidate, no output) and raises as
    in ``run_coordinator``.
    """

    def finalize(attempt: Attempt) -> Attempt:
        return dataclasses.replace(attempt, id=role_input.candidate_id)

    invoked = invoke_role(
        client, registry, context, ROLE, PURPOSES[ROLE],
        build_modeling_prompt(role_input), Attempt, role_input,
        permissions, model=model, finalize=finalize, now=now,
    )
    return CandidateRoleResult(invoked.value, invoked.trace)
