"""Feature engineering role: proposes the transformations of a candidate as an ``Attempt``.

One of the two roles with a candidate: the result carries a ``PendingTrace``
that the caller writes with ``record_trace`` after registering the attempt, so
the trace links to the candidate. A failed or rejected call is traced at once.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from kaggle_agent.llm import LlmClient, ToolPermissions
from kaggle_agent.proposals import Attempt, CarryOver, ShortPlan, Steering, TryThis
from kaggle_agent.proposals.errors import check_keys, require_text

from .call import PURPOSES, Registry, CandidateRoleResult, TraceContext, utc_now, invoke_role
from .prompting import render_prompt
from .summaries import RunLimits, DataFindings, JsonObject, RegistrySummary, object_from

ROLE = "feature_engineering"

INSTRUCTIONS = """\
You are the feature engineering specialist of a data science team. From the
plan, the findings of the data role, the task of the round and the history of
the attempts, propose one falsifiable hypothesis about the features: what you
change, why, the evidence you have (or mark the attempt exploratory) and how to
measure it. Leave bundle_id out. The id of the attempt is assigned by the fixed
code: any id you write is replaced. You only see summaries, never rows, labels
or holdout data."""

LIMITS_NOTE = """\
Run limits (the "limits" object of the input): stay within the candidates per
round and the current round, and propose only features that use the libraries
in allowed_libraries; anything else not listed (torch, tensorflow) is not
allowed."""


@dataclass(frozen=True)
class FeatureEngineeringInput(JsonObject):
    """Plan, data findings, task, history and the id the candidate will have."""

    LABEL: ClassVar[str] = "feature engineering input"
    candidate_id: str
    plan: ShortPlan
    findings: DataFindings
    task: str
    history: RegistrySummary
    limits: RunLimits | None = None
    steering: Steering | None = None
    carry: CarryOver | None = None
    request: TryThis | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = ("candidate_id", "plan", "findings", "task", "history", "limits", "steering", "carry", "request")

    def __post_init__(self) -> None:
        require_text("candidate_id", self.candidate_id)
        if not isinstance(self.plan, ShortPlan):
            raise ValueError("plan must be a ShortPlan")
        if not isinstance(self.findings, DataFindings):
            raise ValueError("findings must be DataFindings")
        require_text("task", self.task)
        if not isinstance(self.history, RegistrySummary):
            raise ValueError("history must be a RegistrySummary")
        if self.limits is not None and not isinstance(self.limits, RunLimits):
            raise ValueError("limits must be RunLimits or None")
        for name, kind in (("steering", Steering), ("carry", CarryOver), ("request", TryThis)):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), kind):
                raise ValueError(f"{name} must be {kind.__name__} or None")

    def to_dict(self) -> dict:
        data = {
            "candidate_id": self.candidate_id,
            "plan": self.plan.to_dict(),
            "findings": self.findings.to_dict(),
            "task": self.task,
            "history": self.history.to_dict(),
        }
        if self.limits is not None:
            data["limits"] = self.limits.to_dict()
        for name in ("steering", "carry", "request"):
            if getattr(self, name) is not None:
                data[name] = getattr(self, name).to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureEngineeringInput":
        check_keys("feature engineering input", data, cls._FIELDS)
        return cls(
            candidate_id=data.get("candidate_id"),
            plan=object_from("plan", data.get("plan"), ShortPlan),
            findings=object_from("findings", data.get("findings"), DataFindings),
            task=data.get("task"),
            history=object_from("history", data.get("history"), RegistrySummary),
            limits=None if data.get("limits") is None else object_from("limits", data["limits"], RunLimits),
            steering=None if data.get("steering") is None else Steering.from_dict(data["steering"]),
            carry=None if data.get("carry") is None else CarryOver.from_dict(data["carry"]),
            request=None if data.get("request") is None else TryThis.from_dict(data["request"]),
        )


def build_feature_engineering_prompt(role_input: FeatureEngineeringInput) -> str:
    """The prompt of the feature engineering role: instructions and the serialized input."""
    return render_prompt(INSTRUCTIONS, role_input, LIMITS_NOTE, libraries=True, describe=True)


def run_feature_engineering(
    client: LlmClient,
    registry: Registry,
    context: TraceContext,
    role_input: FeatureEngineeringInput,
    permissions: ToolPermissions,
    model: str | None = None,
    now: Callable[[], datetime] = utc_now,
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
        build_feature_engineering_prompt(role_input), Attempt, role_input,
        permissions, model=model, finalize=finalize, now=now,
    )
    return CandidateRoleResult(invoked.value, invoked.trace)
