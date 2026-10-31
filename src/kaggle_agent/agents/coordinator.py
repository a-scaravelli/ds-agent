"""Coordinator role: reads the problem and the history, writes the plan and directs the round.

Typed input (summaries only), typed output (``CoordinatorOutput``, built from
the proposal types) and one call on ``LlmClient``. The role does not call or
import any other role; the caller passes its output on.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from kaggle_agent.llm import LlmClient, ToolPermissions
from kaggle_agent.proposals import GoalCriteria, ReviewerReport, ShortPlan
from kaggle_agent.proposals.errors import check_keys, require_text, require_text_list

from .call import PURPOSES, Registry, TraceContext, utc_now, call_role
from .prompting import render_prompt
from .summaries import DataSummary, JsonObject, RegistrySummary, items_from, object_from

ROLE = "coordinator"

INSTRUCTIONS = """\
You are the coordinator of a data science team. You read the summary of the data,
the metric, the history of the attempts, the reports of the reviewer and the
guidance of the lead. You write a short plan (metric, validation, feature
engineering, models, baseline), the tasks that give the direction of the next
round and, if the problem calls for it, goal criteria (thresholds on metrics).
You only see summaries: never ask for labels or holdout data. You propose; the
fixed code decides what runs."""


@dataclass(frozen=True)
class CoordinatorInput(JsonObject):
    """What the coordinator is told: data summary, metric, history, reports and lead guidance."""

    LABEL: ClassVar[str] = "coordinator input"
    data: DataSummary
    metric: str
    history: RegistrySummary
    reports: tuple[ReviewerReport, ...] = ()
    guidance: tuple[str, ...] = ()

    _FIELDS: ClassVar[tuple[str, ...]] = ("data", "metric", "history", "reports", "guidance")

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataSummary):
            raise ValueError("data must be a DataSummary")
        require_text("metric", self.metric)
        if not isinstance(self.history, RegistrySummary):
            raise ValueError("history must be a RegistrySummary")
        object.__setattr__(self, "reports", tuple(self.reports))
        object.__setattr__(
            self, "guidance", require_text_list("guidance", self.guidance, allow_empty=True)
        )

    def to_dict(self) -> dict:
        return {
            "data": self.data.to_dict(),
            "metric": self.metric,
            "history": self.history.to_dict(),
            "reports": [r.to_dict() for r in self.reports],
            "guidance": list(self.guidance),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CoordinatorInput":
        check_keys("coordinator input", data, cls._FIELDS)
        return cls(
            data=object_from("data", data.get("data"), DataSummary),
            metric=data.get("metric"),
            history=object_from("history", data.get("history"), RegistrySummary),
            reports=items_from("reports", data.get("reports", []), ReviewerReport),
            guidance=data.get("guidance", ()),
        )


@dataclass(frozen=True)
class CoordinatorOutput(JsonObject):
    """The plan, the tasks of the round and, optionally, the goal criteria."""

    LABEL: ClassVar[str] = "coordinator output"
    plan: ShortPlan
    tasks: tuple[str, ...]
    goal: GoalCriteria | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = ("plan", "tasks", "goal")

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ShortPlan):
            raise ValueError("plan must be a ShortPlan")
        object.__setattr__(
            self, "tasks", require_text_list("tasks", self.tasks, allow_empty=False)
        )
        if self.goal is not None and not isinstance(self.goal, GoalCriteria):
            raise ValueError("goal must be GoalCriteria or None")

    @classmethod
    def contract_hints(cls) -> dict[str, str]:
        """Validation constraints per field, including those of the nested types."""
        hints = {f"plan.{k}": v for k, v in ShortPlan.contract_hints().items() if k != "__all__"}
        hints["plan"] = (
            "object with keys metric, validation, feature_engineering, models, baseline; "
            + ShortPlan.contract_hints()["__all__"]
        )
        hints["tasks"] = "non-empty list of non-empty strings"
        hints["goal"] = (
            "null, or an object {\"criteria\": [{\"metric\": string, \"threshold\": number, "
            "\"direction\": \"maximize\" or \"minimize\"}]} with unique metrics"
        )
        return hints

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "tasks": list(self.tasks),
            "goal": None if self.goal is None else self.goal.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CoordinatorOutput":
        check_keys("coordinator output", data, cls._FIELDS)
        goal = data.get("goal")
        return cls(
            plan=object_from("plan", data.get("plan"), ShortPlan),
            tasks=data.get("tasks"),
            goal=None if goal is None else object_from("goal", goal, GoalCriteria),
        )


def build_coordinator_prompt(role_input: CoordinatorInput) -> str:
    """The prompt of the coordinator: instructions and the serialized input."""
    return render_prompt(INSTRUCTIONS, role_input)


def run_coordinator(
    client: LlmClient,
    registry: Registry,
    context: TraceContext,
    role_input: CoordinatorInput,
    permissions: ToolPermissions,
    model: str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> CoordinatorOutput:
    """Ask the coordinator, validate the output and write its trace (no candidate).

    Raises ``RoleRejectedError`` after the bounded retries; other ``LlmError``
    subclasses (rate limit, quota, CLI failure) are raised unchanged. The call
    is traced in every case.
    """
    return call_role(
        client, registry, context, ROLE, PURPOSES[ROLE],
        build_coordinator_prompt(role_input), CoordinatorOutput, role_input,
        permissions, model=model, now=now,
    )
