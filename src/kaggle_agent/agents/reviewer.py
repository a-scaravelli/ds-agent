"""Reviewer role: interprets the results of a round in text. It never decides.

``ReviewerNotes`` is what the model writes: one string, no boolean, enum,
score or choice, so nothing the reviewer says can act as a gate. The fixed
code builds the ``ReviewerReport`` with its own id, the round and the date.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from kaggle_agent.llm import LlmClient, ToolPermissions
from kaggle_agent.proposals import GoalCriteria, ReviewerReport, ShortPlan
from kaggle_agent.proposals.errors import check_keys, require_text
from kaggle_agent.registry import Registry

from .call import PURPOSES, TraceContext, _utc_now, call_role
from .prompting import render_prompt
from .summaries import JsonObject, MetricValue, items_from, object_from, require_items

ROLE = "reviewer"

INSTRUCTIONS = """\
You are the reviewer of a data science team. You interpret the results of one
round: what the candidates tried, what the validation metrics suggest about the
hypotheses, how far the results are from the goal criteria, and what looks
fragile or worth checking. You write plain text only. You interpret and do not
decide: you do not accept or reject candidates, you do not pick a winner and you
do not say whether the goal is met; the fixed code does that."""


@dataclass(frozen=True)
class CandidateResult(JsonObject):
    """One candidate of the round: id, hypothesis, outcome and validation metrics."""

    LABEL: ClassVar[str] = "candidate result"
    candidate_id: str
    hypothesis: str
    outcome: str
    metrics: tuple[MetricValue, ...] = ()

    _FIELDS: ClassVar[tuple[str, ...]] = ("candidate_id", "hypothesis", "outcome", "metrics")

    def __post_init__(self) -> None:
        for name in ("candidate_id", "hypothesis", "outcome"):
            require_text(name, getattr(self, name))
        object.__setattr__(self, "metrics", require_items("metrics", self.metrics, MetricValue))

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "hypothesis": self.hypothesis,
            "outcome": self.outcome,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateResult":
        check_keys("candidate result", data, cls._FIELDS)
        kwargs = {name: data.get(name) for name in cls._FIELDS[:3]}
        if "metrics" in data:
            kwargs["metrics"] = items_from("metrics", data["metrics"], MetricValue)
        return cls(**kwargs)


@dataclass(frozen=True)
class ReviewerInput(JsonObject):
    """The round, its plan, the goal criteria and the results of its candidates."""

    LABEL: ClassVar[str] = "reviewer input"
    round_id: str
    plan: ShortPlan
    goal: GoalCriteria
    results: tuple[CandidateResult, ...]

    _FIELDS: ClassVar[tuple[str, ...]] = ("round_id", "plan", "goal", "results")

    def __post_init__(self) -> None:
        require_text("round_id", self.round_id)
        if not isinstance(self.plan, ShortPlan):
            raise ValueError("plan must be a ShortPlan")
        if not isinstance(self.goal, GoalCriteria):
            raise ValueError("goal must be GoalCriteria")
        object.__setattr__(self, "results", require_items("results", self.results, CandidateResult))

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "plan": self.plan.to_dict(),
            "goal": self.goal.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewerInput":
        check_keys("reviewer input", data, cls._FIELDS)
        return cls(
            round_id=data.get("round_id"),
            plan=object_from("plan", data.get("plan"), ShortPlan),
            goal=object_from("goal", data.get("goal"), GoalCriteria),
            results=items_from("results", data.get("results"), CandidateResult),
        )


@dataclass(frozen=True)
class ReviewerNotes(JsonObject):
    """What the model writes: the interpretation as text, and nothing else."""

    LABEL: ClassVar[str] = "reviewer notes"
    text: str

    def __post_init__(self) -> None:
        require_text("text", self.text)

    @classmethod
    def contract_hints(cls) -> dict[str, str]:
        """Validation constraints per field, for the JSON contract asked of the model."""
        return {"text": "non-empty string, your interpretation of the round"}

    def to_dict(self) -> dict:
        return {"text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewerNotes":
        check_keys("reviewer notes", data, ("text",))
        return cls(text=data.get("text"))


def build_reviewer_prompt(role_input: ReviewerInput) -> str:
    """The prompt of the reviewer: instructions and the serialized input."""
    return render_prompt(INSTRUCTIONS, role_input)


def run_reviewer(
    client: LlmClient,
    registry: Registry,
    context: TraceContext,
    role_input: ReviewerInput,
    permissions: ToolPermissions,
    report_id: str,
    model: str | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> ReviewerReport:
    """Ask the reviewer and return a ``ReviewerReport`` with the given id.

    The id comes from the caller and the date from ``now``; the round comes
    from the input and the text from the model. The trace stores the report.
    Errors behave as in ``run_coordinator``.
    """
    require_text("report_id", report_id)

    def finalize(notes: ReviewerNotes) -> ReviewerReport:
        return ReviewerReport(
            id=report_id,
            round_id=role_input.round_id,
            text=notes.text,
            created_at=now().isoformat(),
        )

    return call_role(
        client, registry, context, ROLE, PURPOSES[ROLE],
        build_reviewer_prompt(role_input), ReviewerNotes, role_input,
        permissions, model=model, finalize=finalize, now=now,
    )
