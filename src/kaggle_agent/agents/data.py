"""Data role: profiles the data summary, finds quality problems and leakage hints, proposes cleaning.

The output type ``DataFindings`` lives in ``summaries`` because other roles
take it as input and roles do not import each other. It is re-exported here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from kaggle_agent.llm import LlmClient, ToolPermissions
from kaggle_agent.proposals.errors import check_keys, require_text
from kaggle_agent.registry import Registry

from .call import PURPOSES, TraceContext, _utc_now, call_role
from .prompting import render_prompt
from .summaries import DataFindings, DataSummary, JsonObject, object_from

ROLE = "data"

INSTRUCTIONS = """\
You are the data analyst of a data science team. From the summary of the data
(sizes, target name and per-column statistics) and the task of the round, write
a profile, list the data quality issues, the hints of possible target leakage
and the cleaning you propose. You only see summaries, never rows, labels or
holdout data: say so when a conclusion would need more."""

__all__ = ["DataFindings", "DataInput", "build_data_prompt", "run_data"]


@dataclass(frozen=True)
class DataInput(JsonObject):
    """The data summary and the task of the round."""

    LABEL: ClassVar[str] = "data input"
    summary: DataSummary
    task: str

    _FIELDS: ClassVar[tuple[str, ...]] = ("summary", "task")

    def __post_init__(self) -> None:
        if not isinstance(self.summary, DataSummary):
            raise ValueError("summary must be a DataSummary")
        require_text("task", self.task)

    def to_dict(self) -> dict:
        return {"summary": self.summary.to_dict(), "task": self.task}

    @classmethod
    def from_dict(cls, data: dict) -> "DataInput":
        check_keys("data input", data, cls._FIELDS)
        return cls(
            summary=object_from("summary", data.get("summary"), DataSummary),
            task=data.get("task"),
        )


def build_data_prompt(role_input: DataInput) -> str:
    """The prompt of the data role: instructions and the serialized input."""
    return render_prompt(INSTRUCTIONS, role_input)


def run_data(
    client: LlmClient,
    registry: Registry,
    context: TraceContext,
    role_input: DataInput,
    permissions: ToolPermissions,
    model: str | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> DataFindings:
    """Ask the data role, validate the findings and write its trace (no candidate).

    Errors behave as in ``run_coordinator``.
    """
    return call_role(
        client, registry, context, ROLE, PURPOSES[ROLE],
        build_data_prompt(role_input), DataFindings, role_input,
        permissions, model=model, now=now,
    )
