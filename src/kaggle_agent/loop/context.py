"""Everything the cycle needs from outside, injected in one frozen bundle."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kaggle_agent.agents import CoordinatorOutput, DataSummary
from kaggle_agent.budget import CycleLimits
from kaggle_agent.bundle import LoadedBundle
from kaggle_agent.evaluation import Candidate, EvaluationEngine
from kaggle_agent.registry import Registry
from kaggle_agent.review import LeadInput

from .roles import RoleSet
from .writer import BundleWriter


@dataclass(frozen=True)
class Hooks:
    """The steps of the cycle that are not the round pipeline; ``None`` selects the default.

    - ``plan(ctx, revisions)``: returns the ``CoordinatorOutput`` of the planning step (default:
      one call to the coordinator role; ``revisions`` are the lead's change requests so far).
    - ``split(ctx)``: prepares the frozen split (default: builds it once to check the data).
    - ``baseline(ctx)``: produces the baseline attempt (default: one candidate in the setup round).
    - ``report(ctx, stop_reason)``: the final report (default: nothing; the report is a later step).
    """

    plan: Callable[..., CoordinatorOutput] | None = None
    split: Callable[..., None] | None = None
    baseline: Callable[..., None] | None = None
    report: Callable[..., None] | None = None


@dataclass(frozen=True)
class CycleContext:
    """The registry, the frozen engine, the roles, the writer, the runner factory and the lead.

    ``limits`` is only a fallback: a cycle that has stored parameters runs with those.
    ``runner_factory`` builds the ``Candidate`` that runs a loaded bundle (the real
    ``BundleRunner`` in production, a fake in tests). ``now`` is the clock for every timestamp.
    """

    registry: Registry
    cycle_id: str
    engine: EvaluationEngine
    frame: object
    data_hash: str
    data_summary: DataSummary
    roles: RoleSet
    writer: BundleWriter
    runner_factory: Callable[[LoadedBundle], Candidate]
    lead: LeadInput
    workspace: Path
    now: Callable[[], datetime]
    limits: CycleLimits = field(default_factory=CycleLimits)
    hooks: Hooks = field(default_factory=Hooks)
