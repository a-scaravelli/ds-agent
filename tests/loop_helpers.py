"""Builders for the loop tests: a frozen engine, fake roles, a fake runner, a simulated lead.

Everything is injected and offline: no network, no real Claude call, no real process. All
filesystem work goes under ``tmp_path``.
"""

import dataclasses
import json
from collections import deque
from datetime import datetime, timedelta, timezone

from evaluation_helpers import classification_frame, make_protocol
from registry_helpers import PLAN

from kaggle_agent.agents import (
    CandidateRoleResult,
    ColumnSummary,
    CoordinatorOutput,
    DataFindings,
    DataSummary,
)
from kaggle_agent.budget import CycleLimits
from kaggle_agent.bundle import BundleRunError
from kaggle_agent.evaluation import EvaluationEngine
from kaggle_agent.llm import ToolPermissions
from kaggle_agent.llm.errors import CliFailureError
from kaggle_agent.llm.fake import (
    FakeLlmClient,
    Malformed,
    QuotaExhausted,
    RateLimited,
    RawResponse,
    Reply,
    valid_attempt_json,
    valid_plan_json,
)
from kaggle_agent.llm.interface import BackoffPolicy
from kaggle_agent.llm.fake import CliCrash
from kaggle_agent.loop import (
    CycleContext,
    CycleMachine,
    FakeBundleWriter,
    Hooks,
    RoleSet,
    client_roles,
)
from kaggle_agent.loop.states import PLANNING
from kaggle_agent.proposals import Attempt, GoalCriteria, ReviewerReport
from kaggle_agent.registry import Cycle, RoleTrace, SqliteRegistry
from kaggle_agent.review import SimulatedLead

CYCLE = "C1"
FRAME = classification_frame()
ROLE_MARKERS = {
    "coordinator": "You are the coordinator",
    "data": "You are the data analyst",
    "feature_engineering": "You are the feature engineering specialist",
    "modeling": "You are the modeling specialist",
    "reviewer": "You are the reviewer",
}


class Boom(Exception):
    """An injected crash: it is not caught by the machine, so the process 'dies' there."""


class ManualClock:
    """A clock that moves one minute per reading."""

    def __init__(self) -> None:
        self.value = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.value += timedelta(minutes=1)
        return self.value


# -- runner ------------------------------------------------------------------------


def crash(reason="boom"):
    return BundleRunError("crashed", "code_error", reason)


class FakeRunner:
    """A ``Candidate``: predicts the true labels with a share of them flipped (exact accuracy)."""

    def __init__(self, loaded, outcome, frame) -> None:
        self.bundle = loaded
        self.outcome = outcome
        self.frame = frame

    def fit_predict(self, x_train, y_train, x_valid, seed):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        truth = self.frame.loc[x_valid.index, "y"].to_numpy().copy()
        flips = int(round(self.outcome * len(truth)))
        truth[:flips] = 1 - truth[:flips]
        return truth


class FakeRunnerFactory:
    """``rule(bundle_id, n)`` gives the error rate (a float) or an exception for run n of a bundle."""

    def __init__(self, rule=None, frame=FRAME) -> None:
        self.rule = rule or (lambda bundle_id, n: 0.3)
        self.frame = frame
        self.runs: list[str] = []

    def __call__(self, loaded):
        count = self.runs.count(loaded.bundle_id) + 1
        self.runs.append(loaded.bundle_id)
        return FakeRunner(loaded, self.rule(loaded.bundle_id, count), self.frame)


def by_round(rates, default=0.3, baseline=0.3):
    """Error rate per round index (1-based list); the baseline and unlisted rounds use defaults."""

    def rule(bundle_id, n):
        if "-R0-" in bundle_id:
            return baseline
        index = int(bundle_id.split("-R")[1].split("-")[0])
        return rates[index - 1] if index <= len(rates) else default

    return rule


# -- roles through a role-aware fake client -------------------------------------------


class RoleClient(FakeLlmClient):
    """A fake client that answers by role (from the prompt), with queued overrides per role."""

    def __init__(self, overrides=None, **kwargs) -> None:
        kwargs.setdefault("backoff", BackoffPolicy(max_retries=0))
        kwargs.setdefault("max_validation_retries", 0)
        super().__init__([], **kwargs)
        self.overrides = {role: deque(steps) for role, steps in (overrides or {}).items()}
        self.roles_called: list[str] = []
        self.prompts: list[tuple[str, str]] = []
        self.counts: dict[str, int] = {}

    @staticmethod
    def role_of(prompt):
        for role, marker in ROLE_MARKERS.items():
            if prompt.startswith(marker):
                return role
        raise AssertionError("unknown role prompt")

    def default_reply(self, role, n):
        if role == "coordinator":
            return json.dumps({"plan": json.loads(valid_plan_json()), "tasks": [f"task {n}"], "goal": None})
        if role == "data":
            return json.dumps({"profile": f"profile {n}", "quality_issues": [], "leakage_hints": [],
                               "proposed_cleaning": []})
        if role == "reviewer":
            return json.dumps({"text": f"review {n}"})
        return valid_attempt_json(hypothesis=f"{role} idea {n}", what_changed=f"{role} change {n}")

    def _run(self, prompt, permissions, model):
        role = self.role_of(prompt)
        self.calls.append(None)
        queue = self.overrides.get(role)
        if queue:
            step = queue.popleft()
        else:
            self.counts[role] = self.counts.get(role, 0) + 1
            step = Reply(self.default_reply(role, self.counts[role]))
        self.roles_called.append(role)
        self.prompts.append((role, prompt))
        if isinstance(step, RateLimited):
            from kaggle_agent.llm import RateLimitError
            raise RateLimitError("rate limited (simulated)", retry_after=step.retry_after)
        if isinstance(step, QuotaExhausted):
            from kaggle_agent.llm import QuotaExhaustedError
            raise QuotaExhaustedError("quota exhausted (simulated)", resets_at=step.resets_at)
        if isinstance(step, CliCrash):
            raise CliFailureError(step.reason)
        return RawResponse(text=step.text, usage=step.usage, session_id=None, model=None)


# -- roles as plain callables ------------------------------------------------------------


class FakeRoles:
    """Plain fake role callables that record their inputs and order, and leave coordinator traces."""

    def __init__(self, registry, clock, reject=None, same_change=False) -> None:
        self.registry = registry
        self.clock = clock
        self.calls: list[tuple[str, object]] = []
        # reject[role][candidate_id] is a list of exceptions raised on the successive calls
        self.reject = reject or {}
        self.same_change = same_change
        self.n = 0

    def _count(self):
        self.n += 1
        return self.n

    def coordinator(self, context, role_input):
        n = self._count()
        self.calls.append(("coordinator", role_input))
        output = CoordinatorOutput(PLAN, (f"task {n}",))
        self.registry.add_role_trace(RoleTrace(
            cycle_id=context.cycle_id, role="coordinator", round_id=context.round_id,
            input_type="CoordinatorInput", input=role_input.to_dict(), outcome="ok",
            created_at=self.clock().isoformat(), output_type="CoordinatorOutput",
            output=output.to_dict(),
        ))
        return output

    def data(self, context, role_input):
        self.calls.append(("data", role_input))
        return DataFindings(f"profile {self._count()}")

    def _candidate(self, role, role_input):
        n = self._count()
        self.calls.append((role, role_input))
        pending = self.reject.get(role, {}).get(role_input.candidate_id)
        if pending:
            raise pending.pop(0)
        attempt = Attempt(
            id=role_input.candidate_id, phase="iteration", hypothesis=f"{role} idea {n}",
            rationale="r", what_changed=f"{role} change" if self.same_change else f"{role} change {n}",
            measure="m", exploratory=True,
        )
        return CandidateRoleResult(attempt, None)

    def feature_engineering(self, context, role_input):
        return self._candidate("feature_engineering", role_input)

    def modeling(self, context, role_input):
        return self._candidate("modeling", role_input)

    def reviewer(self, context, role_input, report_id):
        self.calls.append(("reviewer", role_input))
        return ReviewerReport(report_id, role_input.round_id, f"review {self._count()}",
                              self.clock().isoformat())

    def roleset(self):
        return RoleSet(self.coordinator, self.data, self.feature_engineering, self.modeling,
                       self.reviewer)

    def names(self):
        return [name for name, _ in self.calls]


# -- the whole setup ------------------------------------------------------------------------


@dataclasses.dataclass
class Setup:
    registry: SqliteRegistry
    engine: EvaluationEngine
    ctx: CycleContext
    client: object
    lead: SimulatedLead
    factory: FakeRunnerFactory
    writer: FakeBundleWriter
    clock: ManualClock
    freezes: list

    def machine(self, **changes):
        ctx = dataclasses.replace(self.ctx, **changes) if changes else self.ctx
        return CycleMachine(ctx)

    def run(self, **changes):
        return self.machine(**changes).run()

    def cycle(self):
        return self.registry.get_cycle(CYCLE)

    def states(self):
        """The states in order of persistence: wrap ``set_cycle_state`` before running."""
        return self.seen


def data_summary():
    return DataSummary(240, "y", (ColumnSummary("x1", "float64", 0.0, 240),))


def make_setup(
    tmp_path, *, rounds=3, candidates=2, stagnation_k=2, fix_retries=3, pause_every=1,
    max_failures=3, max_rejections=3, goal=None, threshold_k=1.0, lead=None, rule=None,
    client=None, roles=None, hooks=None, writer=None, state=PLANNING, registry=None,
    parameters=None, create_cycle=True,
):
    """A cycle in ``state`` with a protocol, a (not yet frozen) engine and a full context."""
    registry = registry or SqliteRegistry()
    if create_cycle:  # False when the test starts the cycle itself (pipeline.start_cycle)
        registry.add_cycle(Cycle(
            id=CYCLE, plan=PLAN, iterations=rounds, stagnation_k=stagnation_k, state=state,
            max_fix_retries=fix_retries, parameters=parameters,
        ))
    overrides = {"goal": GoalCriteria.from_dict(goal)} if goal else {}
    engine = EvaluationEngine(registry, CYCLE, make_protocol(threshold_k=threshold_k, **overrides))
    freezes = []
    real_freeze = engine.freeze

    def counting_freeze():
        freezes.append(1)
        return real_freeze()

    engine.freeze = counting_freeze
    clock = ManualClock()
    client = client or RoleClient()
    workspace = tmp_path / "work"
    workspace.mkdir()
    if roles is None:
        roles = client_roles(client, registry, ToolPermissions(working_dir=workspace), now=clock)
    lead = lead or SimulatedLead()
    factory = FakeRunnerFactory(rule)
    writer = writer or FakeBundleWriter()
    limits = CycleLimits(
        rounds=rounds, candidates=candidates, stagnation_rounds=stagnation_k,
        pause_every=pause_every, fix_retries=fix_retries, max_consecutive_failures=max_failures,
        max_rejections=max_rejections,
    )
    ctx = CycleContext(
        registry=registry, cycle_id=CYCLE, engine=engine, frame=FRAME, data_hash="data-hash",
        data_summary=data_summary(), roles=roles, writer=writer, runner_factory=factory,
        lead=lead, workspace=workspace, now=clock, limits=limits, hooks=hooks or Hooks(),
    )
    return Setup(registry, engine, ctx, client, lead, factory, writer, clock, freezes)


def record_states(registry):
    """Wrap ``set_cycle_state`` so every saved state is appended to the returned list."""
    seen = []
    original = registry.set_cycle_state

    def spy(cycle_id, state):
        seen.append(state)
        original(cycle_id, state)

    registry.set_cycle_state = spy
    return seen


def attempts(registry):
    return registry.list_attempts(CYCLE)


def closed_outcomes(registry):
    return [(a.attempt_id, a.outcome, a.failure_kind) for a in attempts(registry)]
