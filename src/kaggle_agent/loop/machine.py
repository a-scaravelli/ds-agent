"""The cycle state machine, fixed code (design section 2).

``CycleMachine.run`` reads the saved ``Cycle.state``, runs the handler of that state, checks that
the transition is allowed, saves the new state and repeats, until the cycle stops, waits for the
lead or is suspended. The state is saved after every transition and all progress inside a state
is derived from the registry, so a crash, a suspension or a rate limit resumes without losing work
and without repeating a finished step.

Outcomes: ``stopped`` (STOPPED reached or already there), ``waiting`` (the lead has not answered:
nothing is approved or accepted by default) and ``suspended`` (rate limit, quota, CLI failure or a
rejected coordinator; the state is intact).
"""

from dataclasses import dataclass

from kaggle_agent.agents import CoordinatorInput, RegistrySummary, TraceContext
from kaggle_agent.budget import CycleLimits
from kaggle_agent.evaluation import FrozenSplit
from kaggle_agent.llm import LlmError
from kaggle_agent.proposals import Round
from kaggle_agent.review import Accept, Approve, LeadContext, Revise

from . import ids
from .candidate import CandidateRunner
from .context import CycleContext
from .derive import CycleView
from .errors import CycleSuspended, TransitionError
from .round import RoundPipeline
from .states import (
    AWAITING_GUIDANCE,
    AWAITING_PLAN_APPROVAL,
    AWAITING_REVIEW,
    BASELINE,
    ITERATING,
    PLANNING,
    REPORTING,
    SPLITTING,
    STATES,
    STOPPED,
    check_transition,
)

STATUS_STOPPED = "stopped"
STATUS_WAITING = "waiting"
STATUS_SUSPENDED = "suspended"


@dataclass(frozen=True)
class CycleOutcome:
    """How a call to ``run`` ended and where the cycle stands."""

    status: str
    state: str
    reason: str | None = None


class CycleMachine:
    """Runs one cycle from its saved state with everything external injected."""

    def __init__(self, ctx: CycleContext) -> None:
        self.ctx = ctx
        self.registry = ctx.registry
        self.view = CycleView(ctx.registry, ctx.cycle_id, ctx.engine)
        self.limits = ctx.limits
        self._revisions: list[str] = []

    # -- run ----------------------------------------------------------------------

    def run(self) -> CycleOutcome:
        cycle = self.registry.get_cycle(self.ctx.cycle_id)
        if cycle.state == STOPPED:
            return CycleOutcome(STATUS_STOPPED, STOPPED, "the cycle was already stopped")
        if cycle.state not in STATES:
            raise TransitionError(f"unknown state {cycle.state!r}")
        try:
            self._prepare()
            while True:
                state = self.registry.get_cycle(self.ctx.cycle_id).state
                new = self._handler(state)()
                if new is None:
                    return CycleOutcome(STATUS_WAITING, state, f"waiting for the lead in {state}")
                check_transition(state, new)
                self.registry.set_cycle_state(self.ctx.cycle_id, new)
                if new == STOPPED:
                    return CycleOutcome(STATUS_STOPPED, STOPPED, self.view.stop_reason(self.limits))
        except (CycleSuspended, LlmError) as exc:
            state = self.registry.get_cycle(self.ctx.cycle_id).state
            reason = exc.reason if isinstance(exc, CycleSuspended) else str(exc)
            return CycleOutcome(STATUS_SUSPENDED, state, reason)

    def _prepare(self) -> None:
        """Store the limits once, resolve them from the cycle and create the setup round."""
        ctx = self.ctx
        cycle = self.registry.get_cycle(ctx.cycle_id)
        if cycle.parameters is None:
            self.registry.set_cycle_parameters(ctx.cycle_id, ctx.limits.to_dict())
            cycle = self.registry.get_cycle(ctx.cycle_id)
        self.limits = CycleLimits.resolve(cycle, ctx.limits)
        if self.view.setup_round() is None:
            self.registry.add_round(
                ctx.cycle_id,
                Round(ids.setup_round_id(ctx.cycle_id), ("Set up the cycle: plan, split and baseline",)),
            )

    def _handler(self, state: str):
        return {
            PLANNING: self._planning,
            AWAITING_PLAN_APPROVAL: self._plan_approval,
            SPLITTING: self._splitting,
            BASELINE: self._baseline,
            ITERATING: self._iterating,
            AWAITING_GUIDANCE: self._guidance,
            REPORTING: self._reporting,
            AWAITING_REVIEW: self._review,
        }[state]

    def _lead_context(self, state: str, summary: str) -> LeadContext:
        return LeadContext(cycle_id=self.ctx.cycle_id, state=state, summary=summary)

    # -- handlers (each returns the next state, or None to wait) -----------------------

    def _planning(self) -> str:
        ctx = self.ctx
        if ctx.hooks.plan is not None:
            ctx.hooks.plan(ctx, tuple(self._revisions))
        else:
            role_input = CoordinatorInput(
                data=ctx.data_summary,
                metric=ctx.engine.protocol.metric,
                history=RegistrySummary(),
                guidance=tuple(self._revisions),
            )
            ctx.roles.coordinator(
                TraceContext(ctx.cycle_id, ids.setup_round_id(ctx.cycle_id)), role_input
            )
        return AWAITING_PLAN_APPROVAL

    def _plan_approval(self) -> str | None:
        ctx = self.ctx
        decision = ctx.lead.plan_decision(
            self._lead_context(AWAITING_PLAN_APPROVAL, "Approve the plan to freeze the protocol.")
        )
        if decision is None:
            return None
        if isinstance(decision, Revise):
            self._revisions.append(decision.text)
            return PLANNING
        if not isinstance(decision, Approve):
            raise TransitionError(f"unexpected plan decision {decision!r}")
        if self.registry.get_cycle(ctx.cycle_id).protocol_hash is None:
            ctx.engine.freeze()
        return SPLITTING

    def _splitting(self) -> str:
        ctx = self.ctx
        if ctx.hooks.split is not None:
            ctx.hooks.split(ctx)
        else:
            FrozenSplit.create(ctx.frame, ctx.engine.protocol)  # fails early on unusable data
        return BASELINE

    def _baseline(self) -> str:
        ctx = self.ctx
        if ctx.hooks.baseline is not None:
            ctx.hooks.baseline(ctx)
        else:
            plan = self.registry.get_cycle(ctx.cycle_id).plan
            CandidateRunner(ctx, self.limits, self.view).run(
                ids.setup_round_id(ctx.cycle_id), 1, f"Baseline: {plan.baseline}"
            )
        return ITERATING

    def _iterating(self) -> str:
        return RoundPipeline(self.ctx, self.limits, self.view).iterate()

    def _guidance(self) -> str | None:
        return RoundPipeline(self.ctx, self.limits, self.view).guidance()

    def _reporting(self) -> str:
        ctx = self.ctx
        if ctx.hooks.report is not None:
            ctx.hooks.report(ctx, self.view.stop_reason(self.limits) or "stopped")
        return AWAITING_REVIEW

    def _review(self) -> str | None:
        decision = self.ctx.lead.review_decision(
            self._lead_context(AWAITING_REVIEW, "Accept the report to stop the cycle.")
        )
        if decision is None:
            return None
        if not isinstance(decision, Accept):
            raise TransitionError(f"unexpected review decision {decision!r}")
        return STOPPED
