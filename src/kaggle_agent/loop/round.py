"""The round pipeline: coordinator, candidates, reviewer, comparison, guidance, coordinator.

``RoundPipeline.iterate`` is one pass of the ITERATING state; it reads where the cycle stands from
the registry (``CycleView``), does the next unfinished step and returns the next state.
``RoundPipeline.guidance`` is the AWAITING_GUIDANCE state. Nothing is held in memory between
calls, so any call can be the first one of a resumed run.
"""

from kaggle_agent.agents import (
    CandidateResult,
    CoordinatorInput,
    MetricValue,
    ReviewerInput,
    RoleRejectedError,
    TraceContext,
)
from kaggle_agent.budget import CONSECUTIVE_FAILURES, CycleLimits, pause_due
from kaggle_agent.proposals import GoalCriteria, GoalCriterion, Guidance, ReviewerReport, Round
from kaggle_agent.review import CONTINUE, GUIDANCE, PROTOCOL_CHANGE, LeadContext

from . import ids
from .candidate import CandidateRunner, clean
from .context import CycleContext
from .derive import VALIDATION, CycleView, RoundState, is_placeholder
from .errors import LoopError
from .states import AWAITING_GUIDANCE, ITERATING, REPORTING

DEFAULT_TASK = "Improve the validation metric with a new, falsifiable hypothesis."
PROTOCOL_NOTICE = (
    "A change of the evaluation protocol is not guidance: it needs a new plan approval, "
    "so it was not applied. Give guidance, or let the cycle continue."
)


class RoundPipeline:
    def __init__(self, ctx: CycleContext, limits: CycleLimits, view: CycleView) -> None:
        self.ctx = ctx
        self.limits = limits
        self.view = view
        self.registry = ctx.registry
        self.candidates = CandidateRunner(ctx, limits, view)

    # -- ITERATING ----------------------------------------------------------------

    def iterate(self) -> str:
        """Advance the iteration phase by the next unfinished step; return the next state.

        After a finished round the lead is asked for guidance when a pause is due, also after
        the last round (the guidance then feeds the next cycle). A cap on consecutive failed
        attempts stops the cycle at once, in the middle of a round, with no pause.
        """
        view, limits = self.view, self.limits
        reason = view.stop_reason(limits)
        if reason == CONSECUTIVE_FAILURES:
            return REPORTING
        state = view.last_round()
        if state is None or (state.complete and state.has_direction):
            if reason:
                return REPORTING
            state = self._open_round(1 if state is None else state.index + 1)
        round_id = state.round_id
        tasks = state.round.round.tasks
        for index in range(1, limits.candidates + 1):
            self.candidates.run(round_id, index, tasks[(index - 1) % len(tasks)])
            if view.stop_reason(limits) == CONSECUTIVE_FAILURES:
                return REPORTING
        state = view.last_round()
        if not state.has_report:
            self._review(state)
            state = view.last_round()
        if not state.complete:
            self._compare(state)
            state = view.last_round()
        reason = view.stop_reason(limits)
        if not state.has_direction:
            if pause_due(state.index, limits.pause_every):
                return AWAITING_GUIDANCE
            if reason:
                return REPORTING
            self._direct(state)
        return ITERATING

    def _open_round(self, index: int) -> RoundState:
        tasks = self.view.latest_tasks() or (DEFAULT_TASK,)
        round_id = ids.round_id(self.ctx.cycle_id, index)
        self.registry.add_round(self.ctx.cycle_id, Round(round_id, tasks))
        return self.view.last_round()

    # -- reviewer -----------------------------------------------------------------

    def _reviewer_goal(self) -> GoalCriteria:
        """The goal of the protocol; without one, 'beat the baseline' stands in for it."""
        protocol = self.ctx.engine.protocol
        if protocol.goal is not None:
            return protocol.goal
        baseline = self.view.baseline()
        threshold = self.view.score(baseline) if baseline is not None else 0.0
        return GoalCriteria((GoalCriterion(protocol.metric, threshold, protocol.direction),))

    def _results(self, state: RoundState) -> tuple[CandidateResult, ...]:
        results = []
        for detail in state.details:
            score = self.view.score(detail)
            metrics = () if score is None else (MetricValue(self.ctx.engine.protocol.metric, score),)
            if is_placeholder(detail):
                outcome = "inconclusive"
            else:
                outcome = detail.outcome or "open"
            results.append(CandidateResult(detail.attempt.id, detail.attempt.hypothesis, outcome, metrics))
        return tuple(results)

    def _review(self, state: RoundState) -> None:
        """The reviewer writes its report on the round; it is never a gate."""
        ctx = self.ctx
        results = self._results(state)
        plan = self.registry.get_cycle(ctx.cycle_id).plan
        role_input = ReviewerInput(state.round_id, plan, self._reviewer_goal(), results)
        report_id = ids.report_id(state.round_id)
        try:
            report = ctx.roles.reviewer(TraceContext(ctx.cycle_id, state.round_id), role_input, report_id)
        except RoleRejectedError as exc:
            lines = "; ".join(f"{r.candidate_id}: {r.outcome}" for r in results) or "no candidates"
            report = ReviewerReport(
                id=report_id,
                round_id=state.round_id,
                text=f"Automatic summary: the reviewer gave no valid report ({clean(exc.reason)}). {lines}.",
                created_at=ctx.now().isoformat(),
            )
        self.registry.add_reviewer_report(report)

    # -- comparison ---------------------------------------------------------------

    def _compare(self, state: RoundState) -> None:
        """One paired comparison per round: its best candidate against the best so far."""
        best, reference = state.best, state.reference
        if best is None or reference is None or state.comparison is not None:
            return
        engine = self.ctx.engine
        baseline = self.view.baseline() or reference
        scores = engine.scores_from_execution
        engine.compare(
            best.attempt.id,
            reference.attempt.id,
            scores(best.executions[-1]),
            scores(reference.executions[-1]),
            scores(baseline.executions[-1]),
        )

    # -- coordinator ----------------------------------------------------------------

    def _direct(self, state: RoundState) -> None:
        """The coordinator reads the summary, the reports and the guidance and sets the next tasks."""
        ctx, view = self.ctx, self.view
        role_input = CoordinatorInput(
            data=ctx.data_summary,
            metric=ctx.engine.protocol.metric,
            history=view.registry_summary(),
            reports=view.recent_reports(),
            guidance=view.guidance_texts(),
        )
        ctx.roles.coordinator(TraceContext(ctx.cycle_id, state.round_id), role_input)
        if not view.last_round().has_direction:
            raise LoopError("the coordinator call left no trace with an output")

    # -- AWAITING_GUIDANCE ---------------------------------------------------------

    def guidance(self) -> str | None:
        """Ask the lead, record the guidance, then stop or let the coordinator set the next tasks.

        Returns the next state, or None when the lead has not answered yet.
        """
        ctx, view = self.ctx, self.view
        state = view.last_round()
        if state is None or not state.complete:
            return ITERATING
        reason = view.stop_reason(self.limits)
        if state.has_direction and not reason:
            return ITERATING
        recorded = any(g.round_id == state.round_id for g in self.registry.list_guidance(ctx.cycle_id))
        if not recorded and not self._ask_lead(state):
            return None
        if reason:
            return REPORTING
        self._direct(state)
        return ITERATING

    def _ask_lead(self, state: RoundState) -> bool:
        ctx = self.ctx
        context = LeadContext(
            cycle_id=ctx.cycle_id,
            state=AWAITING_GUIDANCE,
            summary=f"Round {state.index} of {self.limits.rounds} is finished.",
            round_id=state.round_id,
            round_position=state.index,
        )
        while True:
            response = ctx.lead.guidance(context)
            if response is None:
                return False
            if response.kind == CONTINUE:
                return True
            if response.kind == GUIDANCE:
                self.registry.add_guidance(
                    ctx.cycle_id,
                    Guidance(ids.guidance_id(state.round_id), state.round_id, response.text,
                             ctx.now().isoformat()),
                )
                return True
            assert response.kind == PROTOCOL_CHANGE
            ctx.lead.notice(PROTOCOL_NOTICE)
