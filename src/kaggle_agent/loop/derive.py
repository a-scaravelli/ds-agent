"""Read-only functions that rebuild all progress of a cycle from the registry.

Nothing here writes. A resumed run asks these functions where it stands, so no counter lives in
memory: the rounds, the candidates, the improvement flags, the failure streak and the stop reason
are recomputed from the stored records every time.

Conventions:

- The setup round ``<cycle>-R0`` (planning and baseline) is not an iteration round: it is
  excluded from N, K and M.
- The reference of the first round is the baseline attempt (the earliest succeeded attempt of the
  setup round). The best so far starts as the baseline and moves to the best candidate of a
  round only when that round's comparison is real. If there is no baseline, the best candidate
  of the first round with a succeeded candidate becomes the reference, without a comparison.
- A round is complete when its reviewer report exists and its single comparison is recorded
  (no comparison is needed when it has no succeeded candidate or no reference yet).
- An inconclusive placeholder (a candidate that never produced a valid proposal) is recognised by
  ``is_placeholder`` and is skipped by the summary, the best-so-far chain, the baseline lookup
  and the target check.
"""

from dataclasses import dataclass

from kaggle_agent.agents import AttemptDigest, RegistrySummary
from kaggle_agent.budget import CycleLimits, rounds_without_improvement, stop_reason
from kaggle_agent.evaluation import EvaluationEngine, evaluate_goal
from kaggle_agent.registry import (
    AttemptDetail,
    ComparisonRecord,
    NotFoundError,
    Registry,
    RoundDetail,
)

from . import ids

PLACEHOLDER_PREFIX = "no valid proposal after"
SUMMARY_BEST = 3
SUMMARY_FAILURES = 5
SUMMARY_REPORTS = 5
VALIDATION = "validation"


def is_placeholder(detail: AttemptDetail) -> bool:
    """True for the attempt recorded when a candidate never produced a valid proposal.

    Identified by outcome ``inconclusive`` plus the reason prefix ``no valid proposal after``.
    """
    return (
        detail.outcome == "inconclusive"
        and detail.reason is not None
        and detail.reason.startswith(PLACEHOLDER_PREFIX)
    )


@dataclass(frozen=True)
class RoundState:
    """What the registry says about one iteration round."""

    round: RoundDetail
    index: int
    details: tuple[AttemptDetail, ...]
    best: AttemptDetail | None
    reference: AttemptDetail | None
    comparison: ComparisonRecord | None
    has_report: bool
    has_direction: bool
    complete: bool
    improved: bool

    @property
    def round_id(self) -> str:
        return self.round.round.id


class CycleView:
    """The derived view of one cycle."""

    def __init__(self, registry: Registry, cycle_id: str, engine: EvaluationEngine) -> None:
        self.registry = registry
        self.cycle_id = cycle_id
        self.engine = engine
        self.protocol = engine.protocol

    # -- attempts -------------------------------------------------------------

    def detail(self, attempt_id: str) -> AttemptDetail | None:
        try:
            return self.registry.get_attempt(attempt_id)
        except NotFoundError:
            return None

    def score(self, detail: AttemptDetail) -> float | None:
        """Mean validation metric of a succeeded attempt, as the engine computed it."""
        if detail.outcome != "succeeded":
            return None
        for metric in detail.executions[-1].metrics:
            if metric.partition == VALIDATION:
                return metric.mean
        return None

    def oriented(self, detail: AttemptDetail) -> float | None:
        """The score with the sign set so that larger is always better."""
        value = self.score(detail)
        if value is None:
            return None
        return value if self.protocol.direction == "maximize" else -value

    def usable(self, detail: AttemptDetail | None) -> bool:
        """Closed, succeeded, scored and not a placeholder."""
        return (
            detail is not None
            and detail.closed
            and not is_placeholder(detail)
            and self.oriented(detail) is not None
        )

    def best_of(self, details) -> AttemptDetail | None:
        """The best usable attempt (the earliest wins a tie)."""
        best = None
        for detail in details:
            if self.usable(detail) and (best is None or self.oriented(detail) > self.oriented(best)):
                best = detail
        return best

    # -- rounds ---------------------------------------------------------------

    def all_rounds(self) -> list[RoundDetail]:
        return self.registry.list_rounds_with_candidates(self.cycle_id)

    def setup_round(self) -> RoundDetail | None:
        wanted = ids.setup_round_id(self.cycle_id)
        return next((r for r in self.all_rounds() if r.round.id == wanted), None)

    def iteration_rounds(self) -> list[RoundDetail]:
        setup = ids.setup_round_id(self.cycle_id)
        return [r for r in self.all_rounds() if r.round.id != setup]

    def baseline(self) -> AttemptDetail | None:
        setup = self.setup_round()
        if setup is None:
            return None
        details = (self.detail(c.attempt_id) for c in setup.candidates)
        return next((d for d in details if self.usable(d)), None)

    def _reports(self) -> set[str]:
        return {r.round_id for r in self.registry.list_reviewer_reports(self.cycle_id)}

    def _directions(self) -> set[str]:
        traces = self.registry.list_role_traces(self.cycle_id, role="coordinator")
        return {t.round_id for t in traces if t.output is not None}

    def round_states(self) -> list[RoundState]:
        """One state per iteration round, in order, with the improvement chain applied."""
        reports, directions = self._reports(), self._directions()
        comparisons = self.registry.list_comparisons(self.cycle_id)
        reference = self.baseline()
        states: list[RoundState] = []
        chain_open = True
        for index, round_detail in enumerate(self.iteration_rounds(), start=1):
            details = tuple(
                d for d in (self.detail(c.attempt_id) for c in round_detail.candidates) if d
            )
            best = self.best_of(details)
            comparison = None
            if best is not None:
                comparison = next(
                    (c for c in comparisons if c.candidate_attempt_id == best.attempt.id), None
                )
            round_id = round_detail.round.id
            has_report = round_id in reports
            needs_comparison = best is not None and reference is not None
            complete = chain_open and has_report and (not needs_comparison or comparison is not None)
            improved = False
            if complete and best is not None:
                improved = True if reference is None else comparison.is_real
            states.append(
                RoundState(
                    round_detail, index, details, best, reference, comparison,
                    has_report, round_id in directions, complete, improved,
                )
            )
            if not complete:
                chain_open = False
            elif improved:
                reference = best
        return states

    def reference(self) -> AttemptDetail | None:
        """The best so far, after the complete rounds: what the next comparison is made against."""
        reference = self.baseline()
        for state in self.round_states():
            if state.complete and state.improved:
                reference = state.best
        return reference

    def last_round(self) -> RoundState | None:
        states = self.round_states()
        return states[-1] if states else None

    # -- counters and stop rules ---------------------------------------------

    def consecutive_failures(self) -> int:
        """Failed attempts in a row at the end of the iteration phase.

        A succeeded attempt resets the count; an inconclusive one (a placeholder included)
        neither counts nor resets; an open attempt is not counted yet.
        """
        count = 0
        for state in self.round_states():
            for detail in state.details:
                if not detail.closed or is_placeholder(detail):
                    continue
                if detail.outcome == "failed":
                    count += 1
                elif detail.outcome == "succeeded":
                    count = 0
        return count

    def target_met(self) -> bool:
        """True if a candidate of a complete round meets every goal criterion on validation."""
        goal = self.protocol.goal
        if goal is None:
            return False
        for state in self.round_states():
            if not state.complete:
                continue
            for detail in state.details:
                if not self.usable(detail):
                    continue
                values = self.engine.goal_values_from_execution(detail.executions[-1])
                if evaluate_goal(goal, values).met:
                    return True
        return False

    def stop_reason(self, limits: CycleLimits) -> str | None:
        complete = [s for s in self.round_states() if s.complete]
        return stop_reason(
            target_met=self.target_met(),
            rounds_done=len(complete),
            max_rounds=limits.rounds,
            stagnant_rounds=rounds_without_improvement([s.improved for s in complete]),
            stagnation_k=limits.stagnation_rounds,
            consecutive_failures=self.consecutive_failures(),
            max_consecutive_failures=limits.max_consecutive_failures,
        )

    # -- coordinator input ----------------------------------------------------

    def latest_tasks(self) -> tuple[str, ...] | None:
        """Tasks of the latest coordinator call that produced an output."""
        for trace in reversed(self.registry.list_role_traces(self.cycle_id, role="coordinator")):
            if trace.output is not None and trace.output.get("tasks"):
                return tuple(trace.output["tasks"])
        return None

    def registry_summary(self) -> RegistrySummary:
        """Top attempts and recent failures, with small fixed caps; placeholders excluded."""
        summaries = self.registry.list_attempts(self.cycle_id)
        details = [d for d in (self.detail(s.attempt_id) for s in summaries) if d is not None]
        usable = [d for d in details if self.usable(d)]
        usable.sort(key=lambda d: -self.oriented(d))  # stable: the earliest wins a tie
        best = tuple(
            AttemptDigest(d.attempt.id, d.attempt.hypothesis, "succeeded", self.score(d))
            for d in usable[:SUMMARY_BEST]
        )
        mine = {d.attempt.id for d in details if not is_placeholder(d)}
        failures = [f for f in self.registry.list_failures() if f.attempt_id in mine]
        by_id = {d.attempt.id: d for d in details}
        recent = tuple(
            AttemptDigest(
                f.attempt_id, by_id[f.attempt_id].attempt.hypothesis, f"{f.outcome}/{f.failure_kind}"
            )
            for f in failures[-SUMMARY_FAILURES:]
        )
        return RegistrySummary(best=best, recent_failures=recent)

    def recent_reports(self):
        return tuple(self.registry.list_reviewer_reports(self.cycle_id)[-SUMMARY_REPORTS:])

    def guidance_texts(self) -> tuple[str, ...]:
        return tuple(g.text for g in self.registry.list_guidance(self.cycle_id))
