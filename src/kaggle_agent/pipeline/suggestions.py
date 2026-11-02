"""Suggestions for the next cycle, derived by fixed rules from the registry (design sections 7 and 15).

``derive_suggestions`` is deterministic code with no model call. It builds ``Suggestion`` records
with ids ``<cycle>-sg-<kind>`` from four rules:

1. ``stop``: the stop reason (stagnation, rounds used up, target met, failure cap);
2. ``failures``: the most frequent failure kind of the cycle, when it occurred at least
   ``MIN_FAILURES`` times;
3. ``holdout``: a failed holdout run, or a holdout worse than validation by more than the
   dispersion of the chosen bundle;
4. ``guidance-r<n>``: a guidance of the lead after round n that the next round did not turn into a
   real improvement, quoting the guidance.

The guidance itself is already stored as a suggestion by ``add_guidance`` (same id as the
guidance); it is not duplicated. The thresholds are provisional. ``store_suggestions`` skips ids
that exist, so a resumed run writes nothing twice. A suggestion only records an origin: starting
the next cycle automatically is the orchestrator (step 12).
"""

from collections import Counter

from kaggle_agent.budget import CONSECUTIVE_FAILURES, MAX_ROUNDS, STAGNATION, TARGET_REACHED
from kaggle_agent.loop import CycleView
from kaggle_agent.proposals import Suggestion
from kaggle_agent.registry import HoldoutOutcome, NotFoundError, Registry

MIN_FAILURES = 2
QUOTE_LIMIT = 160


def suggestion_id(cycle_id: str, kind: str) -> str:
    """Id of the derived suggestion of ``kind`` for the cycle."""
    return f"{cycle_id}-sg-{kind}"


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= QUOTE_LIMIT else text[: QUOTE_LIMIT - 3] + "..."


def _stop_text(stop_reason: str | None, view: CycleView, chosen) -> str:
    rounds = len([s for s in view.round_states() if s.complete])
    name = chosen.attempt.id if chosen is not None else "none"
    if stop_reason == STAGNATION:
        return (f"The cycle stopped for stagnation after {rounds} rounds: the last rounds brought no "
                "real improvement. Start the next cycle from a different direction (new features or "
                f"another model family) instead of tuning the current best ({name}).")
    if stop_reason == MAX_ROUNDS:
        return (f"The cycle used all its {rounds} rounds. Continue from the best attempt ({name}) with "
                "more rounds, or a wider search, if the last rounds were still improving.")
    if stop_reason == TARGET_REACHED:
        return (f"The target was reached with {name}. Use this bundle, or set a stricter goal for "
                "the next cycle.")
    if stop_reason == CONSECUTIVE_FAILURES:
        return ("The cycle stopped after too many failed candidates in a row. Fix the cause of the "
                "failures before the next cycle (see the failure kinds in the report).")
    return f"The cycle stopped ({stop_reason or 'no reason recorded'}). Review the report before the next cycle."


def _failure_text(registry: Registry, cycle_id: str) -> str | None:
    mine = {s.attempt_id for s in registry.list_attempts(cycle_id)}
    kinds = Counter(f.failure_kind for f in registry.list_failures() if f.attempt_id in mine)
    if not kinds:
        return None
    best = max(sorted(kinds), key=lambda kind: kinds[kind])
    if kinds[best] < MIN_FAILURES:
        return None
    return (f"{kinds[best]} attempts of the cycle failed with {best}. Prevent it in the next cycle "
            "(check the bundles for this failure before running them).")


def _holdout_text(outcome: HoldoutOutcome | None, view: CycleView, chosen) -> str | None:
    if outcome is None:
        return None
    if outcome.status == "failed":
        return (f"The holdout evaluation failed ({outcome.failure_kind}: {outcome.reason}). Fix the "
                "chosen bundle so that it predicts from its stored model, then evaluate it again in "
                "a new cycle with a new holdout.")
    if outcome.status != "succeeded" or chosen is None:
        return None
    validation = view.score(chosen)
    if validation is None:
        return None
    metric = next(m for m in chosen.executions[-1].metrics if m.partition == "validation")
    gap = validation - outcome.score if view.protocol.direction == "maximize" else outcome.score - validation
    if gap > metric.dispersion and gap > 0:
        return (f"The holdout {outcome.metric} ({outcome.score:.4f}) is worse than validation "
                f"({validation:.4f}) by {gap:.4f}, more than the validation dispersion "
                f"({metric.dispersion:.4f}). Suspect overfitting to the validation scheme or a "
                "shift between development and holdout; prefer simpler models next time.")
    return None


def _guidance_texts(registry: Registry, view: CycleView, cycle_id: str) -> list[tuple[str, str]]:
    states = view.round_states()
    by_round = {s.round_id: s for s in states}
    out = []
    for guidance in registry.list_guidance(cycle_id):
        state = by_round.get(guidance.round_id)
        if state is None:
            continue
        following = next((s for s in states if s.index == state.index + 1), None)
        if following is None or not following.complete or following.improved:
            continue
        text = (f"The lead's guidance after round {state.index} ({_quote(guidance.text)!r}) was followed by "
                f"round {following.index} with no real improvement. Revisit it in the next cycle, or "
                "try the opposite.")
        out.append((f"guidance-r{state.index}", text))
    return out


def derive_suggestions(registry: Registry, view: CycleView, outcome: HoldoutOutcome | None,
                       stop_reason: str | None) -> list[Suggestion]:
    """The suggestions for the next cycle, in a fixed order; the same state always gives the same list."""
    cycle_id = view.cycle_id
    chosen = view.reference()
    items = [("stop", _stop_text(stop_reason, view, chosen)),
             ("failures", _failure_text(registry, cycle_id)),
             ("holdout", _holdout_text(outcome, view, chosen))]
    items += _guidance_texts(registry, view, cycle_id)
    return [Suggestion(suggestion_id(cycle_id, kind), text, cycle_id)
            for kind, text in items if text is not None]


def store_suggestions(registry: Registry, suggestions: list[Suggestion]) -> list[Suggestion]:
    """Add the suggestions that are not stored yet; return the ones written by this call."""
    written = []
    for suggestion in suggestions:
        try:
            registry.get_suggestion(suggestion.id)
        except NotFoundError:
            registry.add_suggestion(suggestion)
            written.append(suggestion)
    return written
