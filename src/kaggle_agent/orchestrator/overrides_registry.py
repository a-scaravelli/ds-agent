"""The registry as the cycle sees it when the DS overrode the improvement rule (ticket 30).

``OverridingRegistry`` delegates every call and overlays ONLY ``list_comparisons``: a candidate the strict bar
rejected but the DS accepted is returned with ``is_real`` True (difference and threshold untouched). The
stored comparison row is never rewritten (the registry stays frozen); the derived chain of ``CycleView`` then
moves the reference to that candidate and resets the stagnation count by itself.

Each ``bar_override`` record of ``cockpit/decisions.jsonl`` is checked against the last COMPLETE round: a
round-scope record is valid only while its round is the last complete (and last opened) one, a cycle-scope
record from round k on is valid until round k is behind. A valid record is marked applied once; a stale one
is marked stale (reason, ``decision_stale`` event) and ignored. A record for a round that is not complete yet
waits. It layers on EVERY run, supervised or not.
"""

from kaggle_agent.loop import ids
from kaggle_agent.overrides import apply_bar_overrides, records_for_cycle, valid_bar_overrides

from .applied import applied_outcomes, mark
from .cockpit import CockpitFiles, read_jsonl


class OverridingRegistry:
    def __init__(self, inner, files: CockpitFiles, emitter, cycle_id: str, first_cycle: str) -> None:
        self._inner = inner
        self._files = files
        self._emitter = emitter
        self._cycle_id = cycle_id
        self._first = first_cycle

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def bar_records(self) -> list[dict]:
        records, _ = read_jsonl(self._files.decisions)
        mine = records_for_cycle(records, self._cycle_id, self._first)
        return [r for r in mine if r.get("kind") == "bar_override" and isinstance(r.get("id"), str)]

    def list_comparisons(self, cycle_id: str):
        comparisons = self._inner.list_comparisons(cycle_id)
        if cycle_id != self._cycle_id:
            return comparisons
        records = self.bar_records()
        if not records:
            return comparisons
        setup = ids.setup_round_id(cycle_id)
        rounds = [r for r in self._inner.list_rounds_with_candidates(cycle_id) if r.round.id != setup]
        reports = {r.round_id for r in self._inner.list_reviewer_reports(cycle_id)}
        compared = {c.candidate_attempt_id for c in comparisons}
        round_of: dict[str, int] = {}
        last_complete = 0
        for index, detail in enumerate(rounds, start=1):
            members = {c.attempt_id for c in detail.candidates}
            round_of.update({attempt: index for attempt in members})
            if detail.round.id in reports and members & compared:
                last_complete = index
        valid, stale, _pending = valid_bar_overrides(records, applied_outcomes(self._files), len(rounds),
                                                     last_complete)
        for record, reason in stale:
            mark(self._files, self._emitter, record, "stale", reason)
        for record in valid:
            mark(self._files, self._emitter, record, "applied")
        return apply_bar_overrides(comparisons, valid, round_of)
