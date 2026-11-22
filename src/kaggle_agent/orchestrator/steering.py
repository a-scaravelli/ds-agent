"""The roles with the typed guidance of the DS: steering, a carry-over and a 'Try this' request (ticket 30).

``steering_roles`` wraps ``ctx.roles`` like ``eventing_roles`` does: for the coordinator, feature engineering and
modeling it replaces the typed input with ``dataclasses.replace(input, steering=..., carry=..., request=...)``.
The data and reviewer roles pass through unchanged.

- ``steering`` is the fold of the latest ``steering`` decision of the cycle (a clear removes it). A record is
  marked applied the first time it reaches a role; an older one that was superseded before is marked stale.
- ``carry`` is the immutable ``carry/<cycle>.json`` of a forked cycle (None for the first cycle).
- ``request`` ('Try this') is bound to the FIRST proposal made in a round >= its ``from_round``. The binding
  (decision id -> candidate id) is written to ``decisions-applied.jsonl`` BEFORE the role call, so a resume
  reuses the same candidate and never binds twice; the later request of the same candidate (modeling, a
  retry after a rejection) finds it there.
"""

import dataclasses

from kaggle_agent.loop import ids
from kaggle_agent.overrides import records_for_cycle
from kaggle_agent.proposals import Steering, TryThis

from .applied import applied_outcomes, applied_records, mark
from .cockpit import CockpitFiles, read_jsonl

STEERED = ("coordinator", "feature_engineering", "modeling")
CANDIDATE_ROLES = ("feature_engineering", "modeling")


class SteeringState:
    """Reads the decisions of one cycle and applies them to role inputs."""

    def __init__(self, files: CockpitFiles, emitter, cycle_id: str, first_cycle: str, carry) -> None:
        self.files, self.emitter, self.cycle_id, self.first, self.carry = files, emitter, cycle_id, first_cycle, carry

    def _records(self, kind: str) -> list[dict]:
        records, _ = read_jsonl(self.files.decisions)
        return [r for r in records_for_cycle(records, self.cycle_id, self.first)
                if r.get("kind") == kind and isinstance(r.get("id"), str)]

    def steering(self) -> Steering | None:
        records = self._records("steering")
        if not records:
            return None
        latest = records[-1]
        for record in records[:-1]:
            mark(self.files, self.emitter, record, "stale", "superseded by a later steering")
        mark(self.files, self.emitter, latest, "applied")
        return None if latest.get("clear") else Steering.from_dict(latest["steering"])

    def request_for(self, candidate_id: str, round_no: int) -> TryThis | None:
        """The 'Try this' request of this candidate, binding the oldest waiting one when there is none."""
        records = {r["id"]: r for r in self._records("try_this")}
        if not records:
            return None
        for marker in applied_records(self.files):
            if marker.get("kind") == "try_this" and marker.get("bound_to") == candidate_id \
                    and marker.get("decision") in records:
                return TryThis.from_dict(records[marker["decision"]]["request"])
        if round_no < 1:
            return None
        done = applied_outcomes(self.files)
        for record in records.values():
            if record["id"] not in done and record.get("from_round", 1) <= round_no:
                mark(self.files, self.emitter, record, "applied", bound_to=candidate_id)
                return TryThis.from_dict(record["request"])
        return None

    def decorate(self, role: str, context, role_input):
        changes = {}
        steering = self.steering()
        if steering is not None:
            changes["steering"] = steering
        if self.carry is not None:
            changes["carry"] = self.carry
        if role in CANDIDATE_ROLES:
            round_no = ids.round_index(self.cycle_id, context.round_id)
            request = self.request_for(role_input.candidate_id, round_no)
            if request is not None:
                changes["request"] = request
        return dataclasses.replace(role_input, **changes) if changes else role_input


def steering_roles(roles, files: CockpitFiles, emitter, cycle_id: str, first_cycle: str, carry=None):
    """Every steered role callable wrapped with the typed guidance (the rest unchanged)."""
    state = SteeringState(files, emitter, cycle_id, first_cycle, carry)

    def wrap(name, call):
        if name not in STEERED:
            return call
        return lambda context, role_input, *rest: call(context, state.decorate(name, context, role_input), *rest)

    return dataclasses.replace(roles, **{f.name: wrap(f.name, getattr(roles, f.name))
                                         for f in dataclasses.fields(roles)})
