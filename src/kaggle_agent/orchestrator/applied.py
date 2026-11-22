"""The 'applied once' marker of a web decision: one record in ``decisions-applied.jsonl`` and one event.

Factored out of ``FileLeadInput._mark`` (same file, same record and event shapes); the override wrappers use
it too. A decision is marked at most once: the marker is written BEFORE the effect that needs it to be
unique where the effect is not idempotent by itself (a 'Try this' bound to a candidate).
"""

from .cockpit import CockpitFiles, append_jsonl, read_jsonl, utc_text


def applied_outcomes(cockpit: CockpitFiles | None) -> dict[str, str]:
    """decision id -> outcome (``applied`` or ``stale``) of every marked decision."""
    if cockpit is None:
        return {}
    records, _ = read_jsonl(cockpit.applied)
    return {r.get("decision"): r.get("outcome") for r in records}


def applied_records(cockpit: CockpitFiles | None) -> list[dict]:
    return [] if cockpit is None else read_jsonl(cockpit.applied)[0]


def mark(cockpit: CockpitFiles | None, emitter, record: dict, outcome: str, reason: str = "",
         **extra) -> bool:
    """Mark ``record`` once; returns True when this call wrote the marker."""
    if cockpit is None or record["id"] in applied_outcomes(cockpit):
        return False
    append_jsonl(cockpit.applied, {"decision": record["id"], "kind": record["kind"], "outcome": outcome,
                                   "reason": reason, "at": utc_text(), **extra}, sync=True)
    if emitter is not None:
        if outcome == "applied":
            emitter.emit("decision_applied", decision=record["id"], kind=record["kind"], source="web")
        else:
            emitter.emit("decision_stale", decision=record["id"], kind=record["kind"], reason=reason,
                         source="web")
    return True
