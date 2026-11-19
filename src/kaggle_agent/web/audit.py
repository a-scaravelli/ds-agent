"""The audit log: every DS action, written BEFORE it takes effect (``cockpit/audit.jsonl``).

One record per action: ``id``, ``at`` (UTC), ``who`` (the local OS user), ``what`` (a short action name),
``why`` (the reason the DS gave, where one is required) and a small ``payload`` without secrets. An action
whose audit write fails is refused (``AuditError``). The audit page merges these records chronologically
with the relevant events of the orchestrator.
"""

from . import store

SYSTEM_EVENTS = ("run_start", "finished", "paused", "stopped", "suspended", "error", "waiting",
                 "decision_applied", "decision_stale", "limits_applied")


class AuditError(Exception):
    """The audit record could not be written: the action must not happen."""


def record(project_dir, what: str, why: str | None = None, payload: dict | None = None) -> dict:
    entry = {"id": store.new_id(), "at": store.utc_text(), "who": store.os_user(), "what": what,
             "why": why or None, "payload": payload or {}}
    try:
        store.append_jsonl(store.cockpit_path(project_dir, "audit.jsonl"), entry, sync=True)
    except (OSError, ValueError) as exc:
        raise AuditError(f"the audit log cannot be written ({exc}): the action was not done") from None
    return entry


def entries(project_dir) -> list[dict]:
    records, _ = store.read_jsonl(store.cockpit_path(project_dir, "audit.jsonl"))
    return records


def timeline(project_dir, limit: int = 500) -> list[dict]:
    """DS actions and relevant system events in chronological order (newest last)."""
    rows = [{**e, "source": "ds"} for e in entries(project_dir)]
    events, _ = store.read_jsonl(store.cockpit_path(project_dir, "events.jsonl"))
    for event in events:
        if event.get("type") in SYSTEM_EVENTS:
            rows.append({"id": f"event-{event.get('id')}", "at": event.get("at"), "who": "system",
                         "what": event.get("type"), "why": None, "payload": event.get("data") or {},
                         "source": "system"})
    rows.sort(key=lambda r: r.get("at") or "")
    return rows[-limit:]
