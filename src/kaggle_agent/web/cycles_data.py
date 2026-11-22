"""Which cycle is active and which override files belong to a cycle (read only; ticket 30).

A project has ONE first cycle (``project.json``) and, after a fork, more cycles listed in ``cycles.json`` (written
by the orchestrator). A record of ``decisions.jsonl`` or ``holdout-reads.jsonl`` carries the ``cycle`` it is for; one
without it (an older file) belongs to the first cycle. Nothing here writes.
"""

from kaggle_agent.overrides import records_for_cycle

from . import store


def first_cycle_id(project_dir) -> str | None:
    info = store.read_json(store.project_path(project_dir, "project.json")) if project_dir else None
    return info.get("cycle_id") if isinstance(info, dict) and isinstance(info.get("cycle_id"), str) else None


def entries(project_dir) -> list[dict]:
    if not project_dir:
        return []
    value = store.read_json(store.project_json_file(project_dir, "cycles.json"), {})
    items = value.get("cycles") if isinstance(value, dict) else None
    return [i for i in items if isinstance(i, dict) and isinstance(i.get("cycle_id"), str)] \
        if isinstance(items, list) else []


def entry_of(project_dir, cycle_id: str) -> dict | None:
    return next((e for e in entries(project_dir) if e["cycle_id"] == cycle_id), None)


def active_cycle_id(project_dir) -> str | None:
    items = entries(project_dir)
    return items[-1]["cycle_id"] if items else first_cycle_id(project_dir)


def cycle_ids(project_dir) -> list[str]:
    first = first_cycle_id(project_dir)
    return ([first] if first else []) + [e["cycle_id"] for e in entries(project_dir)]


def _of_cycle(project_dir, records: list[dict], cycle_id: str) -> list[dict]:
    return records_for_cycle(records, cycle_id, first_cycle_id(project_dir) or cycle_id)


def decisions_of(project_dir, cycle_id: str, kinds=None) -> list[dict]:
    if not project_dir:
        return []
    records, _ = store.read_jsonl(store.cockpit_path(project_dir, "decisions.jsonl"))
    items = _of_cycle(project_dir, records, cycle_id)
    return [r for r in items if kinds is None or r.get("kind") in kinds]


def reads_of(project_dir, cycle_id: str) -> list[dict]:
    if not project_dir:
        return []
    records, _ = store.read_jsonl(store.cockpit_path(project_dir, "holdout-reads.jsonl"))
    return _of_cycle(project_dir, records, cycle_id)


def outcomes(project_dir) -> dict[str, dict]:
    if not project_dir:
        return {}
    applied, _ = store.read_jsonl(store.cockpit_path(project_dir, "decisions-applied.jsonl"))
    return {a.get("decision"): a for a in applied}
