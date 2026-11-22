"""The run history of the competition and the learnings notebook (ticket 30): what each run tried and learned."""

from kaggle_agent.overrides import search_items

from . import cycles_data, data, labels_data

PAGE = 50


def run_history(src, run_id: str) -> dict:
    reg = src.registry
    cycle = reg.get_cycle(run_id)
    rows = data.attempts(src, run_id)
    guidance = [g["text"] for g in data.guidance(src, run_id)]
    reviewer = [r["text"] for r in data.reviewer_reports(src, run_id)]
    suggestions = [s["text"] for s in data.suggestions(src) if s["origin_cycle"] == run_id]
    entry = cycles_data.entry_of(src.project_dir, run_id)
    notes = [n["text"] for (kind, ident), v in src.notes().items()
             if kind == "run" and ident == run_id for n in v["notes"]]
    return {
        "id": run_id, "state": cycle.state,
        "origin": None if entry is None else entry.get("origin"),
        "protocol_label": "same" if entry is None else entry.get("protocol_label"),
        "protocol_changes": {} if entry is None else entry.get("changes") or {},
        "tried": [{"attempt_id": r["id"], "round_id": r["round_id"], "hypothesis": r["hypothesis"],
                   "outcome": r["outcome"], "score": r["mean"]} for r in rows],
        "learned": {"conclusions": [{"attempt_id": r["id"], "text": r["reason"] or ""} for r in rows if r["reason"]],
                    "reviewer": reviewer, "guidance": guidance, "suggestions": suggestions, "notes": notes},
        "labels": labels_data.labels(src, run_id),
    }


def history(src) -> list[dict]:
    cycles = src.section("cycles", lambda: src.registry.list_cycles(), [])
    return [run_history(src, c.id) for c in cycles]


def learnings_items(src) -> list[dict]:
    """Notes (competition, run and attempt level) and agent suggestions as one searchable list."""
    items = []
    for (kind, ident), value in src.notes().items():
        for note in value["notes"]:
            items.append({"source": "note", "target_type": kind, "target_id": ident, "text": note["text"],
                          "tags": list(value["tags"]), "who": note.get("who"), "at": note.get("at"), "title": ""})
        if value["tags"] and not value["notes"]:
            items.append({"source": "tags", "target_type": kind, "target_id": ident, "text": "",
                          "tags": list(value["tags"]), "who": None, "at": None, "title": ""})
    for s in data.suggestions(src):
        items.append({"source": "suggestion", "target_type": "run", "target_id": s["origin_cycle"],
                      "text": s["text"], "tags": [], "who": "the agent", "at": None, "title": s["id"]})
    return items


def learnings(src, q: str | None, offset: int = 0, limit: int = PAGE) -> dict:
    found = search_items(learnings_items(src), q)
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    return {"q": q or "", "total": len(found), "offset": offset, "limit": limit,
            "items": found[offset:offset + limit]}
