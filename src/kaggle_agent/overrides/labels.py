"""The labels of a run, derived at read time from the files and the registry facts passed in."""

LABELS = ("chosen by the DS", "bar overridden by the DS", "holdout re-read", "new protocol (set by the DS)",
          "steered by the DS", "re-run with another seed")


def _status(record: dict, outcomes: dict) -> str:
    return outcomes.get(record.get("id"), "pending")


def run_labels(cycle_id: str, first_cycle: str, decisions: list[dict], outcomes: dict[str, str],
               reads: list[dict], entry: dict | None, facts: dict | None = None) -> list[dict]:
    """Labels with their reason and the status of the record behind them (pending, applied or stale).

    ``decisions`` are the records of THIS cycle, ``outcomes`` maps a decision id to its applied outcome,
    ``reads`` are the holdout reads of this cycle, ``entry`` is the ``cycles.json`` entry (None for the first
    cycle) and ``facts`` may hold ``origin_read`` (the origin cycle read the holdout under the same protocol)
    and ``seed_reruns`` (attempt ids re-run with another seed, from the registry).
    """
    facts = facts or {}
    out: list[dict] = []

    def add(label, reason="", status="applied", **extra):
        out.append({"label": label, "reason": reason, "status": status, **extra})

    for record in decisions:
        status = _status(record, outcomes)
        kind = record.get("kind")
        if kind == "choose_model" and status != "stale":
            add("chosen by the DS", record.get("reason", ""), status, attempt=record.get("attempt"))
        elif kind == "bar_override" and status != "stale":
            add("bar overridden by the DS", record.get("reason", ""), status, scope=record.get("scope"),
                round=record.get("round"), attempt=record.get("attempt"))
        elif kind == "steering" and status != "stale" and not record.get("clear"):
            add("steered by the DS", "", status)
        elif kind == "rerun" and status != "stale" and record.get("seed_offset"):
            add("re-run with another seed", "", status, attempt=record.get("attempt"),
                seed_offset=record.get("seed_offset"))
    if len(reads) and any(r.get("kind") in ("early", "second") for r in reads):
        reasons = "; ".join(r.get("reason", "") for r in reads if r.get("reason"))
        add("holdout re-read", reasons, "applied", reads=len(reads))
    elif entry and entry.get("protocol_label") == "same" and facts.get("origin_read"):
        add("holdout re-read", "the origin cycle already read the holdout under the same protocol", "applied",
            repeated_look=True)
    if entry and entry.get("protocol_label") == "changed":
        add("new protocol (set by the DS)", entry.get("reason", ""), "applied",
            changes=entry.get("changes"), protocol_hash=facts.get("protocol_hash"))
    # a seed re-run seen in the registry but not in the decisions (a job started from a terminal)
    known = {l.get("attempt") for l in out if l["label"] == "re-run with another seed"}
    for attempt in facts.get("seed_reruns", ()):
        if attempt not in known:
            add("re-run with another seed", "", "applied", attempt=attempt)
    return out
