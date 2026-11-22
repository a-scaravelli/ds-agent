"""The labels of a run, the program's own choice beside the DS's, and the status of every override (ticket 30).

Labels are derived at read time from files and registry facts (``kaggle_agent.overrides.labels``). A hand-edited
cockpit file could therefore produce a label without an effect: every label carries the status of the record behind
it (pending, applied or stale) and the audit log stays the authority.
"""

from kaggle_agent.overrides import KINDS, run_labels

from . import cycles_data, store

NOT_USED = "not used"


def _seed_reruns(src, run_id: str) -> list[str]:
    out = []
    for summary in src.section("attempts", lambda: src.registry.list_attempts(run_id), []):
        detail = src.registry.get_attempt(summary.attempt_id)
        if detail.executions and (detail.executions[-1].config or {}).get("seed_offset"):
            out.append(summary.attempt_id)
    return out


def labels(src, run_id: str) -> list[dict]:
    reg, project = src.registry, src.project_dir
    cycle = reg.get_cycle(run_id)
    first = cycles_data.first_cycle_id(project) or run_id
    entry = cycles_data.entry_of(project, run_id)
    facts = {"protocol_hash": cycle.protocol_hash, "seed_reruns": _seed_reruns(src, run_id)}
    if entry is not None:
        origin = (entry.get("origin") or {}).get("cycle_id")
        facts["origin_read"] = bool(origin) and src.section(
            "holdout", lambda: reg.get_holdout_outcome(origin), None) is not None
    outcomes = {k: v.get("outcome") for k, v in cycles_data.outcomes(project).items()}
    return run_labels(run_id, first, cycles_data.decisions_of(project, run_id), outcomes,
                      cycles_data.reads_of(project, run_id), entry, facts)


def choice(src, run_id: str) -> dict:
    """The program's own pick and the DS's, side by side (the report names both)."""
    project = src.project_dir
    stored = src.section("report", lambda: src.registry.get_final_report(run_id), None)
    chosen = (stored.content.get("chosen") if stored else None) or {}
    after = store.read_json(store.out_file(project, "chosen", run_id)) if project else None
    outcomes = cycles_data.outcomes(project)
    pending = [{"id": d["id"], "attempt": d.get("attempt"), "reason": d.get("reason"),
                "status": (outcomes.get(d["id"]) or {}).get("outcome", "pending")}
               for d in cycles_data.decisions_of(project, run_id, ("choose_model",))]
    out = {"program": chosen.get("attempt_id"), "ds": None, "decisions": pending}
    if isinstance(after, dict):
        out.update(program=after.get("program_choice"), ds={"attempt": after.get("attempt"),
                   "reason": after.get("reason"), "when": "after the report", "bundle": after.get("bundle")})
    elif chosen.get("chosen_by") == "the DS":
        out.update(program=chosen.get("program_choice"), ds={
            "attempt": chosen.get("attempt_id"), "reason": chosen.get("reason"), "when": "before the report",
            "bundle": chosen.get("bundle_id")})
    return out


def override_statuses(src) -> list[dict]:
    """Every DS override with its status: pending, applied, stale, or 'not used' (a request that never ran)."""
    project = src.project_dir
    if not project:
        return []
    outcomes = cycles_data.outcomes(project)
    cycle_id = cycles_data.active_cycle_id(project)
    state = None
    try:
        state = src.registry.get_cycle(cycle_id).state if cycle_id else None
    except Exception:
        state = None
    records, _ = store.read_jsonl(store.cockpit_path(project, "decisions.jsonl"))
    first = cycles_data.first_cycle_id(project)
    rows = []
    for record in records:
        if record.get("kind") not in KINDS:
            continue
        done = outcomes.get(record.get("id"))
        status = done.get("outcome") if done else "pending"
        reason = done.get("reason") if done else None
        if status == "pending" and record.get("kind") == "try_this" and (
                state == "STOPPED" or record.get("cycle", first) != cycle_id):
            status, reason = NOT_USED, "the run stopped before another round started; nothing was applied"
        rows.append({"id": record["id"], "kind": record["kind"], "cycle": record.get("cycle", first),
                     "who": record.get("who"), "at": record.get("at"), "status": status, "reason": reason,
                     "bound_to": (done or {}).get("bound_to"),
                     "summary": _summary(record)})
    return rows


def _summary(record: dict) -> str:
    kind = record["kind"]
    if kind == "choose_model":
        return f"choose attempt {record.get('attempt')}"
    if kind == "bar_override":
        return f"accept despite the bar ({record.get('scope')}, round {record.get('round')})"
    if kind == "steering":
        return "clear the steering" if record.get("clear") else "steer the roles"
    if kind == "try_this":
        return f"try this from round {record.get('from_round')}: {(record.get('request') or {}).get('hypothesis', '')[:80]}"
    return f"re-run {record.get('attempt')} (seed offset {record.get('seed_offset')})"


# -- the improvement rule: the last complete round and the numbers the DS sees before overriding it ------------


def round_state(src, cycle_id: str) -> dict:
    """Rounds opened, the last COMPLETE round (reviewer report and a stored comparison) and its comparison."""
    reg = src.registry
    setup = f"{cycle_id}-R0"
    rounds = [r for r in src.section("rounds", lambda: reg.list_rounds_with_candidates(cycle_id), [])
              if r.round.id != setup]
    reports = {r.round_id for r in src.section("reviewer_reports", lambda: reg.list_reviewer_reports(cycle_id), [])}
    comparisons = {c.candidate_attempt_id: c for c in src.section(
        "comparisons", lambda: reg.list_comparisons(cycle_id), [])}
    last = 0
    found = None
    for index, detail in enumerate(rounds, start=1):
        members = [c.attempt_id for c in detail.candidates]
        hit = next((m for m in members if m in comparisons), None)
        if detail.round.id in reports and hit is not None:
            last, found = index, (detail.round.id, hit, comparisons[hit])
    return {"rounds_started": len(rounds), "last_complete": last,
            "round_id": found[0] if found else None, "candidate": found[1] if found else None,
            "comparison": found[2] if found else None}


def bar_panel(src, run_id: str) -> dict:
    """The bar, the paired differences and the win count of the last complete round (from the analytics view)."""
    from . import analytics_data

    state = round_state(src, run_id)
    comparison = state["comparison"]
    out = {"run_id": run_id, "available": comparison is not None, "round": state["last_complete"],
           "rounds_started": state["rounds_started"], "round_id": state["round_id"],
           "candidate": state["candidate"], "stale_warning": None}
    if comparison is None:
        out["reason"] = "no round has finished with a comparison yet"
        return out
    out.update(reference=comparison.reference_attempt_id, difference=comparison.difference,
               bar=comparison.threshold, is_real=comparison.is_real)
    try:
        paired = analytics_data.paired(src, run_id, comparison.reference_attempt_id, comparison.candidate_attempt_id,
                                       with_correlation=False)["comparison"]
    except Exception:
        paired = None
    if isinstance(paired, dict) and not paired.get("not_comparable") and "wins" in paired:
        out.update(wins=paired["wins"], losses=paired["losses"], ties=paired["ties"], folds=paired["common"],
                   differences=paired["differences"], interval=paired["interval"])
    out["text"] = ("The strict bar says this candidate is " + ("a real improvement." if comparison.is_real else
                   "NOT a real improvement: its mean paired difference is not above the bar.") +
                   " If you accept it anyway, it becomes the reference and the stagnation count resets; the run is "
                   "labelled 'bar overridden by the DS' and the stored numbers are not changed.")
    return out


def steering_now(src, run_id: str) -> dict | None:
    """The steering in force for a run: the latest ``steering`` decision (a clear removes it)."""
    records = cycles_data.decisions_of(src.project_dir, run_id, ("steering",))
    if not records or records[-1].get("clear"):
        return None
    return records[-1].get("steering")
