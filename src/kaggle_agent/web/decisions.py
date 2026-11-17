"""Decisions of the DS from the page: validation against the registry state and the records.

The web never writes the registry. A decision is appended to ``cockpit/decisions.jsonl`` (after its audit
entry); the orchestrator applies it exactly once when it asks the matching question and records the
outcome in ``decisions-applied.jsonl``. Entries are ADDRESSED (plan number, round number), so a stale
decision is refused here (409) and ignored by the orchestrator. ``lead.toml`` keeps working; a web decision
for the same question wins, and a different ``lead.toml`` answer is shown as a conflict.
"""

from kaggle_agent.proposals import ShortPlan, ValidationError
from kaggle_agent.proposals.errors import require_text, require_text_list

from . import data, store
from .controller import Conflict

PLAN_STATE = "AWAITING_PLAN_APPROVAL"
GUIDANCE_STATE = "AWAITING_GUIDANCE"
REVIEW_STATE = "AWAITING_REVIEW"
MAX_TEXT = 4000
KINDS = {"plan": ("approve", "revise", "own"), "guidance": ("guidance", "continue")}


def cycle_id_of(src: data.Source) -> str | None:
    info = data.read_project_file(src.project_dir)
    if info and isinstance(info.get("cycle_id"), str):
        return info["cycle_id"]
    cycles = src.section("cycles", lambda: src.registry.list_cycles(), [])
    return cycles[-1].id if cycles else None


def text_of(body: dict, key: str, required: bool = True) -> str | None:
    value = body.get(key)
    if value is None or value == "":
        if required:
            raise ValidationError(key, "is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(key, "must be a non-empty text")
    if len(value) > MAX_TEXT:
        raise ValidationError(key, f"must be at most {MAX_TEXT} characters")
    return value.strip()


def plan_view(src: data.Source, cycle_id: str) -> dict | None:
    """The plan the coordinator (or the DS) wrote last, from its trace output (keys plan and tasks)."""
    traces = src.section("role_traces", lambda: src.registry.list_role_traces(cycle_id, role="coordinator"), [])
    traces = [t for t in traces if t.output]
    if not traces:
        return None
    last = traces[-1]
    plan = last.output.get("plan") or {}
    author = (last.input or {}).get("author") if isinstance(last.input, dict) else None
    return {"validation": plan.get("validation"), "metric": plan.get("metric"),
            "feature_engineering": plan.get("feature_engineering") or [], "models": plan.get("models") or [],
            "baseline": plan.get("baseline"), "tasks": last.output.get("tasks") or [],
            "author": "the DS" if author == "lead" else "the coordinator"}


def round_position(src: data.Source, cycle_id: str) -> int:
    rounds = src.section("rounds", lambda: src.registry.list_rounds_with_candidates(cycle_id), [])
    return len([r for r in rounds if not r.round.id.endswith("-R0")])


def current_question(src: data.Source) -> dict | None:
    """What the cycle waits for, from the read-only registry state (None when nothing is awaited)."""
    cycle_id = cycle_id_of(src)
    if cycle_id is None:
        return None
    try:
        cycle = src.registry.get_cycle(cycle_id)
    except Exception:
        return None
    project = src.project_dir
    if cycle.state == PLAN_STATE:
        number = len(store.read_ledger(project)) + 1
        return {"kind": "plan", "address": number, "state": cycle.state, "cycle": cycle_id,
                "plan": plan_view(src, cycle_id)}
    if cycle.state == GUIDANCE_STATE:
        position = round_position(src, cycle_id)
        rounds = data.rounds(src, cycle_id)
        last = rounds[-1] if rounds else None
        return {"kind": "guidance", "address": position, "state": cycle.state, "cycle": cycle_id,
                "round": last}
    if cycle.state == REVIEW_STATE:
        return {"kind": "review", "address": None, "state": cycle.state, "cycle": cycle_id}
    return None


def check_address(question: dict | None, kind: str, address) -> dict:
    if question is None or question["kind"] != kind:
        raise Conflict(f"the cycle is not waiting for a {kind} decision now: this answer is stale")
    if kind != "review" and address != question["address"]:
        raise Conflict(f"this answer is for {kind} {address}, but the question now is {kind} "
                       f"{question['address']}: it is stale and was not saved")
    return question


def own_plan_fields(body: dict) -> dict:
    """The typed plan, validated like a coordinator plan (``ShortPlan``) plus the tasks of the round."""
    raw = body.get("plan")
    if not isinstance(raw, dict):
        raise ValidationError("plan", "is required: the fields of your plan")
    for key in ("metric", "validation", "baseline"):
        if not isinstance(raw.get(key), str):
            raise ValidationError(f"plan.{key}", "must be a text")
    plan = ShortPlan.from_dict({k: raw.get(k) for k in ("metric", "validation", "feature_engineering",
                                                        "models", "baseline")})
    tasks = require_text_list("plan.tasks", raw.get("tasks"), allow_empty=False)
    return {"plan": plan.to_dict(), "tasks": list(tasks)}


def decision_statuses(project_dir) -> list[dict]:
    """Every saved web decision with its status: pending, applied or stale."""
    records, _ = store.read_jsonl(store.cockpit_path(project_dir, "decisions.jsonl"))
    applied, _ = store.read_jsonl(store.cockpit_path(project_dir, "decisions-applied.jsonl"))
    outcome = {a.get("decision"): a for a in applied}
    rows = []
    for record in records:
        done = outcome.get(record.get("id"))
        rows.append({"id": record.get("id"), "kind": record.get("kind"), "address": record.get("address"),
                     "who": record.get("who"), "at": record.get("at"),
                     "status": done.get("outcome") if done else "pending",
                     "reason": done.get("reason") if done else None})
    return rows


def find_by_request(project_dir, request_id: str) -> dict | None:
    records, _ = store.read_jsonl(store.cockpit_path(project_dir, "decisions.jsonl"))
    return next((r for r in records if r.get("request_id") == request_id), None)


def conflicts(project_dir, question: dict | None) -> list[str]:
    """A web decision and a different lead.toml entry for the same question (names and numbers only)."""
    summary = store.lead_toml_summary(project_dir)
    if question is None:
        return []
    web = [d for d in decision_statuses(project_dir)
           if d["status"] == "pending" and str(d["kind"]).startswith(question["kind"])]
    table = summary["tables"].get(question["kind"])
    if not web or not table:
        return []
    for key in ("round", "plan"):
        if key in table and question["address"] is not None and table[key] != question["address"]:
            return []
    return [f"lead.toml also answers the {question['kind']} question ({', '.join(f'{k} = {v}' for k, v in table.items())}); "
            "your decision from the page wins and the lead.toml entry is ignored."]
