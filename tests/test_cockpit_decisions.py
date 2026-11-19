"""Decisions from the page: plan (approve, revise, own), guidance, review, once, stale, lead.toml, auto resume."""

import json

from cockpit_live_helpers import finish_cycle, live, live_factory, registry_summary  # noqa: F401  (fixtures)

from kaggle_agent.orchestrator.lead_file import Ledger
from kaggle_agent.registry import SqliteRegistry

OWN_PLAN = {"metric": "accuracy", "validation": "3-fold random validation", "feature_engineering": ["none"],
            "models": ["majority"], "baseline": "majority class", "tasks": ["Try my own idea"]}


def to_plan_question(project):
    project.ok("/api/run/start")
    project.wait_exit(10)


def to_guidance(project, round_number=1):
    to_plan_question(project)
    project.decide_plan()
    project.ok("/api/run/resume")
    project.wait_exit(10)
    assert project.api("/api/decisions/pending")["question"]["address"] == round_number


def decisions_file(project):
    path = project.dir / "cockpit" / "decisions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def applied(project):
    path = project.dir / "cockpit" / "decisions-applied.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def coordinator_traces(project):
    registry = SqliteRegistry(project.registry_path(), read_only=True)
    try:
        cycle = registry.list_cycles()[0]
        return registry.list_role_traces(cycle.id, role="coordinator")
    finally:
        registry.close()


# 26
def test_auto_resume_on_starts_exactly_one_resume_and_off_only_offers_the_button(live):
    to_guidance(live)
    assert 'id="resume-offer"' not in live.get("/decisions")[1]  # nothing was decided yet
    assert live.ok("/api/settings", {"auto_resume": True}) == {"auto_resume": True}
    pid_before = live.run_state()["pid"]
    body = {"action": "continue", "address": 1, "request_id": "same-request-id-1"}
    first = live.ok("/api/decisions/guidance", body)
    second = live.ok("/api/decisions/guidance", body)  # the double submit
    assert first["resumed"] is True and second["duplicate"] is True and second["resumed"] is False
    state = live.wait_exit(10)
    assert state["pid"] != pid_before
    starts = [e for e in live.events() if e["type"] == "run_start"]
    assert len(starts) == 3  # start, resume after the plan, ONE resume after the guidance
    assert len(decisions_file(live)) == 2  # the plan and the guidance, once each
    # off: the decision is saved and the page only offers the button
    assert live.ok("/api/settings", {"auto_resume": False}) == {"auto_resume": False}
    pid = live.run_state()["pid"]
    result = live.ok("/api/decisions/guidance", {"action": "continue", "address": 2})
    assert result["resumed"] is False
    assert live.run_state()["pid"] == pid and not live.controller.active()
    offer = live.get("/decisions")[1]
    assert 'id="resume-offer"' in offer and "Resume the run" in offer


# 27
def test_revise_adds_a_ledger_entry_and_the_coordinator_replans(live):
    to_plan_question(live)
    live.decide_plan("revise", 1, text="use simpler features")
    live.ok("/api/run/resume")
    live.wait_exit(10)
    ledger = Ledger(live.dir / "lead-ledger.json")
    assert [r["plan"] for r in ledger.revisions()] == [1] and ledger.texts() == ("use simpler features",)
    assert live.api("/api/decisions/pending")["question"]["address"] == 2  # plan number 2 now waits
    assert len(coordinator_traces(live)) == 2
    assert any(a["outcome"] == "applied" for a in applied(live))
    assert "decision_applied" in live.event_types()


# 27
def test_an_own_plan_skips_the_coordinator_and_is_recorded_as_the_ds_plan(live):
    to_plan_question(live)
    body = {"action": "own", "address": 1, "plan": OWN_PLAN}
    assert live.post("/api/decisions/plan", body)[0] == 400  # a reason and the confirmation are required
    assert live.post("/api/decisions/plan", {**body, "reason": "I know this data"})[0] == 400
    assert live.post("/api/decisions/plan", {**body, "reason": "I know this data", "confirm": True,
                                             "plan": {**OWN_PLAN, "models": []}})[0] == 400
    live.ok("/api/decisions/plan", {**body, "reason": "I know this data", "confirm": True})
    before = len(coordinator_traces(live))
    live.ok("/api/run/resume")
    live.wait_exit(10)  # approved in one step: the cycle goes on to the first round
    traces = coordinator_traces(live)
    assert len(traces) == before + 1  # the own plan is a coordinator trace
    own = traces[-1]
    assert own.input["author"] == "lead" and own.input["plan_number"] == 1 and own.input["reason"] == "I know this data"
    assert own.input_type == "LeadPlan" and set(own.output) >= {"plan", "tasks"}
    assert own.output["tasks"] == ["Try my own idea"] and own.output["plan"]["models"] == ["majority"]
    ledger = Ledger(live.dir / "lead-ledger.json")
    item = ledger.revisions()[0]
    assert item["kind"] == "own" and item["text"] and "approved by the DS" in item.get("note", item["text"]) \
        or "plan written by the DS" in item["text"]
    assert ledger.texts() == () and ledger.plan_number() == 2  # no KeyError, never given to the coordinator
    assert registry_summary(live.registry_path())["state"] == "AWAITING_GUIDANCE"


# 28
def test_guidance_is_recorded_exactly_like_lead_toml(live_factory):
    page = live_factory()
    lead = live_factory()
    for project in (page, lead):
        to_plan_question(project)
    # through the page
    page.decide_plan()
    page.ok("/api/run/resume")
    page.wait_exit(10)
    page.ok("/api/decisions/guidance", {"action": "guidance", "address": 1, "text": "focus on trees"})
    page.ok("/api/run/resume")
    page.wait_exit(10)
    page.ok("/api/decisions/guidance", {"action": "continue", "address": 2})
    page.ok("/api/run/resume")
    page.wait_exit(10)
    # through lead.toml
    plan = '[plan]\ndecision = "approve"\n'
    (lead.dir / "lead.toml").write_text(plan, encoding="utf-8")
    lead.ok("/api/run/resume")
    lead.wait_exit(10)
    (lead.dir / "lead.toml").write_text(
        plan + '[guidance]\nround = 1\naction = "guidance"\ntext = "focus on trees"\n', encoding="utf-8")
    lead.ok("/api/run/resume")
    lead.wait_exit(10)
    (lead.dir / "lead.toml").write_text(plan + '[guidance]\nround = 2\naction = "continue"\n', encoding="utf-8")
    lead.ok("/api/run/resume")
    lead.wait_exit(10)
    a, b = registry_summary(page.registry_path()), registry_summary(lead.registry_path())
    assert a == b
    assert a["guidance"] == ["focus on trees"]  # "continue" is recorded nowhere in the registry
    assert a["state"] == "AWAITING_REVIEW"


# 29
def test_review_accept_needs_the_confirmation_and_stops_the_cycle(live):
    to_guidance(live)
    for n in (1, 2):
        live.ok("/api/decisions/guidance", {"action": "continue", "address": n})
        live.ok("/api/run/resume")
        live.wait_exit(10)
    assert live.api("/api/decisions/pending")["question"]["kind"] == "review"
    assert "Accept" in live.get("/decisions")[1]
    before = len(decisions_file(live))
    status, body = live.post("/api/decisions/review", {})
    assert status == 400 and "confirm" in json.dumps(body)
    assert len(decisions_file(live)) == before and registry_summary(live.registry_path())["state"] == "AWAITING_REVIEW"
    live.ok("/api/decisions/review", {"confirm": True})
    live.ok("/api/run/resume")
    state = live.wait_exit(0)
    assert state["state"] == "finished"
    assert registry_summary(live.registry_path())["state"] == "STOPPED"
    assert "finished" in live.get("/run")[1]


# 30
def test_a_decision_is_applied_exactly_once(live):
    to_plan_question(live)
    body = {"action": "revise", "address": 1, "text": "try again", "request_id": "request-once-0001"}
    live.ok("/api/decisions/plan", body)
    again = live.ok("/api/decisions/plan", body)
    assert again["duplicate"] is True
    assert len(decisions_file(live)) == 1
    audit = [e for e in live.api("/api/audit")["entries"] if e["what"] == "decision.plan_revise"]
    assert len(audit) == 1
    pending = live.api("/api/decisions/pending")["decisions"]
    assert [d["status"] for d in pending] == ["pending"]
    live.ok("/api/run/resume")
    live.wait_exit(10)
    live.ok("/api/run/resume")  # resumed twice
    live.wait_exit(10)
    assert len(Ledger(live.dir / "lead-ledger.json").revisions()) == 1
    assert [a["outcome"] for a in applied(live)] == ["applied"]
    assert [d["status"] for d in live.api("/api/decisions/pending")["decisions"]] == ["applied"]


# 30
def test_a_guidance_decision_makes_one_guidance_row_after_two_resumes(live):
    to_guidance(live)
    live.ok("/api/decisions/guidance", {"action": "guidance", "address": 1, "text": "only once"})
    for _ in range(2):
        live.ok("/api/run/resume")
        live.wait_exit(10)
    assert registry_summary(live.registry_path())["guidance"] == ["only once"]


# 31
def test_stale_decisions_are_ignored_by_the_orchestrator_and_refused_by_the_page(live):
    to_guidance(live)
    live.ok("/api/decisions/guidance", {"action": "continue", "address": 1})
    live.ok("/api/run/resume")
    live.wait_exit(10)  # now the cycle waits at round 2
    assert live.api("/api/decisions/pending")["question"]["address"] == 2
    stale = {"id": "stale-guidance-1", "request_id": "stale-request-1", "kind": "guidance", "address": 1,
             "action": "guidance", "text": "too late", "who": "someone", "at": "2026-01-01T00:00:00+00:00"}
    with open(live.dir / "cockpit" / "decisions.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(stale) + "\n")
    live.ok("/api/run/resume")
    live.wait_exit(10)  # it keeps waiting
    assert registry_summary(live.registry_path())["state"] == "AWAITING_GUIDANCE"
    assert registry_summary(live.registry_path())["guidance"] == []
    assert "decision_stale" in live.event_types()
    assert any(a["decision"] == "stale-guidance-1" and a["outcome"] == "stale" for a in applied(live))
    notices = (live.dir / "NOTICES.txt").read_text(encoding="utf-8")
    assert "stale" in notices.lower() or "round 1" in notices
    # the page refuses stale answers
    for path, body in (("/api/decisions/guidance", {"action": "continue", "address": 1}),
                       ("/api/decisions/plan", {"action": "revise", "address": 1, "text": "old plan"}),
                       ("/api/decisions/plan", {"action": "approve", "address": 1})):
        status, answer = live.post(path, body)
        assert status == 409 and "stale" in answer["error"]["message"], (path, answer)
    assert len(decisions_file(live)) == 3  # plan, round 1 and the planted entry: the stale answers were not saved


# 32
def test_the_web_decision_wins_over_lead_toml_and_the_conflict_is_shown(live):
    to_plan_question(live)
    (live.dir / "lead.toml").write_text(
        '[plan]\ndecision = "revise"\nplan = 1\ntext = "lead.toml wants a new plan"\n', encoding="utf-8")
    live.decide_plan("approve", 1)
    page = live.get("/decisions")[1]
    assert "lead.toml also answers" in page and "your decision from the page wins" in page
    assert live.api("/api/decisions/pending")["conflicts"]
    live.ok("/api/run/resume")
    live.wait_exit(10)
    # approved (web), not revised (lead.toml): the cycle went on to the first round
    assert registry_summary(live.registry_path())["state"] == "AWAITING_GUIDANCE"
    assert Ledger(live.dir / "lead-ledger.json").revisions() == []
    notices = (live.dir / "NOTICES.txt").read_text(encoding="utf-8")
    assert "lead.toml" in notices
    assert any(e["type"] == "decision_stale" and "conflict" in json.dumps(e["data"]) for e in live.events())
