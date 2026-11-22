"""Every DS override through the page: guard, reason, audit, label, once, stale (cases 31-35, 38-40, 42-48)."""

import pytest
from cockpit_finish_helpers import GUIDE_CONTINUE, PLAN_APPROVE, FinishWorld, iterating_world
from orchestrator_helpers import InProcessSandbox

from kaggle_agent.agents.prompting import REQUEST_NOTE, STEERING_NOTE
from kaggle_agent.web import audit, override_writes, store
from kaggle_agent.web.writes import WRITE_ENDPOINTS

R1 = "toy-1-R1-c1"
CHOOSE = {"attempt": R1, "reason": "simpler model", "confirm": True}
BAR = {"scope": "round", "round": 1, "attempt": R1, "reason": "the gain is real", "confirm": True}
STEER = {"steering": {"focus": "feature_engineering", "pin_features": ["x1"], "avoid_features": ["id"]}}
TRY = {"request": {"hypothesis": "interaction of x1 and x2", "constraints": ["no new libraries"]}}


def applied(fw):
    return fw.lines("decisions-applied.jsonl")


def status_of(fw, decision_id):
    return next(o["status"] for o in fw.get_json("/api/overrides/pending")["overrides"] if o["id"] == decision_id)


# 31 (the page side of the choice before the report; the orchestrator side is in test_orchestrator_finalize)
def test_a_pick_made_while_iterating_is_one_decision_applied_once_and_labelled(tmp_path):
    fw = iterating_world(tmp_path)
    data = fw.ok("/api/decisions/choose-model", CHOOSE)
    assert data["needs_finalize"] is False and data["duplicate"] is False
    assert status_of(fw, data["decision"]) == "pending"
    assert [x["label"] for x in fw.get_json("/api/runs/toy-1/labels")["labels"]] == ["chosen by the DS"]
    fw.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    fw.world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    assert fw.world.run("resume", "--supervised")[0] == 10
    marks = [m for m in applied(fw) if m["decision"] == data["decision"]]
    assert len(marks) == 1 and marks[0]["outcome"] == "applied"
    assert status_of(fw, data["decision"]) == "applied"


# 32
def test_a_pick_after_the_report_asks_for_finalize_and_keeps_both_choices_visible(tmp_path):
    fw = FinishWorld(tmp_path)
    assert fw.post("/api/jobs/finalize").status_code == 409  # nothing picked yet
    data = fw.ok("/api/decisions/choose-model", CHOOSE)
    assert data["needs_finalize"] is True
    assert any("Finalize" in w for w in data["warnings"])
    before = fw.registry_hash()
    done = fw.wait_job(fw.ok("/api/jobs/finalize")["job"])
    assert done["status"] == "done"
    assert fw.registry_hash() != before  # the orchestrator persisted the pick: it is the one writer
    choice = fw.get_json("/api/runs/toy-1/labels")["choice"]
    assert choice["program"] == "toy-1-R0-c1" and choice["ds"]["attempt"] == R1 and choice["ds"]["when"] == "after the report"
    assert fw.get_json("/api/runs/toy-1/bundle")["chosen_by"] == "the DS"
    report = fw.client.get("/api/runs/toy-1/report.md").get_data(as_text=True)
    assert "chosen by the DS" in report and "program" in report.lower()


# 33
def test_choose_model_is_validated_warns_and_is_audited(tmp_path, monkeypatch):
    fw = FinishWorld(tmp_path)
    assert fw.post("/api/decisions/choose-model", {"attempt": R1, "confirm": True}).status_code == 400  # no reason
    assert fw.post("/api/decisions/choose-model", {"attempt": R1, "reason": "x"}).status_code == 400  # no confirm
    assert fw.post("/api/decisions/choose-model", {**CHOOSE, "attempt": "toy-1-nope"}).status_code == 404
    assert fw.lines("decisions.jsonl") == [] and fw.lines("audit.jsonl") == []
    monkeypatch.setattr(override_writes, "_scores", lambda src, cycle: {"toy-1-R0-c1": 0.9, R1: 0.5})
    data = fw.ok("/api/decisions/choose-model", CHOOSE)  # a worse pick is accepted, with a warning
    assert any("scores 0.5" in w and "chosen by the DS" in w for w in data["warnings"])
    assert [a["what"] for a in fw.lines("audit.jsonl")] == ["decision.choose_model"]
    # an attempt of another run is technical: it cannot be the model of this one
    fw.ok("/api/run/fork", {"from_run": "toy-1"})
    response = fw.post("/api/decisions/choose-model", {**CHOOSE, "request_id": "other-request-01"})
    assert response.status_code == 400 and "another run" in response.get_json()["error"]["message"]


def failed_world(tmp_path) -> FinishWorld:
    state = {"fail": False}
    fw = FinishWorld(tmp_path, stop=False, sandbox=InProcessSandbox(fail_when=lambda stage: state["fail"]))
    assert fw.world.run("start", "--supervised")[0] == 10
    fw.world.lead(PLAN_APPROVE)
    assert fw.world.run("resume", "--supervised")[0] == 10
    state["fail"] = True
    fw.world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    fw.world.run("resume", "--supervised")
    state["fail"] = False
    fw.state = state
    return fw


def test_a_failed_attempt_cannot_be_the_final_model(tmp_path):
    fw = failed_world(tmp_path)
    failed = [a for a in fw.attempts() if fw.registry().get_attempt(a).outcome == "failed"]
    assert failed
    response = fw.post("/api/decisions/choose-model", {**CHOOSE, "attempt": failed[0]})
    assert response.status_code == 400 and "failed" in response.get_json()["error"]["message"]
    assert fw.lines("decisions.jsonl") == []


# 34
def test_the_bar_panel_shows_the_numbers_and_the_override_is_applied_without_changing_the_stored_row(tmp_path):
    fw = iterating_world(tmp_path)
    panel = fw.get_json("/api/runs/toy-1/bar-override")
    assert panel["available"] and panel["is_real"] is False and panel["wins"] == 0 and panel["bar"] > 0
    assert panel["differences"] and panel["candidate"] == R1
    assert "Accept despite the bar" in fw.page("/cycles/toy-1/rounds/toy-1-R1")
    before = fw.registry_hash()
    data = fw.ok("/api/decisions/bar-override", BAR)
    assert fw.registry_hash() == before  # the web never writes the registry
    fw.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert status_of(fw, data["decision"]) == "applied"
    assert "bar overridden by the DS" in [x["label"] for x in fw.get_json("/api/runs/toy-1/labels")["labels"]]
    reg = fw.registry()
    try:
        row = [c for c in reg.list_comparisons("toy-1") if c.candidate_attempt_id == R1][0]
    finally:
        reg.close()
    assert row.is_real is False  # the stored comparison is as the engine wrote it
    assert fw.post("/api/decisions/bar-override", {**BAR, "request_id": "again-0001"}).status_code == 409  # now stale


# 35
def test_a_stale_override_is_409_and_nothing_is_saved(tmp_path):
    fw = iterating_world(tmp_path, to_round=2)
    response = fw.post("/api/decisions/bar-override", BAR)  # round 1 is not the last complete round any more
    assert response.status_code == 409 and "stale" in response.get_json()["error"]["message"]
    assert fw.lines("decisions.jsonl") == [] and fw.lines("audit.jsonl") == []
    wrong = fw.post("/api/decisions/bar-override", {**BAR, "round": 2, "attempt": R1})
    assert wrong.status_code == 400 and "toy-1-R2-c1" in wrong.get_json()["error"]["message"]


# 36
def test_a_cycle_scope_override_is_a_standing_rule_from_a_round(tmp_path):
    fw = iterating_world(tmp_path)
    assert fw.post("/api/decisions/bar-override", {"scope": "cycle", "round": 1, "confirm": True}).status_code == 400
    data = fw.ok("/api/decisions/bar-override", {"scope": "cycle", "round": 1, "reason": "keep going", "confirm": True})
    assert any("to the end of this cycle" in w for w in data["warnings"])
    assert fw.post("/api/decisions/bar-override", {"scope": "cycle", "round": 0, "reason": "x", "confirm": True}
                   ).status_code == 400


# 38
def test_steering_reaches_the_roles_and_is_labelled_and_a_clear_removes_it(tmp_path):
    fw = iterating_world(tmp_path)
    data = fw.ok("/api/decisions/steering", STEER)
    assert any("nothing checks the code" in w for w in data["warnings"])
    mark = len(fw.world.claude.prompts)
    fw.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    prompts = fw.world.claude.prompts[mark:]
    assert prompts and any(STEERING_NOTE in p and "feature_engineering" in p for p in prompts)
    assert "steered by the DS" in [x["label"] for x in fw.get_json("/api/runs/toy-1/labels")["labels"]]
    other = iterating_world(tmp_path / "cleared")
    other.ok("/api/decisions/steering", STEER)
    other.ok("/api/decisions/steering", {"clear": True})  # the latest record wins: nothing is passed to the roles
    mark = len(other.world.claude.prompts)
    other.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert other.world.claude.prompts[mark:] and not any(STEERING_NOTE in p for p in other.world.claude.prompts[mark:])
    assert "steered by the DS" not in [x["label"] for x in other.get_json("/api/runs/toy-1/labels")["labels"]]


# 39
def test_steering_is_validated_and_never_blocked_for_a_soft_reason(tmp_path):
    fw = iterating_world(tmp_path)
    clash = fw.post("/api/decisions/steering", {"steering": {"pin_features": ["x1"], "avoid_features": ["x1"]}})
    assert clash.status_code == 400 and "x1" in clash.get_json()["error"]["message"]
    focus = fw.post("/api/decisions/steering", {"steering": {"focus": "magic"}})
    assert focus.status_code == 400 and "feature_engineering" in focus.get_json()["error"]["message"]
    assert fw.post("/api/decisions/steering", {"steering": {}}).status_code == 400  # empty: use clear
    assert fw.lines("decisions.jsonl") == [] and fw.lines("audit.jsonl") == []
    data = fw.ok("/api/decisions/steering", {"steering": {"allowed_model_families": ["catboost"]}})
    assert data["duplicate"] is False
    assert [a["what"] for a in fw.lines("audit.jsonl")] == ["decision.steering"]


# 40
def test_try_this_is_bound_once_to_the_next_round_and_reaches_the_two_candidate_roles(tmp_path):
    fw = iterating_world(tmp_path)
    data = fw.ok("/api/decisions/try-this", TRY)
    [record] = fw.lines("decisions.jsonl")
    assert record["from_round"] == 2 and record["request"]["hypothesis"] == "interaction of x1 and x2"
    mark = len(fw.world.claude.prompts)
    fw.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    carrying = [k for k, p in zip(fw.world.claude.kinds[mark:], fw.world.claude.prompts[mark:]) if REQUEST_NOTE in p]
    assert carrying == ["feature_engineering", "modeling"]
    assert [m["bound_to"] for m in applied(fw) if m["decision"] == data["decision"]] == ["toy-1-R2-c1"]
    assert fw.post("/api/decisions/try-this", {"request": {"hypothesis": ""}}).status_code == 400


# 42
def test_a_failed_attempt_is_retried_as_a_new_attempt_and_a_run_blocks_it(tmp_path):
    fw = failed_world(tmp_path)
    failed = [a for a in fw.attempts() if fw.registry().get_attempt(a).outcome == "failed"][0]
    fw.controller.active = lambda: True
    assert fw.post("/api/jobs/rerun", {"attempt": failed}).status_code == 409
    fw.controller.active = fw.controller.lock_held
    done = fw.wait_job(fw.ok("/api/jobs/rerun", {"attempt": failed})["job"])
    assert done["result"]["ok"] is True and done["result"]["origin"] == failed
    assert fw.registry().get_attempt(failed).outcome == "failed"
    assert fw.post("/api/jobs/rerun", {"attempt": "toy-9-R1-c1"}).status_code in (400, 404)


# 43
def test_a_seed_rerun_is_labelled_and_the_repeat_names_the_existing_attempt(tmp_path):
    fw = FinishWorld(tmp_path)
    assert fw.post("/api/jobs/rerun", {"attempt": R1, "seed_offset": 0}).status_code == 400
    done = fw.wait_job(fw.ok("/api/jobs/rerun", {"attempt": R1, "seed_offset": 2})["job"])
    assert done["result"]["attempt"] == "toy-1-R1-c1-rr2"
    config = fw.registry().get_attempt("toy-1-R1-c1-rr2").executions[-1].config
    assert config["seed_offset"] == 2 and config["rerun_of"] == R1
    assert "re-run with another seed" in [x["label"] for x in fw.get_json("/api/runs/toy-1/labels")["labels"]]
    again = fw.wait_job(fw.ok("/api/jobs/rerun", {"attempt": R1, "seed_offset": 2})["job"])
    assert again["result"]["ok"] is False and again["result"]["existing"] == "toy-1-R1-c1-rr2"


ENDPOINTS = {
    "/api/holdout/unseal": {"kind": "first", "confirm": True},
    "/api/decisions/choose-model": CHOOSE, "/api/decisions/bar-override": BAR, "/api/decisions/steering": STEER,
    "/api/decisions/try-this": TRY, "/api/run/fork": {"from_run": "toy-1"}, "/api/jobs/reproduce": {},
    "/api/jobs/rerun": {"attempt": R1, "seed_offset": 1}, "/api/jobs/finalize": {},
    "/api/submission/preview": {}, "/api/submission/confirm": {"sha256": "0" * 64, "confirm": True, "typed": "SEND"},
    "/api/submission/recover": {"action": "not_sent", "reason": "x", "confirm": True},
    "/api/jobs/predict": {},
}


# 44
@pytest.mark.parametrize("path", sorted(ENDPOINTS))
def test_every_new_write_endpoint_needs_the_token_and_the_origin_and_is_post_only(tmp_path, path):
    fw = FinishWorld(tmp_path)
    assert path in WRITE_ENDPOINTS
    before = fw.registry_hash()
    body = ENDPOINTS[path]
    assert fw.post(path, body, token=False).status_code == 403
    assert fw.post(path, body, token="bad-token").status_code == 403
    assert fw.post(path, body, origin="http://evil.example").status_code == 403
    assert fw.client.get(path).status_code in ((404, 405) if path.startswith("/api/jobs/") else (405,))  # /jobs/<id>
    for name in ("audit.jsonl", "decisions.jsonl", "holdout-reads.jsonl", "limit-changes.jsonl"):
        assert fw.lines(name) == [], name
    assert fw.controller.argvs == [] and fw.registry_hash() == before and not (fw.project / "cycles.json").exists()


# 45
def test_a_reason_is_required_where_risky_and_nowhere_else(tmp_path):
    fw = iterating_world(tmp_path)
    for path, body in (("/api/decisions/choose-model", {"attempt": R1, "confirm": True}),
                       ("/api/decisions/bar-override", {**BAR, "reason": ""}),
                       ("/api/decisions/bar-override", {"scope": "cycle", "round": 1, "confirm": True})):
        response = fw.post(path, body)
        assert response.status_code == 400 and response.get_json()["error"]["message"].startswith("reason:"), path
    assert fw.lines("decisions.jsonl") == []
    fw.ok("/api/decisions/steering", STEER)  # no reason needed
    fw.ok("/api/decisions/try-this", TRY)
    fw.ok("/api/decisions/choose-model", CHOOSE)  # with a reason it succeeds
    fw.ok("/api/decisions/bar-override", BAR)


# 46
def test_the_audit_line_comes_first_and_when_it_cannot_be_written_nothing_is_saved(tmp_path, monkeypatch):
    fw = iterating_world(tmp_path)
    seen = []
    real = store.append_jsonl

    def spy(path, record, **kwargs):
        if path.name == "decisions.jsonl":
            seen.append([a["what"] for a in fw.lines("audit.jsonl")])
        return real(path, record, **kwargs)

    monkeypatch.setattr(store, "append_jsonl", spy)
    for path, body in (("/api/decisions/choose-model", CHOOSE), ("/api/decisions/bar-override", BAR),
                       ("/api/decisions/steering", STEER), ("/api/decisions/try-this", TRY)):
        fw.ok(path, body)
    assert [s[-1] for s in seen] == ["decision.choose_model", "decision.bar_override", "decision.steering",
                                     "decision.try_this"]
    for line in fw.lines("audit.jsonl"):
        assert set(line) >= {"who", "at", "what", "id"} and "why" in line
    assert fw.lines("audit.jsonl")[0]["why"] == "simpler model"

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(audit, "record", broken)
    count = len(fw.lines("decisions.jsonl"))
    response = fw.post("/api/decisions/steering", {"steering": {"focus": "model_tuning"}, "request_id": "after-fail-01"})
    assert response.status_code >= 500 and len(fw.lines("decisions.jsonl")) == count


# 47
def test_a_duplicate_request_id_returns_the_first_answer_and_one_record(tmp_path):
    fw = iterating_world(tmp_path)
    first = fw.ok("/api/decisions/steering", {**STEER, "request_id": "same-request-1"})
    second = fw.ok("/api/decisions/steering", {**STEER, "request_id": "same-request-1"})
    assert second["duplicate"] is True and second["decision"] == first["decision"]
    assert len(fw.lines("decisions.jsonl")) == 1 and len(fw.lines("audit.jsonl")) == 1
    fw.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    fw.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert len([m for m in applied(fw) if m["decision"] == first["decision"]]) == 1


# 48
def test_a_stale_decision_is_shown_as_stale_and_not_labelled(tmp_path):
    from overrides_helpers import decide

    fw = iterating_world(tmp_path, to_round=2)
    old = decide(fw.world, "bar_override", scope="round", round=1, attempt=R1, reason="too late")
    fw.resume(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    assert status_of(fw, old["id"]) == "stale"
    assert "bar overridden by the DS" not in [x["label"] for x in fw.get_json("/api/runs/toy-1/labels")["labels"]]
