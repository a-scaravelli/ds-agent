"""Limit extension: effective limits, running cycle, after stop and resume, old cycles, validation."""

import json

from cockpit_live_helpers import live, live_factory, registry_summary, wait_for_stage  # noqa: F401  (fixtures)

from kaggle_agent.orchestrator.cockpit import overlay_cycle
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web import data

EXTEND = {"rounds": 3, "candidates": 2, "stagnation": 5}


def stored_cycle(project):
    registry = SqliteRegistry(project.registry_path(), read_only=True)
    try:
        return registry.list_cycles()[0]
    finally:
        registry.close()


def start_blocked_after_the_plan(project):
    """Approve the plan and resume with every Claude call blocked: the run is inside round 1."""
    project.ok("/api/run/start")
    project.wait_exit(10)
    project.decide_plan()
    block = project.tmp / "block"
    scenario = json.loads(project.scenario_file.read_text(encoding="utf-8"))
    project.set_scenario({**scenario, "block_file": str(block)})
    block.write_text("x", encoding="utf-8")
    last = project.events()[-1]["id"]
    project.ok("/api/run/resume")
    wait_for_stage(project, last)
    return block


def extend(project, values=None, **extra):
    return project.ok("/api/limits", {"values": values or EXTEND, "reason": "the data needs more rounds",
                                      "confirm": True, **extra})


# 33
def test_a_limit_extension_is_honoured_by_the_running_cycle(live):
    block = start_blocked_after_the_plan(live)
    before = stored_cycle(live)
    assert (before.iterations, before.stagnation_k, before.parameters["candidates"]) == (2, 2, 1)
    result = extend(live)
    assert result["duplicate"] is False
    shown = live.api("/api/run")["limits"]
    assert shown["base"] == {"rounds": 2, "candidates": 1, "stagnation": 2}
    assert shown["effective"] == EXTEND and shown["changes"][0]["reason"] == "the data needs more rounds"
    block.unlink()
    live.wait_exit(10)
    for n in (1, 2):
        live.ok("/api/decisions/guidance", {"action": "continue", "address": n})
        live.ok("/api/run/resume")
        live.wait_exit(10)
    # round 3 exists now: with the stored limits the cycle would be at its review after round 2
    assert live.api("/api/decisions/pending")["question"] == {
        **live.api("/api/decisions/pending")["question"], "kind": "guidance", "address": 3}
    summary = registry_summary(live.registry_path())
    assert summary["state"] == "AWAITING_GUIDANCE" and summary["rounds"] == 4  # the baseline round and 3 rounds
    assert summary["candidates"][-2:] == [2, 2] and summary["candidates"][0] == 1  # the new candidates per round
    after = stored_cycle(live)
    assert (after.iterations, after.stagnation_k, after.parameters) == (before.iterations, before.stagnation_k,
                                                                      before.parameters)  # nothing rewritten
    assert any(e["type"] == "limits_applied" and e["data"]["rounds"] == 3 for e in live.events())
    live.ok("/api/decisions/guidance", {"action": "continue", "address": 3})
    live.ok("/api/run/resume")
    live.wait_exit(10)
    assert live.api("/api/decisions/pending")["question"]["kind"] == "review"


# 33
def test_a_limit_extension_is_used_after_a_stop_and_resume(live):
    block = start_blocked_after_the_plan(live)
    extend(live)
    live.ok("/api/run/stop", {"confirm": True, "reason": "restart with the new limits"})
    block.unlink()
    live.wait_exit(15)
    live.set_scenario({})
    live.ok("/api/run/resume")
    live.wait_exit(10)
    for n in (1, 2):
        live.ok("/api/decisions/guidance", {"action": "continue", "address": n})
        live.ok("/api/run/resume")
        live.wait_exit(10)
    assert live.api("/api/decisions/pending")["question"]["address"] == 3  # a third round, as extended
    assert live.api("/api/run")["limits"]["effective"] == EXTEND


# 33
def test_the_effective_limits_of_the_page_and_the_orchestrator_agree_and_the_last_record_wins(live):
    block = start_blocked_after_the_plan(live)
    extend(live, {"rounds": 4})
    extend(live, {"rounds": 6, "stagnation": 7})
    cycle = stored_cycle(live)
    records, _ = __import__("kaggle_agent.web.store", fromlist=["store"]).read_jsonl(
        live.dir / "cockpit" / "limit-changes.jsonl")
    shown = overlay_cycle(cycle, records)  # what the orchestrator uses
    page = data.overlay_limits({"rounds": cycle.iterations, "candidates": cycle.parameters["candidates"],
                                "stagnation": cycle.stagnation_k}, records)  # what the page shows
    assert (shown.iterations, shown.parameters["candidates"], shown.stagnation_k) == (
        page["rounds"], page["candidates"], page["stagnation"]) == (6, 1, 7)
    block.unlink()
    live.wait_exit(10)


# 34
def test_an_old_cycle_without_a_change_file_is_unchanged_and_the_overlay_touches_only_the_limits(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    live.decide_plan()
    live.ok("/api/run/resume")
    live.wait_exit(10)
    assert not (live.dir / "cockpit" / "limit-changes.jsonl").exists()
    cycle = stored_cycle(live)
    shown = live.api("/api/run")["limits"]
    assert shown["effective"] == shown["base"] == {"rounds": 2, "candidates": 1, "stagnation": 2}
    assert shown["changes"] == []
    assert overlay_cycle(cycle, []) == cycle
    records = [{"values": {"rounds": 9, "candidates": 4, "stagnation": 8}}]
    changed = overlay_cycle(cycle, records)
    assert changed.iterations == 9 and changed.stagnation_k == 8 and changed.parameters["candidates"] == 4
    for field in cycle.__dataclass_fields__:
        if field in ("iterations", "stagnation_k", "parameters"):
            continue
        assert getattr(changed, field) == getattr(cycle, field), field  # protocol_hash and the rest as stored
    assert changed.protocol_hash == cycle.protocol_hash
    assert {k: v for k, v in changed.parameters.items() if k != "candidates"} == {
        k: v for k, v in cycle.parameters.items() if k != "candidates"}
    assert stored_cycle(live).parameters == cycle.parameters


# 34
def test_a_cycle_that_has_not_stored_its_limits_yet_is_never_overlaid(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    cycle = stored_cycle(live)
    if cycle.parameters is None:
        warnings = extend(live)["warnings"]
        assert any("not stored its limits yet" in w for w in warnings)
        assert live.api("/api/run")["limits"]["applies"] is False


# 35
def test_limit_validation_and_confirmation(live):
    block = start_blocked_after_the_plan(live)
    path = live.dir / "cockpit" / "limit-changes.jsonl"
    audit_before = len(live.api("/api/audit")["entries"])
    good = {"values": {"rounds": 3}, "reason": "more", "confirm": True}
    bad = [
        {**good, "values": {"rounds": "3"}},
        {**good, "values": {"rounds": 2.5}},
        {**good, "values": {"rounds": True}},
        {**good, "values": {"rounds": 0}},
        {**good, "values": {"speed": 3}},
        {**good, "values": {}},
        {**good, "reason": ""},
        {key: value for key, value in good.items() if key != "reason"},
        {**good, "confirm": False},
        {key: value for key, value in good.items() if key != "confirm"},
    ]
    for body in bad:
        status, answer = live.post("/api/limits", body)
        assert status == 400, body
    assert not path.exists() and len(live.api("/api/audit")["entries"]) == audit_before
    block.unlink()
    live.wait_exit(10)
    live.ok("/api/decisions/guidance", {"action": "continue", "address": 1})
    live.ok("/api/run/resume")
    live.wait_exit(10)
    # fewer rounds than already done: allowed, with a visible warning (only technical reasons block)
    result = live.ok("/api/limits", {"values": {"rounds": 1}, "reason": "enough", "confirm": True})
    assert any("below" in w for w in result["warnings"])
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
