"""Re-run of an attempt: another seed, a retry, duplicates, the run lock (cases 42, 43, 47)."""

import json
import sqlite3

from overrides_helpers import FIRST, applied, attempts, decide, registry, stopped_world

from kaggle_agent.orchestrator.cockpit import RunLock
from kaggle_agent.orchestrator.cockpit import CockpitFiles

ORIGIN = "toy-1-R1-c1"


def rerun(world, tmp_path, *extra, attempt=ORIGIN, offset=2):
    out = tmp_path / "rerun.json"
    out.unlink(missing_ok=True)
    code, text = world.run("rerun", "--attempt", attempt, "--seed-offset", str(offset), "--out", str(out), *extra)
    return code, json.loads(out.read_text(encoding="utf-8")) if out.exists() else text


def detail(world, attempt_id):
    reg = registry(world)
    try:
        return reg.get_attempt(attempt_id)
    finally:
        reg.close()


# 43
def test_a_seed_variant_is_a_new_attempt_on_the_same_bundle_and_a_repeat_is_refused(tmp_path):
    world = stopped_world(tmp_path)
    original = detail(world, ORIGIN)
    code, result = rerun(world, tmp_path)
    assert code == 0 and result["ok"] is True and result["attempt"] == "toy-1-R1-c1-rr2"
    assert result["origin"] == ORIGIN and result["seed_offset"] == 2 and result["outcome"] == "succeeded"
    new = detail(world, result["attempt"])
    config = new.executions[-1].config
    assert new.attempt.parent_id == ORIGIN and config["bundle_id"] == original.executions[-1].config["bundle_id"]
    assert config["seed_offset"] == 2 and config["rerun_of"] == ORIGIN and config["round_id"] is None
    reg = registry(world)
    try:
        assert all(result["attempt"] not in {c.attempt_id for c in r.candidates}
                   for r in reg.list_rounds_with_candidates(FIRST))  # in no round: not part of N, K or M
    finally:
        reg.close()
    assert len(new.executions) == 1  # a fresh execution under the frozen protocol
    assert detail(world, ORIGIN) == original  # the original is untouched
    assert attempts(world)[-1] == "toy-1-R1-c1-rr2"
    # the same request again: refused as a duplicate, naming the attempt that exists
    code, again = rerun(world, tmp_path)
    assert again["ok"] is False and again["duplicate"] is True and again["existing"] == "toy-1-R1-c1-rr2"
    assert attempts(world).count("toy-1-R1-c1-rr2") == 1
    # another offset is another attempt; a succeeded attempt needs an offset of 1 or more
    assert rerun(world, tmp_path, offset=3)[1]["attempt"] == "toy-1-R1-c1-rr3"
    assert "offset of 1 or more" in rerun(world, tmp_path, offset=0)[1]["error"]


# 42
def test_a_retry_marks_its_decision_once_and_a_run_lock_refuses_it(tmp_path):
    world = stopped_world(tmp_path)
    record = decide(world, "rerun", attempt=ORIGIN, seed_offset=1)
    lock = RunLock(CockpitFiles(world.project))
    assert lock.acquire()
    try:
        code, result = rerun(world, tmp_path, "--decision", record["id"], offset=1)
        assert code != 0 or result["ok"] is False
        assert "toy-1-R1-c1-rr1" not in attempts(world) and applied(world) == []
    finally:
        lock.release()
    code, result = rerun(world, tmp_path, "--decision", record["id"], offset=1)
    assert code == 0 and result["ok"] is True
    assert [(m["decision"], m["outcome"]) for m in applied(world)] == [(record["id"], "applied")]
    rerun(world, tmp_path, "--decision", record["id"], offset=1)  # a duplicate: no second marker
    assert len(applied(world)) == 1


# 42
def test_the_rerun_writes_only_new_rows_and_never_touches_the_origin_rows(tmp_path):
    world = stopped_world(tmp_path)
    db = sqlite3.connect(world.project / "registry.db")
    before = db.execute("SELECT * FROM attempts WHERE id = ?", (ORIGIN,)).fetchall()
    rounds = db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
    db.close()
    assert rerun(world, tmp_path)[1]["ok"] is True
    db = sqlite3.connect(world.project / "registry.db")
    try:
        assert db.execute("SELECT * FROM attempts WHERE id = ?", (ORIGIN,)).fetchall() == before
        assert db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == rounds  # in no round, no N, K or M
    finally:
        db.close()


# 42
def test_a_failed_attempt_is_retried_as_a_new_attempt_and_the_failed_one_stays_failed(tmp_path):
    from orchestrator_helpers import PLAN_APPROVE, InProcessSandbox, World

    state = {"fail": False}
    world = World(tmp_path, sandbox=InProcessSandbox(fail_when=lambda stage: state["fail"]))
    assert world.run("start", "--supervised")[0] == 10
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    failed = [a for a in attempts(world) if detail(world, a).outcome == "failed"]
    assert failed == []  # nothing failed yet: make a failed attempt by hand below
    # a run whose sandbox crashes: the attempt of the next round fails
    state["fail"] = True
    world.lead(PLAN_APPROVE + "[guidance]\nround = 1\naction = \"continue\"\n")
    world.run("resume", "--supervised")
    failed = [a for a in attempts(world) if detail(world, a).outcome == "failed"]
    assert failed, attempts(world)
    state["fail"] = False
    code, result = rerun(world, tmp_path, attempt=failed[0], offset=0)
    assert code == 0 and result["ok"] is True and result["origin"] == failed[0] and result["outcome"] == "succeeded"
    assert detail(world, failed[0]).outcome == "failed"
    assert detail(world, result["attempt"]).attempt.parent_id == failed[0]
