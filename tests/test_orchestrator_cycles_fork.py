"""cycles.json, the carry-over store and the fork command (cases 6, 24, 25, 26, 27, 28)."""

import hashlib
import json
import sqlite3

from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, REVIEW_ACCEPT, World, drive_to_review
from overrides_helpers import FIRST, SECOND, attempts, cockpit_json, fork, registry, reads, stopped_world

from kaggle_agent.orchestrator.cycles import read_carry, read_cycles


def protocol_hashes(world):
    reg = registry(world)
    try:
        return {c.id: c.protocol_hash for c in reg.list_cycles()}
    finally:
        reg.close()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin_rows(world):
    db = sqlite3.connect(world.project / "registry.db")
    try:
        return [db.execute(f"SELECT * FROM {t} WHERE cycle_id = ?", (FIRST,)).fetchall()
                for t in ("attempts", "comparisons")]
    finally:
        db.close()


def finish_fork(world, cycle=SECOND):
    for n in (1, 2):
        world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=n))
        assert world.run("resume", "--supervised")[0] == 10
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2) + REVIEW_ACCEPT)
    code, text = world.run("resume", "--supervised")
    assert code == 0, text


# 25
def test_a_fork_with_the_same_protocol_is_the_next_cycle_with_the_same_hash_and_is_idempotent(tmp_path):
    world = stopped_world(tmp_path)
    before = origin_rows(world)
    code, text = fork(world, "toy-1-R1-c1")
    assert code == 10 and "toy-2" in text
    entries = read_cycles(world.project)
    assert [e["cycle_id"] for e in entries] == [SECOND]
    entry = entries[0]
    assert entry["protocol_label"] == "same" and entry["changes"] == {} and entry["request_id"] == "request-0001"
    assert entry["origin"] == {"cycle_id": FIRST, "attempt_id": "toy-1-R1-c1"}
    hashes = protocol_hashes(world)
    assert hashes[SECOND] == hashes[FIRST]
    assert origin_rows(world) == before  # the origin rows are byte-identical
    # the same request again makes no second cycle
    world.lead(PLAN_APPROVE)
    fork(world, "toy-1-R1-c1")
    assert [e["cycle_id"] for e in read_cycles(world.project)] == [SECOND]
    assert set(protocol_hashes(world)) == {FIRST, SECOND}


# 26
def test_a_fork_with_a_new_threshold_k_freezes_a_new_labelled_protocol(tmp_path):
    world = stopped_world(tmp_path)
    config_before, origin_before = file_hash(world.config), origin_rows(world)
    code, _ = fork(world, None, "request-0009", "--threshold-k", "2.5", "--reason", "a stricter bar")
    assert code == 10
    entry = read_cycles(world.project)[0]
    assert entry["protocol_label"] == "changed" and entry["changes"] == {"threshold_k": [1.0, 2.5]}
    assert entry["reason"] == "a stricter bar" and entry["protocol"]["threshold_k"] == 2.5
    hashes = protocol_hashes(world)
    assert hashes[SECOND] != hashes[FIRST]
    carry = read_carry(world.project, SECOND)
    assert carry.same_protocol is False and "another protocol" in carry.protocol_note
    assert file_hash(world.config) == config_before and origin_rows(world) == origin_before
    assert "origin: forked from toy-1; protocol: changed" in world.run("status")[1]


# 27
def test_a_fork_is_refused_for_technical_reasons_only_and_creates_nothing(tmp_path):
    world = stopped_world(tmp_path)
    code, text = fork(world, "no-such-attempt")
    assert code == 1 and "there is no attempt no-such-attempt" in text and "no cycle was created" in text
    code, text = fork(world, None, "request-0002", "--threshold-k", "2.0")
    assert code == 1 and "needs a reason" in text
    assert not (world.project / "cycles.json").exists()
    drive_forked = fork(world)
    assert drive_forked[0] == 10
    code, text = fork(world, None, "request-0003")  # the current cycle is no longer STOPPED
    assert code == 1 and "is not stopped yet" in text
    assert [e["cycle_id"] for e in read_cycles(world.project)] == [SECOND]


# 24
def test_the_carry_over_reaches_the_three_roles_and_the_first_task_starts_from_the_attempt(tmp_path):
    world = stopped_world(tmp_path)
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1) + REVIEW_ACCEPT)
    mark = len(world.claude.prompts)
    assert fork(world, "toy-1-R1-c1")[0] == 10
    prompts = world.claude.prompts[mark:]
    kinds = world.claude.kinds[mark:]
    seen = {}
    for kind, prompt in zip(kinds, prompts):
        if kind in ("coordinator", "feature_engineering", "modeling"):
            seen.setdefault(kind, prompt)
    assert set(seen) == {"coordinator", "feature_engineering", "modeling"}
    carry = read_carry(world.project, SECOND)
    assert carry.origin_attempt == "toy-1-R1-c1" and carry.same_protocol is True
    digest = carry.attempt[0]
    assert digest["id"] == "toy-1-R1-c1" and digest["hypothesis"].startswith("hypothesis")
    assert "validation_score" in digest and carry.top_attempts and carry.reviewer
    for kind, prompt in seen.items():
        assert '"carry"' in prompt and '"origin_attempt": "toy-1-R1-c1"' in prompt, kind
        assert '"same_protocol": true' in prompt and digest["hypothesis"] in prompt, kind
        assert "no code is carried" in prompt
    # descriptions and numbers only: the carry file holds no code
    assert "def " not in (world.project / "carry" / "toy-2.json").read_text(encoding="utf-8")


# 6
def test_the_labels_of_a_same_protocol_fork_and_of_a_new_protocol_fork_differ(tmp_path):
    from kaggle_agent.overrides import run_labels

    world = stopped_world(tmp_path)
    reads(world)  # the origin read the holdout
    assert fork(world)[0] == 10
    finish_fork(world)
    assert fork(world, None, "request-0005", "--threshold-k", "3.0", "--reason", "new bar")[0] == 10
    entries = read_cycles(world.project)
    assert [e["cycle_id"] for e in entries] == [SECOND, "toy-3"]
    hashes = protocol_hashes(world)
    same = run_labels(SECOND, FIRST, [], {}, [], entries[0], {"origin_read": True})
    new = run_labels("toy-3", FIRST, [], {}, [], entries[1], {"origin_read": True, "protocol_hash": hashes["toy-3"]})
    assert [x["label"] for x in same] == ["holdout re-read"] and same[0]["repeated_look"] is True
    assert [x["label"] for x in new] == ["new protocol (set by the DS)"]
    assert new[0]["protocol_hash"] == hashes["toy-3"] != hashes[FIRST]


# 28
def test_a_project_without_cycles_json_opens_and_resumes_as_the_legacy_single_cycle(tmp_path):
    world = World(tmp_path)
    assert world.run("start")[0] == 10
    assert not (world.project / "cycles.json").exists() and read_cycles(world.project) == []
    code, text = world.run("status")
    assert code == 0 and "cycle: toy-1" in text and "origin" not in text
    world.lead(PLAN_APPROVE)
    assert world.run("resume")[0] == 10
    assert not (world.project / "cycles.json").exists()
    assert attempts(world) and world.state() == "AWAITING_GUIDANCE"
    assert json.loads((world.project / "project.json").read_text())["cycle_id"] == "toy-1"
