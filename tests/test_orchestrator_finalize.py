"""The DS pick: before the report (the report hook) and after it (the finalize job) (cases 31, 32, 33, 47, 49)."""

import json
import sqlite3

from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, REVIEW_ACCEPT, World
from overrides_helpers import FIRST, applied, decide, stopped_world

PICK = "toy-1-R1-c1"
PROGRAM = "toy-1-R0-c1"


def to_last_round(tmp_path) -> World:
    world = World(tmp_path)
    assert world.run("start", "--supervised")[0] == 10
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume", "--supervised")[0] == 10
    return world


def report(world) -> dict:
    reg = world.registry()
    try:
        return reg.get_final_report(FIRST).content
    finally:
        reg.close()


def reads(world) -> int:
    db = sqlite3.connect(world.project / "registry.db")
    try:
        return db.execute("SELECT COUNT(*) FROM holdout_reads").fetchone()[0]
    finally:
        db.close()


def ds_section(world) -> str:
    text = (world.project / "out" / "reports" / "toy-1.md").read_text(encoding="utf-8")
    return text.split("<!-- ds-decisions:begin -->")[1].split("<!-- ds-decisions:end -->")[0]


def finalize(world, tmp_path):
    out = tmp_path / "finalize.json"
    code, _ = world.run("finalize", "--out", str(out))
    return code, json.loads(out.read_text(encoding="utf-8"))


# 31
def test_a_pick_made_before_the_report_is_the_chosen_bundle_persisted_and_read_once(tmp_path):
    world = to_last_round(tmp_path)
    record = decide(world, "choose_model", attempt=PICK, reason="the second idea is easier to explain")
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    assert world.run("resume", "--supervised")[0] == 10  # AWAITING_REVIEW: the report exists
    chosen = report(world)["chosen"]
    assert (chosen["attempt_id"], chosen["chosen_by"], chosen["program_choice"]) == (PICK, "the DS", PROGRAM)
    assert chosen["reason"] == "the second idea is easier to explain"
    assert report(world)["holdout"]["attempt_id"] == PICK  # the one holdout session is the DS pick's
    assert reads(world) == 1
    assert (world.project / "out" / "persisted" / chosen["bundle_id"] / "persisted.json").is_file()
    assert [(m["decision"], m["outcome"]) for m in applied(world)] == [(record["id"], "applied")]
    section = ds_section(world)
    assert "chosen by the DS" in section and PICK in section and PROGRAM in section
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2) + REVIEW_ACCEPT)
    assert world.run("resume", "--supervised")[0] == 0
    assert reads(world) == 1 and len(applied(world)) == 1  # resuming applies nothing twice


# 32
def test_a_pick_made_after_the_report_is_persisted_by_the_finalize_job_and_becomes_the_model(tmp_path):
    world = stopped_world(tmp_path)
    before = report(world)
    assert before["chosen"]["attempt_id"] == PROGRAM and before["chosen"].get("chosen_by") != "the DS"
    decide(world, "choose_model", attempt=PICK, reason="prefer the later attempt")
    code, result = finalize(world, tmp_path)
    assert code == 0 and result == {"ok": True, "attempt": PICK, "bundle": "toy-1-R1-c1-b1", "program_choice": PROGRAM}
    chosen = json.loads((world.project / "out" / "chosen" / "toy-1.json").read_text(encoding="utf-8"))
    assert chosen["attempt"] == PICK and chosen["program_choice"] == PROGRAM and chosen["reason"] == "prefer the later attempt"
    assert (world.project / "out" / "persisted" / "toy-1-R1-c1-b1" / "persisted.json").is_file()
    assert report(world) == before  # the registry report stays write-once
    assert reads(world) == 1  # no second holdout read: the figure of the report is the program's pick
    section = ds_section(world)
    assert "chosen by the DS" in section and "belongs to the program's own pick" in section
    assert [m["outcome"] for m in applied(world)] == ["applied"]
    # the model of reproduce is now the DS pick, and a second finalize applies nothing again
    out = tmp_path / "repro.json"
    assert world.run("reproduce", "--out", str(out))[0] == 0
    assert json.loads(out.read_text(encoding="utf-8"))["bundle"] == "toy-1-R1-c1-b1"
    assert finalize(world, tmp_path)[1]["ok"] is True and len(applied(world)) == 1


# 33
def test_a_finalize_without_a_valid_pick_says_so_and_changes_nothing(tmp_path):
    world = stopped_world(tmp_path)
    code, result = finalize(world, tmp_path)
    assert result["ok"] is False and "no valid pick" in result["error"]
    assert not (world.project / "out" / "chosen").exists()
    decide(world, "choose_model", attempt="toy-1-R9-c9", reason="an attempt that does not exist")
    assert finalize(world, tmp_path)[1]["ok"] is False  # a pick that is not usable is ignored by the program
    assert not (world.project / "out" / "chosen").exists()
