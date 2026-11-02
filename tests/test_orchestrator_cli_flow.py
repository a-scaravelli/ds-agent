"""The command line, end to end with fakes (cases 6, 7, 8, 9, 20, 34)."""

import json

from orchestrator_helpers import (
    GUIDE_CONTINUE,
    PLAN_APPROVE,
    REVIEW_ACCEPT,
    World,
    drive_to_review,
    drive_to_stopped,
)

from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.llm import QuotaExhaustedError
from kaggle_agent.pipeline import __main__ as pipeline_cli


# 6
def test_start_creates_the_project_and_runs_until_the_lead_is_needed(tmp_path):
    world = World(tmp_path)
    code, text = world.run("start")
    assert code == 10
    assert "WAITING FOR THE LEAD" in text and "Plan proposed by the coordinator" in text
    assert "baseline: majority class" in text
    assert "[plan]" in text and 'decision = "approve"' in text and "plan = 1" in text
    assert (world.project / "lead.example.toml").is_file() and not world.lead_file.exists()
    assert (world.project / "project.json").is_file() and world.state() == "AWAITING_PLAN_APPROVAL"
    registry = world.registry()
    try:
        assert len(registry.list_cycles()) == 1 and registry.get_cycle("toy-1").protocol_hash is None
    finally:
        registry.close()


# 7
def test_start_twice_does_not_create_a_second_cycle_nor_call_anything(tmp_path):
    world = World(tmp_path)
    world.run("start")
    kaggle_calls, prompts, factories = list(world.kaggle.calls), list(world.claude.prompts), world.llm_factory_calls
    code, text = world.run("start")
    assert code == 0 and "already started" in text and "resume" in text and "status" in text
    assert world.kaggle.calls == kaggle_calls and world.claude.prompts == prompts
    assert world.llm_factory_calls == factories
    registry = world.registry()
    try:
        assert len(registry.list_cycles()) == 1
    finally:
        registry.close()


# 8
def test_a_full_cycle_runs_through_the_command_line_up_to_the_review(tmp_path):
    world = World(tmp_path)
    assert world.run("start")[0] == 10
    world.lead(PLAN_APPROVE)
    code, text = world.run("resume")
    assert code == 10 and "guidance after round 1" in text
    world.lead(PLAN_APPROVE + '[guidance]\nround = 1\naction = "guidance"\ntext = "try scaling"\n')
    code, text = world.run("resume")
    assert code == 10 and "guidance after round 2" in text
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    code, text = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_REVIEW" and "[review]" in text
    registry = world.registry()
    try:
        cycle = registry.get_cycle("toy-1")
        assert cycle.protocol_hash is not None
        assert [g.text for g in registry.list_guidance("toy-1")] == ["try scaling"]
        assert len(registry.list_rounds_with_candidates("toy-1")) == 3  # setup round and N = 2
        assert registry.get_final_report("toy-1") is not None
        coordinator_texts = [t.input["guidance"] for t in registry.list_role_traces("toy-1", role="coordinator")]
        assert ["try scaling"] in coordinator_texts  # the guidance reached the coordinator
    finally:
        registry.close()


# 9
def test_accept_finishes_the_cycle_and_the_report_and_bundle_exist(tmp_path):
    world = World(tmp_path)
    drive_to_review(world)
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2) + REVIEW_ACCEPT)
    code, text = world.run("resume")
    assert code == 0 and "FINISHED" in text and "submit" in text
    assert world.state() == "STOPPED"
    registry = world.registry()
    try:
        report = registry.get_final_report("toy-1").content
        assert report["holdout"]["status"] == "succeeded" and report["persisted_path"]
        assert registry.get_holdout_outcome("toy-1").status == "succeeded"
    finally:
        registry.close()
    assert (world.project / "out/reports/toy-1.md").is_file()
    assert (world.project / "out/persisted" / report["chosen"]["bundle_id"] / "persisted.json").is_file()
    code, again = world.run("resume")  # a finished cycle stays finished
    assert code == 0 and "FINISHED" in again


# 20
def test_status_tells_the_lead_the_next_key_to_fill_in_every_state(tmp_path):
    world = World(tmp_path)
    code, text = world.run("status")
    assert code == 2 and "no project" in text and "start" in text
    world.run("start")
    code, text = world.run("status")
    assert code == 0 and "state: AWAITING_PLAN_APPROVAL" in text and "round 0 of 2" in text
    assert "[plan]" in text and "Plan proposed" in text and str(world.lead_file) in text and "not created yet" in text
    assert "NOTICES.txt" in text
    calls = len(world.kaggle.calls), len(world.claude.prompts)
    world.lead(PLAN_APPROVE)
    world.run("resume")
    code, text = world.run("status")
    assert "AWAITING_GUIDANCE" in text and "round 1 of 2" in text and "round = 1" in text and "best score so far: 0." in text
    world.claude.fail_on["coordinator"] = QuotaExhaustedError("quota")
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    code, run = world.run("resume")
    assert code == 11 and "SUSPENDED" in run
    code, text = world.run("status")
    assert "last run: SUSPENDED" in text and "resume" in text
    world.run("resume")
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    world.run("resume")
    code, text = world.run("status")
    assert "AWAITING_REVIEW" in text and "[review]" in text and "report:" in text
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2) + REVIEW_ACCEPT)
    world.run("resume")
    code, text = world.run("status")
    assert "STOPPED" in text and "submit" in text and "finished" in text
    assert len(world.kaggle.calls) == calls[0]  # status itself never calls Kaggle
    assert world.run("status", deps=None)[0] == 0  # status needs no dependencies at all


# 34
def test_the_result_is_reproducible_with_a_real_bundle_runner(tmp_path, capsys):
    world = World(tmp_path, real_sandbox=True)
    drive_to_stopped(world)
    registry = world.registry()
    try:
        report = registry.get_final_report("toy-1").content
        stored = json.loads((world.project / "project.json").read_text(encoding="utf-8"))
        state_protocol = Protocol.from_dict(stored["protocol"])
        EvaluationEngine(registry, "toy-1", state_protocol).verify()
    finally:
        registry.close()
    capsys.readouterr()
    code = pipeline_cli.main(["reproduce", "--persisted", report["persisted_path"],
                              "--db", str(world.project / "registry.db"),
                              "--data", str(world.project / "data/train.csv")])
    out = capsys.readouterr().out
    assert code == 0 and "OK" in out and "reproduced validation accuracy" in out
