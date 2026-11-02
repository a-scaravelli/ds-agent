"""resume after a quota error, a crash, a revision and a continue; changed configs are refused (cases 10 to 13, 35)."""

import json

import pytest
from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, World, drive_to_review

from kaggle_agent.llm import QuotaExhaustedError
from kaggle_agent.writer import ClaudeBundleWriter


def summary(world):
    """What the registry holds, in a form that two equal runs share."""
    registry = world.registry()
    try:
        attempts = registry.list_attempts("toy-1")
        return {
            "state": registry.get_cycle("toy-1").state,
            "attempts": [a.attempt_id for a in attempts],
            "outcomes": [a.outcome for a in attempts],
            "bundles": [b.id for b in registry.list_bundles("toy-1")],
            "rounds": [r.round.id for r in registry.list_rounds_with_candidates("toy-1")],
            "comparisons": registry.count_comparisons("toy-1"),
            "reports": len(registry.list_reviewer_reports("toy-1")),
            "executions": sum(len(registry.get_attempt(a.attempt_id).executions) for a in attempts),
            "holdout_reads": len(registry.list_holdout_reads("toy-1")),
        }
    finally:
        registry.close()


# 10
def test_resume_after_a_quota_error_in_the_middle_of_a_round(tmp_path):
    clean = World(tmp_path / "clean")
    drive_to_review(clean)
    # call 8 is the modeling call of round 1 (coordinator, then 4 calls of the baseline, then data, fe, modeling)
    world = World(tmp_path / "quota", errors={8: QuotaExhaustedError("usage limit reached")})
    assert world.run("start")[0] == 10
    world.lead(PLAN_APPROVE)
    code, text = world.run("resume")
    assert code == 11 and "SUSPENDED" in text and "usage limit reached" in text and "resume" in text
    assert world.claude.kinds[7] == "modeling"
    before = summary(world)
    assert before["state"] == "ITERATING" and len(before["attempts"]) == 1  # only the baseline is recorded
    code, text = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_GUIDANCE"
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume")[0] == 10
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    assert world.run("resume")[0] == 10 and world.state() == "AWAITING_REVIEW"
    assert summary(world) == summary(clean)  # the same records as a run with no error, none twice
    registry = world.registry()
    try:
        outcomes = [r.outcome for r in registry.list_llm_calls("toy-1")]
    finally:
        registry.close()
    assert outcomes.count("quota") == 1 and len(outcomes) == len(world.claude.call_log())


# 11
def test_resume_after_a_crash_repeats_no_attempt_and_no_holdout_read(tmp_path, monkeypatch):
    clean = World(tmp_path / "clean")
    drive_to_review(clean)
    world = World(tmp_path / "crash")
    original, count = ClaudeBundleWriter.write, {"n": 0}

    def flaky(self, request):
        count["n"] += 1
        if count["n"] == 2:
            raise RuntimeError("the disk exploded")
        return original(self, request)

    monkeypatch.setattr(ClaudeBundleWriter, "write", flaky)
    assert world.run("start")[0] == 10
    world.lead(PLAN_APPROVE)
    code, text = world.run("resume")
    assert code == 13 and "CRASHED" in text and "disk exploded" in text and "registry is intact" in text and "resume" in text
    assert world.state() == "ITERATING"
    assert world.run("resume")[0] == 10
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume")[0] == 10
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    assert world.run("resume")[0] == 10 and world.state() == "AWAITING_REVIEW"
    assert summary(world) == summary(clean)
    assert summary(world)["holdout_reads"] == 1


def test_a_keyboard_interrupt_is_not_swallowed(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.run("start")
    world.lead(PLAN_APPROVE)

    def interrupt(self, request):
        raise KeyboardInterrupt

    monkeypatch.setattr(ClaudeBundleWriter, "write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        world.run("resume")
    assert world.state() == "BASELINE"


# 12
def test_a_plan_revision_is_applied_once_and_survives_a_suspension(tmp_path):
    world = World(tmp_path)
    world.run("start")
    world.lead('[plan]\ndecision = "revise"\nplan = 1\ntext = "prefer simple models"\n')
    world.claude.fail_on["coordinator"] = QuotaExhaustedError("quota")
    code, text = world.run("resume")
    assert code == 11 and world.state() == "PLANNING"
    ledger = json.loads((world.project / "lead-ledger.json").read_text(encoding="utf-8"))
    assert [r["text"] for r in ledger["revisions"]] == ["prefer simple models"]
    code, text = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_PLAN_APPROVAL" and "plan 2" in text
    coordinator = [p for p, k in zip(world.claude.prompts, world.claude.kinds) if k == "coordinator"]
    assert len(coordinator) == 3  # the plan, the re-plan that hit the quota, the re-plan that worked
    assert "prefer simple models" not in coordinator[0]
    assert "prefer simple models" in coordinator[1] and "prefer simple models" in coordinator[2]
    ledger = json.loads((world.project / "lead-ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["revisions"]) == 1  # not applied twice
    code, _ = world.run("resume")  # the same file again: plan 2 still waits, no new coordinator call
    assert code == 10 and world.claude.kinds.count("coordinator") == 3
    world.lead('[plan]\ndecision = "revise"\nplan = 2\ntext = "add a scaler"\n')
    assert world.run("resume")[0] == 10 and world.claude.kinds.count("coordinator") == 4
    assert "add a scaler" in world.claude.prompts[-1] and "prefer simple models" in world.claude.prompts[-1]


def test_a_crash_between_the_ledger_and_the_replan_asks_the_coordinator_again(tmp_path):
    world = World(tmp_path)
    world.run("start")
    (world.project / "lead-ledger.json").write_text(
        json.dumps({"revisions": [{"plan": 1, "text": "prefer simple models", "at": "x"}]}), encoding="utf-8")
    # the crash happened after the ledger write: the file still holds the (now consumed) revision
    world.lead('[plan]\ndecision = "revise"\nplan = 1\ntext = "prefer simple models"\n')
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_PLAN_APPROVAL"
    assert world.claude.kinds.count("coordinator") == 2 and "prefer simple models" in world.claude.prompts[-1]
    world.lead(PLAN_APPROVE)
    assert world.run("resume")[0] == 10 and world.state() == "AWAITING_GUIDANCE"


# 13
def test_a_continue_answer_is_not_recorded_and_is_reused_after_a_quota_error(tmp_path):
    world = World(tmp_path)
    world.run("start")
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    world.claude.fail_on["coordinator"] = QuotaExhaustedError("quota")
    code, text = world.run("resume")
    assert code == 11 and world.state() == "AWAITING_GUIDANCE"
    assert not (world.project / "lead-ledger.json").exists()
    registry = world.registry()
    try:
        assert registry.list_guidance("toy-1") == []
    finally:
        registry.close()
    code, text = world.run("resume")  # round 1 is asked again and the same entry is reused
    assert code == 10 and world.state() == "AWAITING_GUIDANCE" and "guidance after round 2" in text
    assert "stale guidance for round 1 ignored" in text  # round 2 does not reuse it
    registry = world.registry()
    try:
        assert registry.list_guidance("toy-1") == []
        assert len(registry.list_rounds_with_candidates("toy-1")) == 3
    finally:
        registry.close()


# 35
@pytest.mark.parametrize("old, new, key", [
    ('target = "from_submission_spec"', 'target = "x1"', "competition.target"),
    ('metric = "from_competition_info"', 'metric = "rmse"', "competition.metric"),
    ("validation_params = { n_folds = 3 }", "validation_params = { n_folds = 3 }\nseed = 9", "protocol"),
    ('slug = "toy"', 'slug = "other"', "competition.slug"),
])
def test_a_changed_config_is_refused_after_start(tmp_path, old, new, key):
    world = World(tmp_path)
    world.run("start")
    original = world.config.read_text(encoding="utf-8")
    world.config.write_text(original.replace(old, new), encoding="utf-8")
    code, text = world.run("resume")
    assert code == 2 and key in text and "cannot change" in text
    world.config.write_text(original, encoding="utf-8")
    world.lead(PLAN_APPROVE)
    assert world.run("resume")[0] == 10


def test_changed_data_is_refused_but_model_names_and_timeouts_may_change(tmp_path):
    world = World(tmp_path)
    world.run("start")
    original = world.config.read_text(encoding="utf-8")
    world.config.write_text(original.replace("wait_seconds = 60", "wait_seconds = 90\nbundle_timeout = 30")
                            + '\n[models]\ncoordinator = "other-model"\n', encoding="utf-8")
    world.lead(PLAN_APPROVE)
    assert world.run("resume")[0] == 10
    kaggle_calls = len(world.kaggle.calls)
    with open(world.project / "data" / "train.csv", "a", encoding="utf-8") as handle:
        handle.write("999,1,1,1\n")
    code, text = world.run("resume")
    assert code == 2 and "hash mismatch" in text and "train.csv" in text
    assert len(world.kaggle.calls) == kaggle_calls  # a resume makes no Kaggle call
