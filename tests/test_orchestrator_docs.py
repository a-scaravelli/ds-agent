"""README, design note, seed plan and the test map of the orchestrator ticket (cases 39 to 42)."""

import json
from pathlib import Path

from orchestrator_helpers import World, drive_to_review

from kaggle_agent.orchestrator import codes
from kaggle_agent.orchestrator.config import baseline_text

ROOT = Path(__file__).resolve().parent.parent
PLAN_CASES = 42
README = (ROOT / "README.md").read_text(encoding="utf-8")


def section() -> str:
    start = README.index("## Running a competition")
    return README[start:README.index("\n## ", start + 1)]


# 39
def test_the_readme_documents_the_manual_real_run_and_its_cost():
    text = section()
    assert "KAGGLE_AGENT_RUN_REAL=1" in text and "tests/test_orchestrator_manual_real.py" in text
    assert "Expected cost" in text and "tokens" in text and "never run by pytest" in text


# 40
def test_the_baseline_comes_from_the_seed_plan_built_from_the_config(tmp_path):
    world = World(tmp_path)
    drive_to_review(world)
    registry = world.registry()
    try:
        expected = "Baseline: " + baseline_text("accuracy", "y")
        assert registry.get_cycle("toy-1").plan.baseline == baseline_text("accuracy", "y")
        assert [p for p in world.claude.prompts if expected in p]  # the baseline task reaches the roles
        assert "Baseline: majority class" not in "".join(world.claude.prompts)  # the coordinator's plan item
    finally:
        registry.close()
    text = section()
    assert "simplest sensible model for <metric> on <target>" in text
    assert "competition.metric" in text and "competition.target" in text
    assert "not from the coordinator's plan" in text


# 41
def test_the_known_limits_state_that_the_submission_uses_the_development_only_model():
    limits = section().split("**Known limits.**")[1]
    assert "fitted on the development data only" in limits
    assert "not retrained on development plus holdout" in limits
    assert "validated and scored on the holdout" in limits


# 42
def test_the_documentation_the_design_note_and_the_test_map_are_complete():
    text = section()
    for part in ("**Install.**", "**Credentials**", "**Configuration**", "**The lead file.**", "**Step by step.**",
                 "**Resume.**", "**Exit codes.**", "**The pending marker", "**Where things live**",
                 "**Known limits.**", "--confirm-submit", "--assume-not-sent", "dry run",
                 "[plan]", "[guidance]", "[review]", "decision", "round", "action"):
        assert part in text, part
    for code in codes.ALL:
        assert str(code) in text, code
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    line = next(l for l in design.splitlines() if l.startswith("12. **Orchestrator"))
    assert "Part 2 done (level 1)" in line and "`orchestrator`" in design.split("## 14.")[1]
    assert (ROOT / ".asl" / "process-memory.md").is_file()
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(int(k) for k in cases) == list(range(1, PLAN_CASES + 1))
    for key, entries in cases.items():
        assert entries, key
        for entry in entries:
            source = (ROOT / entry["file"]).read_text(encoding="utf-8")
            assert f"def {entry['name']}(" in source, (key, entry)
