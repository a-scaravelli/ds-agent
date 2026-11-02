"""The end-to-end flow with the existing fakes: start, rounds, holdout, report, suggestions (cases 21, 25, 26)."""

import dataclasses
import subprocess
from pathlib import Path

from loop_helpers import record_states
from pipeline_helpers import CYCLE, GUIDANCE_TEXT, PLAN, make_pipeline

from kaggle_agent.budget import CycleLimits
from kaggle_agent.loop import STATES, TRANSITIONS, Hooks
from kaggle_agent.pipeline import FinalReport, run_pipeline, start_cycle
from kaggle_agent.pipeline.fake import FakeModelStore
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.review import SimulatedLead

ROOT = Path(__file__).resolve().parent.parent


def test_start_cycle_creates_a_planning_cycle_with_the_limits():
    registry = SqliteRegistry()
    limits = CycleLimits(rounds=4, stagnation_rounds=3, fix_retries=2)
    cycle = start_cycle(registry, "N1", PLAN, limits, project_id="p1")
    stored = registry.get_cycle("N1")
    assert stored == cycle
    assert (stored.state, stored.iterations, stored.stagnation_k, stored.max_fix_retries, stored.project_id) == (
        "PLANNING", 4, 3, 2, "p1")
    assert CycleLimits.resolve(stored, limits).rounds == 4


def test_the_whole_flow_runs_from_start_cycle_to_stopped_with_the_fakes(tmp_path):
    registry = SqliteRegistry()
    start_cycle(registry, CYCLE, PLAN, CycleLimits(rounds=5, candidates=2, stagnation_rounds=2))
    pipe = make_pipeline(tmp_path, registry=registry, create_cycle=False, guidance=False)
    states = record_states(registry)
    result = pipe.run()
    assert result.outcome.status == "stopped" and result.outcome.state == "STOPPED"
    assert states[0] == "AWAITING_PLAN_APPROVAL" and states[-3:] == ["REPORTING", "AWAITING_REVIEW", "STOPPED"]
    assert set(states) <= set(STATES)
    previous = "PLANNING"
    for state in states:  # every saved state is an allowed transition from the one before
        assert state in TRANSITIONS[previous]
        previous = state
    assert "ITERATING" in states and states.count("REPORTING") == 1
    outcome = registry.get_holdout_outcome(CYCLE)
    assert outcome.status == "succeeded" and outcome.rows > 0
    assert result.report == FinalReport.from_dict(registry.get_final_report(CYCLE).content)
    assert result.report.stop_reason == result.outcome.reason == "stagnation"
    sg = [s.id for s in registry.list_suggestions() if s.origin_cycle == CYCLE]
    assert f"{CYCLE}-sg-stop" in sg
    assert len(registry.list_holdout_reads(CYCLE)) == 1
    assert (tmp_path / "out" / "reports" / f"{CYCLE}.md").is_file()
    assert len(pipe.store.fits) == 1 and len(pipe.store.predicts) == 1


def test_a_silent_lead_waits_with_the_report_stored_and_a_later_run_adds_nothing(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    silent_at_review = SimulatedLead(reviews=[None])  # no answer the first time, then the default
    waiting = pipe.run(lead=silent_at_review)
    assert waiting.outcome.status == "waiting" and waiting.outcome.state == "AWAITING_REVIEW"
    assert waiting.report is not None
    registry = pipe.registry

    def snapshot():
        attempt = pipe.view().reference()
        artifacts = registry.get_attempt(attempt.attempt.id).executions[-1].artifacts
        return (registry.get_final_report(CYCLE), registry.get_holdout_outcome(CYCLE),
                len(registry.list_holdout_reads(CYCLE)), len(registry.list_suggestions()), artifacts)

    before = snapshot()
    done = pipe.run()  # the default lead accepts
    assert done.outcome.status == "stopped"
    assert snapshot() == before
    assert len(pipe.store.fits) == 1 and len(pipe.store.predicts) == 1
    assert done.report == waiting.report


def test_a_caller_supplied_report_hook_is_respected(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    seen = []
    hooks = Hooks(report=lambda ctx, reason: seen.append(reason))
    result = pipe.run(hooks=hooks)
    assert result.outcome.status == "stopped" and seen == ["stagnation"]
    assert result.report is None and pipe.registry.get_final_report(CYCLE) is None
    assert pipe.registry.get_holdout_outcome(CYCLE) is None and pipe.store.fits == []
    assert not (tmp_path / "out").exists()


def test_the_pipeline_hook_is_installed_without_changing_the_context(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    assert pipe.ctx.hooks.report is None
    run_pipeline(pipe.ctx, pipe.pctx)
    assert pipe.ctx.hooks.report is None  # a copy was used (dataclasses.replace)
    assert dataclasses.is_dataclass(pipe.ctx)


def test_loop_and_web_sources_have_no_diff_against_the_merge_base():
    git = ["git", "-C", str(ROOT)]
    base = subprocess.run([*git, "merge-base", "HEAD", "origin/main"], capture_output=True, text=True)
    if base.returncode != 0:  # no remote in this checkout: nothing to compare with
        return  # pragma: no cover
    diff = subprocess.run(
        [*git, "diff", "--stat", base.stdout.strip(), "--", "src/kaggle_agent/loop", "src/kaggle_agent/web",
         "src/kaggle_agent/evaluation", ":(exclude)src/kaggle_agent/web/views.py",
         ":(exclude)src/kaggle_agent/web/templates/cycle.html"],  # step 12 adds the submissions table
        capture_output=True, text=True,
    )
    assert diff.returncode == 0 and diff.stdout.strip() == ""
