"""The persisted bundle is linked to the chosen attempt and seen by the web; origin links (cases 22, 24)."""

import html

import pytest
from pipeline_helpers import CYCLE, PLAN, make_pipeline
from registry_helpers import make_cycle

from kaggle_agent.budget import CycleLimits
from kaggle_agent.pipeline import link_pipeline, make_report_hook, start_cycle
from kaggle_agent.proposals import Suggestion
from kaggle_agent.registry import NotFoundError, SqliteRegistry
from kaggle_agent.web import create_app
from kaggle_agent.web.views import PIPELINE_PREFIX, pipeline_reference


def artifacts_of(registry, attempt_id):
    return [a for e in registry.get_attempt(attempt_id).executions for a in e.artifacts]


def test_the_persisted_bundle_is_linked_to_the_chosen_attempt_and_shown_by_the_web(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    registry = pipe.registry
    client = create_app(registry).test_client()
    hook = make_report_hook(pipe.pctx)
    before = []

    def spying_hook(ctx, reason):  # the page of the chosen attempt just before the report step
        chosen = pipe.view().reference().attempt.id
        before.append(client.get(f"/attempts/{chosen}").get_data(as_text=True))
        hook(ctx, reason)

    from kaggle_agent.loop import Hooks

    result = pipe.run(hooks=Hooks(report=spying_hook))
    assert result.outcome.status == "stopped" and len(before) == 1
    assert "no pipeline yet" in before[0]
    chosen = result.report.chosen["attempt_id"]
    persisted_dir = result.report.persisted_path
    assert persisted_dir == str((tmp_path / "out").resolve() / "persisted" / result.report.chosen["bundle_id"])
    artifacts = artifacts_of(registry, chosen)
    assert artifacts == [f"{PIPELINE_PREFIX}{persisted_dir}"]
    assert registry.get_attempt(chosen).executions[-1].artifacts == tuple(artifacts)
    assert pipeline_reference(artifacts) == {"ref": persisted_dir, "url": None}
    page = client.get(f"/attempts/{chosen}").get_data(as_text=True)
    assert "no pipeline yet" not in page and html.escape(persisted_dir) in page
    for summary in registry.list_attempts(CYCLE):  # other attempts have no such artifact
        if summary.attempt_id != chosen:
            assert not [a for a in artifacts_of(registry, summary.attempt_id) if a.startswith(PIPELINE_PREFIX)]


def test_linking_twice_and_resuming_do_not_duplicate_the_artifact(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    result = pipe.run()
    chosen = result.report.chosen["attempt_id"]
    link_pipeline(pipe.registry, chosen, result.report.persisted_path)
    hook = make_report_hook(pipe.pctx)
    hook(pipe.ctx, "stagnation")  # the whole report step again, as after a crash
    assert len(artifacts_of(pipe.registry, chosen)) == 1
    assert len(pipe.store.fits) == 1  # the valid persisted copy was reused


def test_the_link_is_written_even_when_the_holdout_later_fails(tmp_path):
    from kaggle_agent.bundle import BundleRunError
    from kaggle_agent.pipeline.fake import FakeModelStore

    store = FakeModelStore(predict_failure=BundleRunError("crashed", "code_error", "exit code 1"))
    pipe = make_pipeline(tmp_path, store=store, guidance=False)
    result = pipe.run()
    assert result.report.holdout["status"] == "failed"
    assert artifacts_of(pipe.registry, result.report.chosen["attempt_id"]) == [
        f"{PIPELINE_PREFIX}{result.report.persisted_path}"]


def test_a_cycle_can_start_from_a_suggestion_and_only_the_origin_is_linked(tmp_path):
    pipe = make_pipeline(tmp_path)
    pipe.run()
    registry = pipe.registry
    suggestion = next(s for s in registry.list_suggestions() if s.id == f"{CYCLE}-sg-stop")
    new = start_cycle(registry, "NEXT", PLAN, CycleLimits(rounds=2), from_suggestion=suggestion.id)
    assert registry.get_cycle("NEXT").state == "PLANNING" and new.id == "NEXT"
    assert [c.id for c in registry.cycles_of_suggestion(suggestion.id)] == ["NEXT"]
    assert registry.get_suggestion(suggestion.id).spawned_cycles == ("NEXT",)
    # the origin only: no attempt, bundle, round or report of the old cycle is in the new one
    assert registry.list_attempts("NEXT") == [] and registry.list_bundles("NEXT") == []
    assert registry.list_rounds_with_candidates("NEXT") == [] and registry.get_final_report("NEXT") is None
    with pytest.raises(NotFoundError):
        start_cycle(registry, "NEXT2", PLAN, CycleLimits(), from_suggestion="nope")
    assert "NEXT2" not in [c.id for c in registry.list_cycles()]
    assert "not carried" in start_cycle.__doc__ or "NOT carried" in start_cycle.__doc__


def test_start_cycle_without_a_suggestion_links_nothing():
    registry = SqliteRegistry()
    registry.add_cycle(make_cycle("OLD"))
    registry.add_suggestion(Suggestion("S", "text", "OLD"))
    start_cycle(registry, "N", PLAN, CycleLimits())
    assert registry.cycles_of_suggestion("S") == []
