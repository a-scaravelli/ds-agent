"""Suggestions derived from the report and the guidance, stored in the registry (cases 27 to 29)."""

import numpy as np
from evaluation_helpers import goal_dict
from loop_helpers import by_round, crash
from pipeline_helpers import CYCLE, FRAME, GUIDANCE_TEXT, make_pipeline

from kaggle_agent.pipeline import derive_suggestions, store_suggestions, suggestion_id
from kaggle_agent.pipeline.fake import FakeModelStore


def stored(registry):
    return {s.id: s for s in registry.list_suggestions() if s.origin_cycle == CYCLE}


def troubled(tmp_path, **kwargs):
    """Stagnation stop, two code_error failures, a holdout below validation, a guidance that did not help."""
    base = by_round([0.1, 0.1, 0.1])

    def rule(bundle_id, n):
        return crash() if bundle_id.startswith(("C1-R2-c1", "C1-R3-c1")) else base(bundle_id, n)

    return make_pipeline(tmp_path, rule=rule, fix_retries=0, **kwargs)


def test_one_suggestion_per_applicable_rule_with_texts_that_name_the_cause(tmp_path):
    pipe = troubled(tmp_path)
    result = pipe.run()
    assert result.report.stop_reason == "stagnation"
    items = stored(pipe.registry)
    assert set(items) == {
        f"{CYCLE}-R1-g",  # the guidance itself, stored by add_guidance
        suggestion_id(CYCLE, "stop"), suggestion_id(CYCLE, "failures"),
        suggestion_id(CYCLE, "holdout"), suggestion_id(CYCLE, "guidance-r1"),
    }
    assert all(s.origin_cycle == CYCLE for s in items.values())
    assert "stagnation" in items[f"{CYCLE}-sg-stop"].text
    assert "code_error" in items[f"{CYCLE}-sg-failures"].text and "2 attempts" in items[f"{CYCLE}-sg-failures"].text
    holdout = pipe.registry.get_holdout_outcome(CYCLE)
    gap = f"{0.90625 - holdout.score:.4f}"
    assert "worse than validation" in items[f"{CYCLE}-sg-holdout"].text and gap in items[f"{CYCLE}-sg-holdout"].text
    assert GUIDANCE_TEXT in items[f"{CYCLE}-sg-guidance-r1"].text
    assert items[f"{CYCLE}-R1-g"].text == GUIDANCE_TEXT  # not duplicated, not rewritten
    assert len([s for s in pipe.registry.list_suggestions() if s.text == GUIDANCE_TEXT]) == 1


def test_a_failed_holdout_gives_a_suggestion_with_its_reason(tmp_path):
    from kaggle_agent.bundle import BundleRunError

    store = FakeModelStore(predict_failure=BundleRunError("crashed", "code_error", "exit code 3"))
    pipe = make_pipeline(tmp_path, store=store, guidance=False)
    pipe.run()
    text = stored(pipe.registry)[f"{CYCLE}-sg-holdout"].text
    assert "holdout evaluation failed" in text and "exit code 3" in text


def test_suggestions_are_deterministic_and_idempotent(tmp_path):
    pipe = troubled(tmp_path)
    pipe.run()
    registry, view = pipe.registry, pipe.view()
    outcome = registry.get_holdout_outcome(CYCLE)
    first = derive_suggestions(registry, view, outcome, "stagnation")
    second = derive_suggestions(registry, view, outcome, "stagnation")
    assert first == second and [s.id for s in first][0] == f"{CYCLE}-sg-stop"
    before = registry.list_suggestions()
    assert store_suggestions(registry, first) == []  # all of them exist already
    assert registry.list_suggestions() == before


class LabelStore(FakeModelStore):
    """A store that predicts the true labels, so the holdout is as good as it can be."""

    def predict_with_model(self, model_dir, x, seed):
        super().predict_with_model(model_dir, x, seed)
        return FRAME.loc[x.index, "y"].to_numpy()


def test_a_cycle_that_met_its_target_with_a_good_holdout_only_gets_the_stop_suggestion(tmp_path):
    pipe = make_pipeline(
        tmp_path, store=LabelStore(), guidance=False, goal=goal_dict(("accuracy", 0.85, "maximize")))
    result = pipe.run()
    assert result.report.stop_reason == "target_reached"
    assert pipe.registry.get_holdout_outcome(CYCLE).score == 1.0
    items = stored(pipe.registry)
    assert list(items) == [suggestion_id(CYCLE, "stop")]
    assert "target was reached" in items[suggestion_id(CYCLE, "stop")].text


def test_the_report_and_the_suggestions_point_to_each_other(tmp_path):
    pipe = troubled(tmp_path)
    result = pipe.run()
    assert len(result.report.suggestions) == 5
    for item in result.report.suggestions:
        suggestion = pipe.registry.get_suggestion(item["id"])
        assert suggestion.text == item["text"] and suggestion.origin_cycle == CYCLE
    assert {s["id"] for s in result.report.suggestions} == set(stored(pipe.registry))
    assert np.isfinite(result.report.chosen["validation_mean"])
