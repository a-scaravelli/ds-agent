"""The final report: content, rendering, storage and validation (cases 1 to 3)."""

import json

import pytest
from pipeline_helpers import CYCLE, GUIDANCE_TEXT, make_pipeline

from kaggle_agent.pipeline import FinalReport, render_text, report_path, store_report
from kaggle_agent.proposals import ValidationError


@pytest.fixture
def finished(tmp_path):
    pipe = make_pipeline(tmp_path)  # 2 candidates per round, one guidance after round 1, stagnation stop
    result = pipe.run()
    return pipe, result, tmp_path


def test_the_report_holds_the_chosen_bundle_the_holdout_the_history_and_the_suggestions(finished):
    pipe, result, _ = finished
    registry = pipe.registry
    report = FinalReport.from_dict(registry.get_final_report(CYCLE).content)
    assert report == result.report and report.stop_reason == "stagnation"
    view = pipe.view()
    chosen = view.reference()
    assert report.chosen["attempt_id"] == chosen.attempt.id
    assert report.chosen["bundle_id"] == chosen.executions[-1].config["bundle_id"]
    metric = next(m for m in chosen.executions[-1].metrics if m.partition == "validation")
    assert report.chosen["validation_mean"] == metric.mean
    assert report.chosen["validation_dispersion"] == metric.dispersion
    assert report.chosen["hypothesis"] == chosen.attempt.hypothesis
    assert report.baseline_score == view.score(view.baseline())
    outcome = registry.get_holdout_outcome(CYCLE)
    assert (report.holdout["status"], report.holdout["score"], report.holdout["rows"]) == (
        "succeeded", outcome.score, outcome.rows)
    assert report.metric == "accuracy" and report.direction == "maximize"
    assert report.protocol_hash == registry.get_cycle(CYCLE).protocol_hash
    assert len(report.rounds) == 3
    for item, state in zip(report.rounds, view.round_states()):
        assert [c["id"] for c in item["candidates"]] == [d.attempt.id for d in state.details]
        assert len(item["candidates"]) == 2
        assert item["reviewer"] == registry.list_reviewer_reports(CYCLE, state.round_id)[-1].text
        assert item["tasks"] == list(state.round.round.tasks)
        assert item["comparison"] == {"difference": state.comparison.difference,
                                      "threshold": state.comparison.threshold,
                                      "is_real": state.comparison.is_real}
    assert [g["text"] for g in report.rounds[0]["guidance"]] == [GUIDANCE_TEXT]
    assert report.rounds[1]["guidance"] == [] and report.rounds[2]["guidance"] == []
    assert [w["round_id"] for w in report.worked] == [f"{CYCLE}-R1"]
    assert report.failures == {}
    stored = [(s.id, s.text) for s in registry.list_suggestions() if s.origin_cycle == CYCLE]
    assert [(s["id"], s["text"]) for s in report.suggestions] == stored
    assert report.persisted_path and report.reproduce_command.startswith(
        "python -m kaggle_agent.pipeline reproduce --persisted " + report.persisted_path)


def test_the_report_counts_failures_by_kind(tmp_path):
    from loop_helpers import by_round, crash

    def rule(bundle_id, n):
        return crash() if "-R2-c1" in bundle_id else by_round([0.1, 0.1, 0.1])(bundle_id, n)

    pipe = make_pipeline(tmp_path, rule=rule, fix_retries=0, guidance=False)
    report = pipe.run().report
    assert report.failures == {"code_error": 1}
    failed = [c for r in report.rounds for c in r["candidates"] if c["outcome"] == "failed"]
    assert [c["failure_kind"] for c in failed] == ["code_error"] and failed[0]["score"] is None


def test_the_text_names_the_bundle_the_metrics_the_stop_reason_and_the_command(finished):
    pipe, result, tmp_path = finished
    report = result.report
    text = render_text(report)
    for needle in (report.chosen["bundle_id"], f"{report.chosen['validation_mean']:.4f}",
                   f"{report.holdout['score']:.4f}", "stagnation", report.reproduce_command,
                   GUIDANCE_TEXT, "Round 3"):
        assert needle in text
    path = report_path(tmp_path / "out", CYCLE)
    assert path.read_text(encoding="utf-8") == text


def test_storing_twice_changes_neither_the_registry_nor_the_file(finished):
    pipe, result, tmp_path = finished
    path = report_path(tmp_path / "out", CYCLE)
    record = pipe.registry.get_final_report(CYCLE)
    text, mtime = path.read_text(encoding="utf-8"), path.stat().st_mtime_ns
    other = FinalReport.from_dict({**result.report.to_dict(), "stop_reason": "something else"})
    again = store_report(pipe.registry, other, tmp_path / "out")
    assert again == result.report  # the stored one wins
    assert pipe.registry.get_final_report(CYCLE) == record
    assert path.read_text(encoding="utf-8") == text and path.stat().st_mtime_ns == mtime
    path.unlink()  # a missing file is recreated from the stored report
    store_report(pipe.registry, other, tmp_path / "out")
    assert path.read_text(encoding="utf-8") == text


def test_invalid_reports_are_rejected_and_a_valid_one_round_trips(finished):
    _, result, _ = finished
    report = result.report
    assert FinalReport.from_json(report.to_json()) == report
    assert json.loads(report.to_json())["cycle_id"] == CYCLE
    with pytest.raises(ValidationError):
        FinalReport.from_dict({**report.to_dict(), "surprise": 1})
    for bad in ("{not json", "[1, 2]", ""):
        with pytest.raises(ValidationError):
            FinalReport.from_json(bad)
    for change in ({"cycle_id": ""}, {"holdout": None}, {"rounds": "x"}, {"chosen": 3},
                   {"failures": {"a": float("nan")}}, {"persisted_path": ""}):
        with pytest.raises(ValidationError):
            FinalReport.from_dict({**report.to_dict(), **change})
    missing = report.to_dict()
    del missing["metric"]
    with pytest.raises(ValidationError):
        FinalReport.from_dict(missing)
