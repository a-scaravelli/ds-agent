"""The holdout is evaluated once also across a crash and a resume (case 7)."""

import pytest
from pipeline_helpers import CYCLE, Boom, make_pipeline

from kaggle_agent.pipeline import INTERRUPTED
from kaggle_agent.pipeline.fake import FakeModelStore


def counts(pipe):
    registry = pipe.registry
    return (
        1 if registry.get_holdout_session(CYCLE) else 0,
        len(registry.list_holdout_reads(CYCLE)),
        1 if registry.get_holdout_outcome(CYCLE) else 0,
        1 if registry.get_final_report(CYCLE) else 0,
    )


def test_a_crash_after_the_session_opened_records_a_failed_outcome_and_never_reads_again(tmp_path):
    store = FakeModelStore(predict_failure=Boom("process killed"))
    pipe = make_pipeline(tmp_path, store=store, guidance=False)
    with pytest.raises(Boom):
        pipe.run()
    assert counts(pipe) == (1, 1, 0, 0)  # session and read, no outcome, no report
    assert pipe.registry.get_cycle(CYCLE).state == "REPORTING"
    store.predict_failure = None  # the process is restarted
    result = pipe.run()
    assert result.outcome.status == "stopped"
    outcome = pipe.registry.get_holdout_outcome(CYCLE)
    assert outcome.status == "failed" and outcome.reason == INTERRUPTED
    assert counts(pipe) == (1, 1, 1, 1)
    assert len(store.predicts) == 1 and len(store.fits) == 1  # no second read, no second fit
    assert result.report.holdout["reason"] == INTERRUPTED
    again = pipe.run()
    assert again.report == result.report and counts(pipe) == (1, 1, 1, 1)


def test_a_crash_after_the_outcome_before_the_report_reuses_the_outcome(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    registry = pipe.registry
    real = registry.add_final_report
    state = {"crash": True}

    def crashing(report):
        if state["crash"]:
            state["crash"] = False
            raise Boom("killed before the report")
        real(report)

    registry.add_final_report = crashing
    with pytest.raises(Boom):
        pipe.run()
    assert counts(pipe) == (1, 1, 1, 0)
    outcome = registry.get_holdout_outcome(CYCLE)
    suggestions = [s.id for s in registry.list_suggestions()]
    result = pipe.run()
    assert result.outcome.status == "stopped"
    assert registry.get_holdout_outcome(CYCLE) == outcome and outcome.status == "succeeded"
    assert counts(pipe) == (1, 1, 1, 1)
    assert [s.id for s in registry.list_suggestions()] == suggestions
    assert len(pipe.store.predicts) == 1 and len(pipe.store.fits) == 1
    assert (tmp_path / "out" / "reports" / f"{CYCLE}.md").is_file()


def test_a_crash_while_persisting_leaves_no_half_written_folder(tmp_path):
    store = FakeModelStore(fit_failure=Boom("killed while fitting"))
    pipe = make_pipeline(tmp_path, store=store, guidance=False)
    with pytest.raises(Boom):
        pipe.run()
    persisted = tmp_path / "out" / "persisted"
    assert not any(p.name for p in persisted.iterdir())
    assert counts(pipe) == (0, 0, 0, 0)
    store.fit_failure = None
    result = pipe.run()
    assert result.report.holdout["status"] == "succeeded" and counts(pipe) == (1, 1, 1, 1)
