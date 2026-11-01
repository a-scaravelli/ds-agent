"""The chosen bundle is evaluated once on the holdout; failures are recorded, never raised (cases 4 to 8)."""

from datetime import datetime, timezone

import pytest
from bundle_helpers import RecordingSandbox, make_runner, snooper_bundle
from evaluation_helpers import chrono_protocol, frozen_engine, make_protocol, regression_frame
from loop_helpers import crash
from pipeline_helpers import CYCLE, FRAME, make_pipeline

from kaggle_agent.bundle import BundleRunner, load_bundle
from kaggle_agent.evaluation import FrozenSplit, HoldoutSessionError
from kaggle_agent.loop import Hooks
from kaggle_agent.pipeline import evaluate_chosen_holdout, make_report_hook, persist_bundle
from kaggle_agent.pipeline.fake import FakeModelStore
from kaggle_agent.bundle import BundleRunError

REG = regression_frame()


def now():
    return datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def test_the_chosen_bundle_is_evaluated_once_through_the_frozen_protocol(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    result = pipe.run()
    registry = pipe.registry
    chosen = result.report.chosen["attempt_id"]
    outcome = registry.get_holdout_outcome(CYCLE)
    assert outcome.status == "succeeded" and outcome.attempt_id == chosen
    session = registry.get_holdout_session(CYCLE)
    reads = registry.list_holdout_reads(CYCLE)
    assert session is not None and not session.is_open
    assert [(r.attempt_id, r.rows) for r in reads] == [(chosen, outcome.rows)]
    # the score is the engine's: accuracy of the stored (majority) model on the holdout rows
    split = FrozenSplit.create(FRAME, pipe.engine.protocol)
    _, y_dev = split.development_data()
    _, y_holdout = split._holdout_data()
    assert outcome.rows == len(y_holdout)
    assert outcome.score == pytest.approx(float((y_holdout == y_dev.mode().iloc[0]).mean()))
    # no second read: neither another call nor another session
    again = evaluate_chosen_holdout(
        registry, pipe.engine, FRAME, chosen_id=chosen, persisted=None, store=None, now=now)
    assert again == outcome and len(registry.list_holdout_reads(CYCLE)) == 1
    with pytest.raises(HoldoutSessionError):
        with pipe.engine.holdout_session(FRAME):
            pass
    assert len(pipe.store.predicts) == 1
    assert pipe.store.predicts[0]["columns"] == ["x1", "x2", "group", "t"]  # features only


@pytest.mark.parametrize("script, kind, fragment", [
    (BundleRunError("crashed", "code_error", "exit code 1: Traceback (most recent call last): boom"),
     "code_error", "crashed"),
    ([0, 1, 0], "code_error", "malformed predictions"),
])
def test_a_failed_holdout_run_is_a_recorded_outcome_and_does_not_raise(tmp_path, script, kind, fragment):
    store = FakeModelStore(predict_failure=script)
    pipe = make_pipeline(tmp_path, store=store, guidance=False)
    result = pipe.run()  # nothing raised
    assert result.outcome.status == "stopped"
    outcome = pipe.registry.get_holdout_outcome(CYCLE)
    assert outcome.status == "failed" and outcome.failure_kind == kind
    assert fragment in outcome.reason and "\n" not in outcome.reason and "Traceback" not in outcome.reason
    assert outcome.score is None
    assert len(pipe.registry.list_holdout_reads(CYCLE)) == 1  # the read happened and is logged
    assert result.report.holdout["status"] == "failed" and result.report.holdout["reason"] == outcome.reason
    assert any(s.id == f"{CYCLE}-sg-holdout" for s in pipe.registry.list_suggestions())


def test_a_malformed_payload_raises_the_genuine_runner_error():
    store = FakeModelStore(predict_failure=[1, 2])
    import pandas as pd

    with pytest.raises(BundleRunError) as error:
        store.predict_with_model("unused", pd.DataFrame({"a": [1, 2, 3]}), 1)
    assert error.value.status == "malformed_predictions" and "expected 3" in error.value.reason


def test_a_protocol_hash_mismatch_is_a_failed_outcome_with_no_read(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    inner = make_report_hook(pipe.pctx)

    def tampering(ctx, reason):
        ctx.engine.protocol = make_protocol(threshold_k=9.0)  # no longer what was frozen
        inner(ctx, reason)

    result = pipe.run(hooks=Hooks(report=tampering))
    assert result.outcome.status == "stopped" and result.outcome.state == "STOPPED"
    outcome = pipe.registry.get_holdout_outcome(CYCLE)
    assert outcome.status == "failed" and outcome.failure_kind == "protocol"
    assert "protocol hash changed" in outcome.reason
    assert pipe.registry.list_holdout_reads(CYCLE) == []
    assert pipe.registry.get_holdout_session(CYCLE) is None
    assert pipe.registry.get_final_report(CYCLE) is not None


def test_no_usable_attempt_skips_the_holdout_and_still_reports(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False, rule=lambda bundle_id, n: crash(), fix_retries=0)
    result = pipe.run()
    assert result.outcome.status == "stopped"
    outcome = pipe.registry.get_holdout_outcome(CYCLE)
    assert (outcome.status, outcome.reason) == ("skipped", "no usable attempt")
    assert pipe.registry.get_holdout_session(CYCLE) is None
    assert pipe.registry.list_holdout_reads(CYCLE) == []
    assert result.report is not None and result.report.chosen is None
    assert result.report.persisted_path is None
    assert pipe.store.fits == [] and not (tmp_path / "out" / "persisted").exists()
    for summary in pipe.registry.list_attempts(CYCLE):
        executions = pipe.registry.get_attempt(summary.attempt_id).executions
        assert all(not a.startswith("pipeline:") for e in executions for a in e.artifacts)


def test_the_holdout_labels_never_reach_the_bundle_code(tmp_path):
    engine, registry = frozen_engine(chrono_protocol(metric="rmse", direction="minimize"))
    split = FrozenSplit.create(REG, engine.protocol)
    holdout_values = sorted({repr(float(v)) for v in REG["y"].iloc[split._holdout_rows]})
    recording = RecordingSandbox()
    loaded, runner = make_runner(
        snooper_bundle(tmp_path / "src", holdout_values), sandbox=recording, input_kind="values")
    persisted = persist_bundle(
        tmp_path / "out", loaded, runner, attempt_id="A1", cycle_id="C1", engine=engine, frame=REG,
        data_hash="d", validation_mean=1.0, validation_dispersion=0.1)
    outcome = evaluate_chosen_holdout(
        registry, engine, REG, chosen_id="A1", persisted=persisted, store=runner, now=now)
    assert outcome.status == "succeeded"
    # zero hits in fit and predict: the prediction is 0 for every row, so the score is the rmse of 0
    assert outcome.score == pytest.approx((REG["y"].iloc[split._holdout_rows] ** 2).mean() ** 0.5)
    assert [c[1].args[1] for c in recording.calls] == ["fit", "predict"]
    for files in recording.staged:
        blob = b"\n".join(files.values()).decode("utf-8", "replace")
        assert not any(n in blob for n in holdout_values)
        assert not any(name in files for name in ("y.csv", "labels.csv", "holdout.csv"))
    predict_files = recording.staged[1]
    assert predict_files["x.csv"].decode().splitlines()[0].split(",") == ["x1", "t"]  # no target column
    assert "train.csv" not in predict_files
    assert isinstance(runner, BundleRunner) and load_bundle(persisted.code_dir).folder_hash == loaded.folder_hash
