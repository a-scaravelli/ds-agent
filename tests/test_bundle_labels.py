"""Labels: the child gets training labels only, never validation or holdout labels (cases 9 and 10)."""

import pytest
from bundle_helpers import RecordingSandbox, make_runner, model, snooper_bundle, write_bundle
from evaluation_helpers import chrono_protocol, frozen_engine, regression_frame

from kaggle_agent.bundle import score_bundle
from kaggle_agent.evaluation import FrozenSplit

REG = regression_frame()


def reg_engine():
    return frozen_engine(chrono_protocol(metric="rmse", direction="minimize"))[0]


def needles(values):
    return sorted({repr(float(v)) for v in values})


def assert_clean(recording, result_count_needles, label_values):
    """Nothing the parent staged or the child printed holds a label; x.csv has no target column."""
    for files in recording.staged:
        if "x.csv" in files:  # the predict stage
            assert files["x.csv"].decode().splitlines()[0].split(",") == ["x1", "t"]
        else:  # the fit stage stages the training rows only
            assert "train.csv" in files
        blob = b"\n".join(files.values()).decode("utf-8", "replace")
        assert not any(n in blob for n in result_count_needles)
    for result in recording.results:
        out = result.stdout + result.stderr
        assert not any(n in out for n in result_count_needles)
    assert any("snoop finished" in r.stdout + r.stderr for r in recording.results)
    assert label_values


def test_validation_labels_never_reach_the_child(tmp_path):
    engine = reg_engine()
    split = FrozenSplit.create(REG, engine.protocol)
    part = split.partitions()[-1]
    x_train, y_train = split.data(part.train_rows)
    x_valid, y_valid = split.data(part.valid_rows)
    found = needles(y_valid)
    recording = RecordingSandbox()
    _, runner = make_runner(
        snooper_bundle(tmp_path, found), sandbox=recording, input_kind="values", target="y"
    )
    predictions = runner.fit_predict(x_train, y_train, x_valid, 1)
    assert predictions.tolist() == [0.0] * len(x_valid)  # zero hits in fit and in predict
    assert_clean(recording, found, found)
    assert "y" in recording.staged[0]["train.csv"].decode().splitlines()[0].split(",")


def test_holdout_labels_never_reach_the_child(tmp_path):
    engine = reg_engine()
    split = FrozenSplit.create(REG, engine.protocol)
    found = needles(REG["y"].iloc[split._holdout_rows])
    recording = RecordingSandbox()
    _, runner = make_runner(
        snooper_bundle(tmp_path, found), sandbox=recording, input_kind="values", target="y"
    )
    with engine.holdout_session(REG) as session:
        result = session.evaluate_holdout(runner, "A1")
    assert result.rows == len(split._holdout_rows)
    assert result.score == pytest.approx((REG["y"].iloc[split._holdout_rows] ** 2).mean() ** 0.5)
    assert_clean(recording, found, found)
    # the validation scoring path through the engine is clean too: holdout labels never staged
    outcome = score_bundle(engine, runner, REG, attempt_id="A1", execution_id="E1", config={}, data_hash="d")
    assert outcome.execution.outcome == "succeeded"
    assert_clean(recording, found, found)


CHECKER = model(
    """
    if "y" in X.columns:
        raise RuntimeError("predict received the target column")
    return [0] * len(X)
    """,
    """
    if "y" not in train.columns or train["y"].isna().any():
        raise RuntimeError("fit did not receive the training labels")
    """,
)


def test_fit_receives_labels_and_predict_does_not(tmp_path):
    engine = reg_engine()
    folder = write_bundle(tmp_path, CHECKER)
    _, runner = make_runner(folder, input_kind="values")
    outcome = score_bundle(engine, runner, REG, attempt_id="A1", execution_id="E1", config={}, data_hash="d")
    assert outcome.error is None and outcome.execution.outcome == "succeeded"


def test_a_bundle_that_asserts_the_opposite_crashes_with_code_error(tmp_path):
    opposite = model(
        "assert 'y' in X.columns, 'no target in X'\nreturn [0] * len(X)",
        "assert 'y' not in train.columns",
    )
    engine = reg_engine()
    _, runner = make_runner(write_bundle(tmp_path, opposite), input_kind="values")
    outcome = score_bundle(engine, runner, REG, attempt_id="A1", execution_id="E1", config={}, data_hash="d")
    record = outcome.execution
    assert record.outcome == "failed" and record.failure_kind == "code_error"
    assert record.resources["sandbox_status"] == "crashed"


def test_the_snooper_finds_labels_that_are_staged(tmp_path):
    """Control: the snooper detects the training labels, which are staged on purpose."""
    split = FrozenSplit.create(REG, reg_engine().protocol)
    part = split.partitions()[-1]
    x_train, y_train = split.data(part.train_rows)
    x_valid, _ = split.data(part.valid_rows)
    _, runner = make_runner(snooper_bundle(tmp_path, needles(y_train)), input_kind="values")
    assert all(p > 0 for p in runner.fit_predict(x_train, y_train, x_valid, 1))
