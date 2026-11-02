"""Model persistence and matrix predictions of the bundle runner (cases 9 to 12, 16 to 19)."""

import json

import numpy as np
import pandas as pd
import pytest
from bundle_helpers import (
    BUNDLE_DIR, FRAME, RecordingSandbox, constant_bundle, make_runner, mean_bundle, model,
    priors_bundle, write_bundle,
)
from evaluation_helpers import frozen_engine, make_protocol
from sklearn.metrics import log_loss

from kaggle_agent.bundle import (
    MALFORMED_PREDICTIONS, BundleRunError, PersistedModelCandidate, folder_hash_of, load_bundle,
    score_bundle, validate_predictions,
)
from kaggle_agent.evaluation import FrozenSplit, ProtocolHashMismatchError
from kaggle_agent.sandbox import ProtectedPath, hash_paths

X = FRAME.drop(columns="y")
Y = FRAME["y"]


def test_fit_persists_the_model_and_predict_runs_in_a_separate_process(tmp_path):
    recording = RecordingSandbox()
    _, runner = make_runner(mean_bundle(tmp_path / "src"), sandbox=recording, input_kind="values")
    model_dir = tmp_path / "model"
    runner.fit_model(X.iloc[:100], Y.iloc[:100], 4, model_dir)
    params = json.loads((model_dir / "params.json").read_text())
    assert params["value"] == pytest.approx(Y.iloc[:100].mean())
    assert len(recording.calls) == 1 and recording.calls[0][1].args[1] == "fit"
    predictions = runner.predict_with_model(model_dir, X.iloc[100:105], 4)
    assert len(recording.calls) == 2 and recording.calls[1][1].args[1] == "predict"
    assert predictions.tolist() == [pytest.approx(params["value"])] * 5
    # predictions came back as JSON; two different staging folders, hence two child processes
    assert recording.calls[0][1].args[0] != recording.calls[1][1].args[0]
    assert "x.csv" in recording.staged[1] and "x.csv" not in recording.staged[0]
    assert "train.csv" in recording.staged[0] and "train.csv" not in recording.staged[1]
    # a stored model works in a brand new runner (another process, no shared state)
    _, other = make_runner(mean_bundle(tmp_path / "src2"), input_kind="values")
    assert other.predict_with_model(model_dir, X.iloc[:2], 1).tolist() == predictions[:2].tolist()


def test_persisted_model_candidate_ignores_the_training_data(tmp_path):
    _, runner = make_runner(mean_bundle(tmp_path / "src"), input_kind="values")
    model_dir = tmp_path / "model"
    runner.fit_model(X, Y, 1, model_dir)
    candidate = PersistedModelCandidate(runner, model_dir)
    out = candidate.fit_predict(None, None, X.iloc[:3], 9)
    assert out.tolist() == [pytest.approx(Y.mean())] * 3


def test_the_runner_never_unpickles_and_the_parent_never_opens_the_model(tmp_path):
    recording = RecordingSandbox()
    folder = write_bundle(tmp_path / "src", model(
        "import pickle, os\nwith open(os.path.join(os.environ['BUNDLE_MODEL_DIR'], 'm.pkl'), 'rb') as h:\n"
        "    v = pickle.load(h)\nreturn [v] * len(X)",
        "import pickle, os\nwith open(os.path.join(os.environ['BUNDLE_MODEL_DIR'], 'm.pkl'), 'wb') as h:\n"
        "    pickle.dump(3, h)",
    ))
    _, runner = make_runner(folder, sandbox=recording, input_kind="values")
    assert runner.fit_predict(X.iloc[:50], Y.iloc[:50], X.iloc[50:53], 1).tolist() == [3, 3, 3]
    for files in recording.staged:  # the parent staged data only, never a model file
        assert not any(name.endswith(".pkl") for name in files)
    source = (BUNDLE_DIR / "runner.py").read_text()
    assert "import pickle" not in source and "json.loads" in source


def test_a_bundle_that_keeps_its_model_only_in_memory_fails_with_code_error(tmp_path):
    memory = write_bundle(tmp_path / "a", model(
        "return [STATE['v']] * len(X)", "STATE['v'] = 1", header="STATE = {}",
    ))
    engine, _ = frozen_engine(make_protocol())
    _, runner = make_runner(memory, input_kind="labels")
    outcome = score_bundle(engine, runner, FRAME, attempt_id="A1", execution_id="E1", config={}, data_hash="d")
    record = outcome.execution
    assert record.outcome == "failed" and record.failure_kind == "code_error"
    assert record.resources["sandbox_status"] == "crashed" and "KeyError" in record.reason
    ok = write_bundle(tmp_path / "b", model(
        "import os\nreturn [open(os.path.join(os.environ['BUNDLE_MODEL_DIR'], 'v')).read() == '1'] * len(X)",
        "import os\nopen(os.path.join(os.environ['BUNDLE_MODEL_DIR'], 'v'), 'w').write('1')",
    ))
    _, runner = make_runner(ok, input_kind="labels")
    good = score_bundle(engine, runner, FRAME, attempt_id="A2", execution_id="E2", config={}, data_hash="d")
    assert good.execution.outcome == "succeeded"


def test_fit_predict_goes_through_two_processes_and_a_clean_model_folder(tmp_path):
    recording = RecordingSandbox()
    _, runner = make_runner(mean_bundle(tmp_path / "src"), sandbox=recording, input_kind="values")
    runner.fit_predict(X.iloc[:60], Y.iloc[:60], X.iloc[60:62], 2)
    assert [call[1].args[1] for call in recording.calls] == ["fit", "predict"]


# -- the hash ignores model/ ---------------------------------------------------------------------


def test_a_model_folder_does_not_change_the_bundle_hash(tmp_path):
    folder = constant_bundle(tmp_path)
    loaded = load_bundle(folder)
    sandbox_hash = hash_paths([ProtectedPath("bundle", folder)])["bundle"]
    assert loaded.folder_hash == sandbox_hash == folder_hash_of(folder)  # byte-compatible
    (folder / "sub").mkdir()
    (folder / "sub" / "helper.py").write_text("X = 1\n")
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "m.pyc").write_bytes(b"1")
    assert folder_hash_of(folder) == hash_paths([ProtectedPath("bundle", folder)])["bundle"]
    before = folder_hash_of(folder)
    (folder / "model").mkdir()
    (folder / "model" / "weights.bin").write_bytes(b"\x00\x01")
    (folder / "model" / "deep").mkdir()
    (folder / "model" / "deep" / "more.json").write_text("{}")
    assert folder_hash_of(folder) == before
    (folder / "model.py").write_text((folder / "model.py").read_text() + "# changed\n")
    assert folder_hash_of(folder) != before


def test_a_registered_bundle_with_a_model_folder_does_not_report_folder_changed(tmp_path):
    folder = mean_bundle(tmp_path / "src")
    _, runner = make_runner(folder, input_kind="values")
    runner.fit_model(X, Y, 1, folder / "model")  # persisted right inside the bundle folder
    assert (folder / "model" / "params.json").is_file()
    out = runner.predict_with_model(folder / "model", X.iloc[:2], 1)
    assert len(out) == 2
    assert folder_hash_of(folder) == runner.bundle.folder_hash
    outcome = score_bundle(
        frozen_engine(make_protocol(metric="rmse", direction="minimize", validation_params={"n_folds": 3},
                                    holdout_params={}))[0],
        runner, FRAME, attempt_id="A1", execution_id="E1", config={}, data_hash="d",
    )
    assert outcome.execution.outcome == "succeeded"


# -- contract unchanged --------------------------------------------------------------------------


def test_contract_signatures_and_documentation(tmp_path):
    from kaggle_agent.bundle import contract

    doc = contract.__doc__
    assert "fit(train)" in doc and "predict(bundle_id, X)" in doc and "BUNDLE_MODEL_DIR" in doc
    folder = write_bundle(tmp_path, model(
        "assert bundle_id == 'B1' and 'y' not in X.columns\nreturn [0] * len(X)",
        "assert 'y' in train.columns",
    ))
    loaded = load_bundle(folder)  # no manifest change: still the same two entry points
    assert set(loaded.manifest.entry_points) == {"fit", "predict"}
    _, runner = make_runner(folder)
    assert runner.fit_predict(X.iloc[:40], Y.iloc[:40], FRAME.iloc[40:43], 3).tolist() == [0, 0, 0]
    readme = (BUNDLE_DIR.parents[2] / "README.md").read_text()
    assert "fit(train)" in readme and "predict(bundle_id, X)" in readme and "BUNDLE_MODEL_DIR" in readme


# -- matrix predictions --------------------------------------------------------------------------


def three_class_frame(n=180, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                         "y": np.tile([0, 1, 2], n // 3)})


MULTI = three_class_frame()


def multi_protocol(**overrides):
    return make_protocol(metric="log_loss", direction="minimize", **overrides)


def test_multiclass_probability_matrix_is_validated_and_scored_like_sklearn(tmp_path):
    from evaluation_helpers import evaluate

    engine, _ = frozen_engine(multi_protocol())
    _, runner = make_runner(priors_bundle(tmp_path), input_kind="probabilities")
    outcome = score_bundle(engine, runner, MULTI, attempt_id="A1", execution_id="E1", config={}, data_hash="d")
    assert outcome.error is None and outcome.execution.outcome == "succeeded"
    split = FrozenSplit.create(MULTI, engine.protocol)
    part = split.partitions()[0]
    x_train, y_train = split.data(part.train_rows)
    x_valid, y_valid = split.data(part.valid_rows)
    matrix = runner.fit_predict(x_train, y_train, x_valid, 1)
    assert matrix.shape == (len(x_valid), 3) and np.allclose(matrix.sum(axis=1), 1)

    class InMemory:  # the same matrix without any process, scored by the same engine
        def fit_predict(self, x_train, y_train, x_valid, seed):
            shares = y_train.value_counts(normalize=True).sort_index().to_numpy()
            return np.tile(shares, (len(x_valid), 1))

    direct = evaluate(engine, InMemory(), MULTI, attempt_id="A2")
    assert outcome.result.scores == pytest.approx(direct.scores)
    assert outcome.result.scores[part.id] == pytest.approx(log_loss(y_valid, matrix))


GOOD = [[0.2, 0.3, 0.5], [1, 0, 0]]


@pytest.mark.parametrize("raw, rows, kind, fragment", [
    ([[0.5, 0.5], [0.5]], 2, "probabilities", "different width"),
    ([[1.0], [1.0]], 2, "probabilities", "at least 2"),
    ([[0.5, float("nan")], [0.5, 0.5]], 2, "probabilities", "NaN"),
    ([[1.5, -0.5], [0.5, 0.5]], 2, "probabilities", "[0, 1]"),
    ([[0.5, 0.4], [0.5, 0.5]], 2, "probabilities", "sum to 1"),
    ([[0.5, 0.5]] * 3, 2, "probabilities", "expected 2"),
    ([[0.5, 0.5]] * 2, 2, "labels", "not a matrix"),
    ([[0.5, 0.5]] * 2, 2, "values", "not a matrix"),
    ([[0.5, 0.5], 0.5], 2, "probabilities", "mixed"),
    ([[0.5, None], [0.5, 0.5]], 2, "probabilities", "null"),
    ([[0.5, "a"], [0.5, 0.5]], 2, "probabilities", "numbers"),
    ([{"a": 1}, {"a": 2}], 2, "probabilities", "numbers"),
])
def test_invalid_matrices_are_rejected_with_distinct_reasons(raw, rows, kind, fragment):
    with pytest.raises(BundleRunError) as error:
        validate_predictions(raw, rows, kind)
    assert error.value.status == MALFORMED_PREDICTIONS and fragment in error.value.reason


def test_valid_matrices_and_one_dimensional_predictions_still_pass():
    assert validate_predictions(GOOD, 2, "probabilities").shape == (2, 3)
    assert validate_predictions([[3.0, -2.0, 7.5]] * 2, 2, "scores").shape == (2, 3)
    assert validate_predictions([0.2, 0.9], 2, "probabilities").tolist() == [0.2, 0.9]
    assert validate_predictions([1, 0], 2, "labels").tolist() == [1, 0]
    assert validate_predictions(["a", "b"], 2, "labels").tolist() == ["a", "b"]
    assert validate_predictions([[0.5, 0.5 + 5e-7]], 1, "probabilities").shape == (1, 2)


def test_a_metric_that_cannot_score_a_matrix_is_a_recorded_failure(tmp_path):
    engine, _ = frozen_engine(make_protocol(metric="roc_auc", direction="maximize"))
    frame = three_class_frame()
    folder = write_bundle(tmp_path, model(
        "return [[0.2, 0.3, 0.5]] * len(X)"
    ))
    _, runner = make_runner(folder, input_kind="scores")
    outcome = score_bundle(engine, runner, frame, attempt_id="A1", execution_id="E1", config={}, data_hash="d")
    record = outcome.execution
    assert record.outcome == "failed" and record.failure_kind == "code_error"
    assert record.resources == {"sandbox_status": "scoring_error", "retryable": False}
    assert "cannot score" in record.reason and outcome.result is None
    assert outcome.error.retryable_as_code_fix is False


def test_a_protocol_hash_mismatch_still_propagates_from_score_bundle(tmp_path):
    engine, _ = frozen_engine(make_protocol())
    engine.protocol = make_protocol(threshold_k=9.0)
    _, runner = make_runner(constant_bundle(tmp_path))
    with pytest.raises(ProtocolHashMismatchError):
        score_bundle(engine, runner, FRAME, attempt_id="A1", execution_id="E1", config={}, data_hash="d")

