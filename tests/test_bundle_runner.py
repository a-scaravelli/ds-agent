"""The runner on fake bundles: scoring, the Candidate interface, import paths, the sandbox (cases 6 to 8, 11, 12)."""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from bundle_helpers import (
    BUNDLE_DIR, FRAME, RecordingSandbox, constant_bundle, engine_and_registry, make_runner, model,
    policy, real_package_bytes, write_bundle,
)
from evaluation_helpers import evaluate

from kaggle_agent.bundle import BundleRunner, load_bundle, score_bundle
from kaggle_agent.evaluation import Candidate, FrozenSplit
from kaggle_agent.sandbox import Sandbox, SandboxResult, hash_paths


class ConstantCandidate:
    def __init__(self, value):
        self.value = value

    def fit_predict(self, x_train, y_train, x_valid, seed):
        return np.full(len(x_valid), self.value)


def score(engine, runner, attempt="A1", n=1):
    return score_bundle(
        engine, runner, FRAME, attempt_id=attempt, execution_id=f"{attempt}-E{n}",
        config={"bundle": runner.bundle.bundle_id}, data_hash="data",
    )


def test_fake_bundle_is_scored_by_the_runner_and_the_engine(tmp_path):
    engine, _ = engine_and_registry()
    loaded, runner = make_runner(constant_bundle(tmp_path, 0))
    outcome = score(engine, runner)
    assert outcome.error is None and outcome.execution.outcome == "succeeded"
    expected = evaluate(engine, ConstantCandidate(0), FRAME, attempt_id="A2")
    assert outcome.result.scores == expected.scores
    got = {m.partition: m.mean for m in outcome.execution.metrics}
    assert got == {m.partition: m.mean for m in expected.execution.metrics}
    assert outcome.execution.code_hash == loaded.folder_hash
    assert outcome.execution.protocol_hash == engine.registry.get_cycle("C1").protocol_hash


def test_runner_is_a_candidate_on_validation_and_holdout_paths(tmp_path):
    engine, _ = engine_and_registry()
    before = real_package_bytes()
    recording = RecordingSandbox()
    _, runner = make_runner(constant_bundle(tmp_path, 1), sandbox=recording)
    assert list(inspect.signature(runner.fit_predict).parameters) == list(
        inspect.signature(Candidate.fit_predict).parameters)[1:]
    split = FrozenSplit.create(FRAME, engine.protocol)
    part = split.partitions()[0]
    x_train, y_train = split.data(part.train_rows)
    x_valid, _ = split.data(part.valid_rows)
    predictions = runner.fit_predict(x_train, y_train, x_valid, 5)
    assert len(predictions) == len(x_valid) and set(predictions.tolist()) == {1}
    with engine.holdout_session(FRAME) as session:
        result = session.evaluate_holdout(runner, "A1")
    holdout_y = FRAME["y"].iloc[split._holdout_rows]
    assert result.rows == len(holdout_y) and result.score == pytest.approx((holdout_y == 1).mean())
    # the second run staged the holdout features only, with no target column
    staged = recording.staged[-1]
    assert staged["x.csv"].decode().splitlines()[0].split(",") == ["x1", "x2", "group", "t"]
    assert len(staged["x.csv"].decode().splitlines()) == len(holdout_y) + 1
    engine.verify()  # the frozen protocol and the evaluation code are unchanged
    assert real_package_bytes() == before


def test_the_bundle_never_reports_its_own_score(tmp_path):
    engine, _ = engine_and_registry()
    claim = model(
        """
        print("score: 999.0")
        open("score.json", "w").write('{"score": 999.0}')
        return [0] * len(X)
        """
    )
    _, runner = make_runner(write_bundle(tmp_path, claim))
    outcome = score(engine, runner)
    assert outcome.execution.outcome == "succeeded"
    assert all(m.mean != 999.0 for m in outcome.execution.metrics)
    expected = evaluate(engine, ConstantCandidate(0), FRAME, attempt_id="A2")
    assert outcome.result.mean == expected.mean


def test_import_paths_are_explicit_despite_python_isolated_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "elsewhere"))
    modules = model(
        """
        import os
        import helpers
        if "PYTHONPATH" in os.environ:
            raise RuntimeError("PYTHONPATH reached the child")
        if "bundle-run-" not in helpers.__file__:
            raise RuntimeError("helper not imported from the staged copy: " + helpers.__file__)
        return [helpers.VALUE] * len(X)
        """
    )
    folder = write_bundle(tmp_path, {**modules, "helpers.py": "VALUE = 0\n"})
    _, runner = make_runner(folder)
    result = score(engine_and_registry()[0], runner)
    assert result.error is None and result.execution.outcome == "succeeded"


class WritingSandbox(Sandbox):
    """Records the call and answers with canned predictions, starting nothing."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def run(self, policy, request):
        self.calls.append((policy, request))
        if request.args[1] == "predict":
            Path(request.args[0], "predictions.json").write_text(json.dumps(self.predictions))
        return SandboxResult(status="succeeded", exit_code=0)


def test_runner_makes_two_sandbox_calls_with_protected_packages_and_baseline(tmp_path):
    fake = WritingSandbox([0, 0, 0])
    caller = policy(timeout=7)
    loaded = load_bundle(constant_bundle(tmp_path))
    runner = BundleRunner(fake, caller, loaded, target_column="y", input_kind="labels")
    x = FRAME.drop(columns="y").iloc[:3]
    out = runner.fit_predict(x, FRAME["y"].iloc[:3], x, 3)
    assert out.tolist() == [0, 0, 0]
    ((used, fit_request), (used_again, request)) = fake.calls
    assert [fit_request.args[1], request.args[1]] == ["fit", "predict"]
    assert used_again is used and fit_request.entry_point == request.entry_point
    assert request.entry_point == BUNDLE_DIR / "child.py" and request.command is None
    assert used.timeout_seconds == 7
    names = {p.name for p in used.protected}
    assert names == {"kaggle_agent/bundle", "kaggle_agent/evaluation"}
    assert set(request.expected_hashes) == names
    assert dict(request.expected_hashes) == hash_paths(used.protected)
    assert not Path(request.args[0]).exists()  # the staging folder is always removed
