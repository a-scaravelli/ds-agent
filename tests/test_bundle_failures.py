"""Failures of a bundle run become execution records with a failure kind (cases 13 to 20)."""

import time

import pytest
from bundle_helpers import (
    BUNDLE_DIR, EVALUATION_DIR, FRAME, SHORT_TIMEOUT, CannedSandbox, RecordingSandbox,
    constant_bundle, engine_and_registry, make_runner, policy, protected_writer_bundle,
    raiser_bundle, real_package_bytes, returning_bundle, sleeper_bundle, write_bundle,
)

from kaggle_agent.bundle import BundleError, BundleRunError, load_bundle, BundleRunner, score_bundle
from kaggle_agent.evaluation import ProtocolHashMismatchError
from kaggle_agent.evaluation.metrics import INPUT_KINDS
from kaggle_agent.registry import FAILURE_KINDS
from kaggle_agent.sandbox import ProtectedPath, SandboxResult, SubprocessSandbox


def score(engine, runner):
    return score_bundle(engine, runner, FRAME, attempt_id="A1", execution_id="E1", config={}, data_hash="d")


def test_timeout_yields_failure_kind_timeout(tmp_path):
    engine, _ = engine_and_registry()
    _, runner = make_runner(sleeper_bundle(tmp_path), sandbox_policy=policy(SHORT_TIMEOUT))
    started = time.monotonic()
    record = score(engine, runner).execution
    assert time.monotonic() - started < SHORT_TIMEOUT + 30
    assert record.outcome == "failed" and record.failure_kind == "timeout"
    assert "timed out" in record.reason
    assert record.resources["sandbox_status"] == "timed_out"
    assert record.failure_kind in FAILURE_KINDS and record.metrics == ()


@pytest.mark.parametrize("where", ["fit", "predict"])
def test_crash_yields_failure_kind_code_error_with_stderr_tail(tmp_path, where):
    engine, _ = engine_and_registry()
    _, runner = make_runner(raiser_bundle(tmp_path, where))
    outcome = score(engine, runner)
    record = outcome.execution
    assert record.outcome == "failed" and record.failure_kind == "code_error"
    assert record.resources == {"sandbox_status": "crashed", "retryable": True}
    assert "boom from the bundle" in record.reason
    assert outcome.result is None and outcome.error.status == "crashed"


MALFORMED = [
    ("[0]", "labels"),
    ("[float('nan')] * len(X)", "values"),
    ("[None] * len(X)", "labels"),
    ("None", "labels"),
    ("[[0, 1]] * len(X)", "values"),
    ("['a'] * len(X)", "values"),
    ("[1.5] * len(X)", "probabilities"),
    ("[float('inf')] * len(X)", "scores"),
    ("[0, 'a'] * (len(X) // 2) + [0] * (len(X) % 2)", "labels"),
]


@pytest.mark.parametrize("expression,kind", MALFORMED)
def test_malformed_predictions_are_rejected_before_scoring(tmp_path, monkeypatch, expression, kind):
    assert kind in INPUT_KINDS
    engine, _ = engine_and_registry()

    def forbidden(*args, **kwargs):
        raise AssertionError("the metric must not run on malformed predictions")

    monkeypatch.setattr(engine, "_score", forbidden)
    _, runner = make_runner(returning_bundle(tmp_path, expression), input_kind=kind)
    outcome = score(engine, runner)
    record = outcome.execution
    assert outcome.error.status == "malformed_predictions" and outcome.error.failure_kind == "code_error"
    assert record.outcome == "failed" and record.failure_kind == "code_error"
    assert record.reason.startswith("malformed predictions")
    assert record.resources["sandbox_status"] == "malformed_predictions"


def test_a_bundle_that_exits_without_predictions_is_malformed(tmp_path):
    engine, _ = engine_and_registry()
    folder = returning_bundle(tmp_path, "__import__('os')._exit(0)")
    _, runner = make_runner(folder)
    assert score(engine, runner).error.status == "malformed_predictions"


def test_protected_copy_modified_is_never_retried_and_real_sources_stay_intact(tmp_path):
    before = real_package_bytes()
    engine, _ = engine_and_registry()
    folder = protected_writer_bundle(tmp_path, "env:kaggle_agent/evaluation/metrics.py", chmod_first=True)
    _, runner = make_runner(folder)
    record = score(engine, runner).execution
    assert record.failure_kind == "code_error"
    assert record.resources == {"sandbox_status": "protected_modified", "retryable": False}
    assert "copy:kaggle_agent/evaluation" in record.reason
    assert real_package_bytes() == before
    engine.verify()


def test_protected_temporary_path_modified_is_never_retried(tmp_path):
    before = real_package_bytes()
    guarded = tmp_path / "guarded.txt"
    guarded.write_text("original")
    caller = policy(protected=(ProtectedPath("guarded.txt", guarded),))
    engine, _ = engine_and_registry()
    folder = protected_writer_bundle(tmp_path, str(guarded))
    _, runner = make_runner(folder, sandbox_policy=caller)
    record = score(engine, runner).execution
    assert record.failure_kind == "code_error"
    assert record.resources == {"sandbox_status": "protected_modified", "retryable": False}
    assert "guarded.txt" in record.reason
    assert guarded.read_text() == "originaltampered"
    assert real_package_bytes() == before


def test_diff_rejected_is_never_retried(tmp_path):
    canned = SandboxResult(
        status="diff_rejected", changed_paths=("kaggle_agent/evaluation",),
        reason="diff touches protected files: kaggle_agent/evaluation",
    )
    sandbox = CannedSandbox(canned)
    _, runner = make_runner(constant_bundle(tmp_path), sandbox=sandbox)
    x = FRAME.drop(columns="y").iloc[:3]
    with pytest.raises(BundleRunError) as caught:
        runner.fit_predict(x, FRAME["y"].iloc[:3], x, 1)
    error = caught.value
    assert (error.status, error.failure_kind) == ("diff_rejected", "code_error")
    assert error.retryable_as_code_fix is False and error.result is canned
    record = score(engine_and_registry()[0], runner).execution
    assert record.resources == {"sandbox_status": "diff_rejected", "retryable": False}


def test_modified_protected_file_blocks_the_run_before_the_child_starts(tmp_path):
    guarded = tmp_path / "guarded.txt"
    guarded.write_text("original")
    marker = tmp_path / "marker"
    folder = write_bundle(tmp_path, {"model.py": f"""
def fit(train):
    open({str(marker)!r}, "w").write("started")

def predict(bundle_id, X):
    return [0] * len(X)
"""})
    caller = policy(protected=(ProtectedPath("guarded.txt", guarded),))
    _, runner = make_runner(folder, sandbox_policy=caller)
    guarded.write_text("changed after the runner was built")
    record = score(engine_and_registry()[0], runner).execution
    assert not marker.exists()
    assert record.resources == {"sandbox_status": "protected_modified", "retryable": False}
    assert "guarded.txt" in record.reason and record.failure_kind == "code_error"


def test_frozen_protocol_change_still_blocks_and_propagates(tmp_path):
    from evaluation_helpers import make_protocol

    engine, _ = engine_and_registry()
    engine.protocol = make_protocol(threshold_k=7.0)
    recording = RecordingSandbox()
    _, runner = make_runner(constant_bundle(tmp_path), sandbox=recording)
    with pytest.raises(ProtocolHashMismatchError):
        score(engine, runner)
    assert recording.calls == []  # a real sandbox is used here; nothing was started


@pytest.mark.parametrize("name", ["kaggle_agent/bundle", "kaggle_agent/evaluation"])
def test_duplicate_protected_names_in_the_callers_policy(tmp_path, name):
    real = BUNDLE_DIR if name.endswith("bundle") else EVALUATION_DIR
    loaded = load_bundle(constant_bundle(tmp_path))

    def build(path):
        caller = policy(protected=(ProtectedPath(name, path),))
        return BundleRunner(SubprocessSandbox(), caller, loaded, target_column="y", input_kind="labels")

    same = build(real)
    names = [p.name for p in same.policy.protected]
    assert names.count(name) == 1 and len(names) == 2
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(BundleError, match=name):
        build(other)


def test_folder_drifting_from_the_registered_hash_is_detected(tmp_path):
    folder = constant_bundle(tmp_path)
    recording = RecordingSandbox()
    _, runner = make_runner(folder, sandbox=recording)
    (folder / "model.py").write_text((folder / "model.py").read_text() + "\n# drift\n")
    record = score(engine_and_registry()[0], runner).execution
    assert record.failure_kind == "code_error" and "folder hash" in record.reason
    assert record.resources == {"sandbox_status": "folder_changed", "retryable": False}
    assert recording.calls == []
