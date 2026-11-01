"""The two commands, with real child processes and few partitions (cases 14 and 15)."""

import json
import subprocess
import sys

import numpy as np
import pytest
from bundle_helpers import RecordingSandbox
from pipeline_helpers import CYCLE, FRAME, make_real_pipeline

from kaggle_agent.pipeline import load_persisted
from kaggle_agent.pipeline.__main__ import main
from kaggle_agent.sandbox import SubprocessSandbox


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """One real pipeline run (module scoped: it starts a dozen child processes)."""
    tmp_path = tmp_path_factory.mktemp("real").resolve()
    pipe, db = make_real_pipeline(tmp_path)
    result = pipe.run()
    pipe.registry.close()
    csv = tmp_path / "data.csv"
    FRAME.to_csv(csv, index=False)
    assert result.outcome.status == "stopped" and result.report.holdout["status"] == "succeeded"
    return result.report, db, csv, tmp_path


def argv(report, db, csv):
    return ["reproduce", "--persisted", report.persisted_path, "--db", str(db), "--data", str(csv)]


def test_reproduce_gives_the_reported_metric_from_the_persisted_bundle_code(stored, capsys):
    report, db, csv, _ = stored
    recording = RecordingSandbox()
    assert main(argv(report, db, csv), sandbox=recording) == 0
    out = capsys.readouterr().out
    reproduced, reported = [float(line.rsplit(": ", 1)[1]) for line in out.splitlines()[:2]]
    assert reported == report.chosen["validation_mean"]
    assert abs(reproduced - reported) <= 1e-9
    assert "reproduced validation accuracy" in out and "reported validation accuracy" in out
    # retrained partition by partition: a fit and a predict process each, on a temporary model
    stages = [call[1].args[1] for call in recording.calls]
    assert stages == ["fit", "predict"] * 3
    persisted_model = f"{report.persisted_path}/model"
    assert all(persisted_model not in " ".join(call[1].args) for call in recording.calls)


def test_a_different_csv_gives_a_different_metric_and_a_non_zero_exit(stored, tmp_path, capsys):
    report, db, _, _ = stored
    other = FRAME.copy()
    other["y"] = (np.random.default_rng(5).random(len(other)) < 0.45).astype(int)  # same columns, other labels
    csv = tmp_path / "other.csv"
    other.to_csv(csv, index=False)
    assert main(argv(report, db, csv), sandbox=SubprocessSandbox()) == 1
    err = capsys.readouterr().err
    assert "differ" in err and "reproduced validation" in err


def test_a_changed_model_or_code_file_fails_the_hash_check_before_any_process_starts(stored, capsys):
    from pathlib import Path

    report, db, csv, _ = stored
    root = Path(report.persisted_path)
    model_file = next((root / "model").iterdir())
    code_file = root / "code" / "model.py"
    for target, part in ((model_file, "model/"), (code_file, "code/")):
        original = target.read_text()
        target.write_text(original + " ")
        try:
            recording = RecordingSandbox()
            assert main(argv(report, db, csv), sandbox=recording) == 1
            err = capsys.readouterr().err
            assert "changed" in err and part + target.name in err
            assert recording.calls == []  # refused by load_persisted, not by a metric mismatch
        finally:
            target.write_text(original)
    assert main(argv(report, db, csv), sandbox=RecordingSandbox()) == 0


def test_reproduce_needs_a_registry_with_a_report(stored, tmp_path, capsys):
    from kaggle_agent.registry import SqliteRegistry

    report, _, csv, _ = stored
    empty = tmp_path / "empty.db"
    SqliteRegistry(str(empty)).close()
    assert main(argv(report, empty, csv), sandbox=RecordingSandbox()) == 1
    assert main(argv(report, tmp_path / "missing.db", csv), sandbox=RecordingSandbox()) == 2
    capsys.readouterr()


def test_predict_uses_the_stored_model_and_writes_json_from_another_directory(stored, tmp_path, monkeypatch):
    report, _, _, _ = stored
    features = FRAME.drop(columns="y").iloc[:7]
    csv = tmp_path / "x.csv"
    features.to_csv(csv, index=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "predictions.json"
    recording = RecordingSandbox()
    assert main(["predict", "--persisted", report.persisted_path, "--input", str(csv),
                 "--output", str(output)], sandbox=recording) == 0
    predictions = json.loads(output.read_text())
    persisted = load_persisted(report.persisted_path)
    expected = int(json.loads(next((persisted.model_dir).iterdir()).read_text())["label"])
    assert predictions == [expected] * 7 and persisted.bundle_id == report.chosen["bundle_id"]
    assert [call[1].args[1] for call in recording.calls] == ["predict"]  # only the predict stage


def test_the_commands_run_as_a_module_in_a_fresh_process(stored, tmp_path):
    report, _, _, _ = stored
    csv = tmp_path / "x.csv"
    FRAME.drop(columns="y").iloc[:3].to_csv(csv, index=False)
    output = tmp_path / "out.json"
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    done = subprocess.run(
        [sys.executable, "-I", "-m", "kaggle_agent.pipeline", "predict", "--persisted", report.persisted_path,
         "--input", str(csv), "--output", str(output)],
        cwd=elsewhere, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert len(json.loads(output.read_text())) == 3


def test_predict_refuses_a_bad_persisted_folder(tmp_path, capsys):
    csv = tmp_path / "x.csv"
    csv.write_text("a\n1\n")
    assert main(["predict", "--persisted", str(tmp_path), "--input", str(csv),
                 "--output", str(tmp_path / "o.json")], sandbox=RecordingSandbox()) == 1
    assert "not a persisted bundle" in capsys.readouterr().err
