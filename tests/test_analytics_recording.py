"""Recording by wrapping the runner: the real orchestrator flow with fakes, limits, failures (cases 2, 3, 5, 8, 9, 10, 12, 13, 43)."""

import hashlib
import io
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analytics_helpers import (  # noqa: F401  (fixtures)
    attempts_of,
    client_of,
    cycle_of,
    failure_world,
    finished_world,
    index_lines,
    labels_world,
    make_world,
    proba_world,
)
from evaluation_helpers import make_protocol
from orchestrator_helpers import InProcessSandbox, drive_to_review, train_csv

from kaggle_agent.analytics.oof_format import decode_oof, decode_snapshot
from kaggle_agent.bundle.errors import BundleRunError
from kaggle_agent.evaluation import FrozenSplit, Protocol, derive_seed
from kaggle_agent.orchestrator import analytics_record, build
from kaggle_agent.orchestrator.analytics_record import RecordingCandidate
from kaggle_agent.proposals import ValidationError
from kaggle_agent.web import store


def protocol_of(world) -> Protocol:
    return Protocol.from_dict(json.loads((world.project / "project.json").read_text(encoding="utf-8"))["protocol"])


def frame_of(world) -> pd.DataFrame:
    return pd.read_csv(world.project / "data" / "train.csv")


def partition_metrics(world, attempt_id: str) -> dict:
    registry = world.registry()
    try:
        done = [e for e in registry.get_attempt(attempt_id).executions if e.outcome == "succeeded"]
        return {m.partition[len("validation/"):]: m.mean for m in done[-1].metrics if m.partition.startswith("validation/")}
    finally:
        registry.close()


def last_lines(world) -> dict:
    out = {}
    for line in index_lines(world):
        out[(line["kind"], line.get("bundle_id"))] = line
    return out


# 8
def test_every_attempt_has_all_partition_scores_equal_to_the_registry(proba_world):
    lines = last_lines(proba_world)
    attempts = attempts_of(proba_world)
    assert len(attempts) == 2
    for attempt in attempts:
        line = lines[("oof", f"{attempt}-b1")]
        assert line["status"] == "ok" and line["partitions"] == 6 and len(line["fold_scores"]) == 6
        registry = partition_metrics(proba_world, attempt)
        assert set(registry) == set(line["fold_scores"]) == {f"r{r}/f{f}" for r in range(2) for f in range(3)}
        for pid, value in registry.items():
            assert line["fold_scores"][pid] == pytest.approx(value, abs=1e-9)


# 9
def test_the_oof_file_is_hashed_complete_and_has_probabilities_only_when_the_metric_has_them(proba_world, labels_world):
    for world, expect_matrix in ((proba_world, True), (labels_world, False)):
        protocol = protocol_of(world)
        development = set(FrozenSplit.create(frame_of(world), protocol).development_rows().tolist())
        for attempt in attempts_of(world):
            line = last_lines(world)[("oof", f"{attempt}-b1")]
            path = world.project / "analytics" / line["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == line["sha256"] == line["file"].rsplit("-", 1)[1].split(".")[0] + line["sha256"][12:]
            table = decode_oof(path.read_bytes(), classification=True)
            frame = table.frame
            assert table.matrix is expect_matrix
            assert list(frame.columns)[:4] == ["row", "rep", "fold", "truth"]
            assert ("p0" in frame.columns and "p1" in frame.columns) is expect_matrix
            assert ("prediction" in frame.columns) is not expect_matrix
            for rep in range(protocol.repetitions):
                rows = frame[frame["rep"] == rep]["row"].tolist()
                assert sorted(rows) == sorted(development)  # every development row once per repetition


# 2
def test_the_holdout_is_never_read_at_runtime_and_never_appears_in_a_file_or_an_answer(proba_world):
    world = proba_world
    protocol = protocol_of(world)
    frame = frame_of(world)
    split = FrozenSplit.create(frame, protocol)
    development = set(split.development_rows().tolist())
    holdout = sorted(set(range(len(frame))) - development)
    assert holdout
    registry = world.registry()
    try:
        cycle = cycle_of(world)
        assert registry.list_holdout_reads(cycle) == [] and registry.get_holdout_outcome(cycle) is None
    finally:
        registry.close()
    marker_ids = {int(frame.iloc[i]["id"]) for i in holdout}
    lines = last_lines(world)
    snapshot = decode_snapshot((world.project / "analytics" / lines[("development", None)]["file"]).read_bytes())
    assert set(snapshot.index) <= development and not marker_ids & set(snapshot["id"])
    client = client_of(world)
    attempt = attempts_of(world)[-1]
    urls = [f"/api/attempts/{attempt}/{part}" for part in ("analytics", "folds", "panels", "segments", "flags", "features")]
    urls += [f"/api/attempts/{attempt}/hardest?limit=200", f"/attempts/{attempt}/analytics"]
    for line in lines.values():
        if line["kind"] == "oof":
            table = decode_oof((world.project / "analytics" / line["file"]).read_bytes())
            assert set(table.frame["row"]) <= development
    hardest = client.get(urls[-2]).get_json()["data"]
    assert {r["row"] for r in hardest["rows"]} <= development
    assert not {r["id"] for r in hardest["rows"]} & marker_ids
    for url in urls:
        assert client.get(url).status_code == 200, url


class FakeInner:
    """A runner without fit_model: ``fit_predict`` only (the recorder then says 'describe missing')."""

    def __init__(self):
        self.calls = 0

    def fit_predict(self, x_train, y_train, x_valid, seed):
        self.calls += 1
        return np.full(len(x_valid), y_train.mode().iloc[0])


def drive(candidate, frame, protocol):
    split = FrozenSplit.create(frame, protocol)
    out = []
    for part in split.partitions():
        x_train, y_train = split.data(part.train_rows)
        x_valid, _ = split.data(part.valid_rows)
        out.append(candidate.fit_predict(x_train, y_train, x_valid, derive_seed(protocol.seed, "candidate", part.repetition)))
    return out


def unit(tmp_path, inner=None):
    frame = pd.read_csv(io.StringIO(train_csv()))
    protocol = make_protocol(validation_params={"n_folds": 3, "stratify_by": "y"})
    folder = Path(tmp_path).resolve() / "project"
    folder.mkdir()
    return RecordingCandidate(inner or FakeInner(), frame, protocol, "B1", folder), frame, protocol, folder


# 3
def test_a_partition_with_a_holdout_row_is_refused_and_nothing_is_written(tmp_path, capsys):
    candidate, frame, protocol, folder = unit(tmp_path)
    real = FrozenSplit.create(frame, protocol)
    holdout_row = int(sorted(set(range(len(frame))) - set(real.development_rows().tolist()))[0])

    class Tampered:
        def partitions(self):
            part = real.partitions()[0]
            return (type(part)(part.id, part.repetition, part.fold, part.train_rows, np.array([holdout_row])),)

        def data(self, rows):
            return real.data(rows)

        def development_data(self):
            return real.development_data()

    candidate._split = Tampered()
    features = frame.drop(columns=["y"]).iloc[[holdout_row]]
    plain = FakeInner()
    expected = plain.fit_predict(frame.iloc[:10].drop(columns=["y"]), frame["y"].iloc[:10], features, 1)
    got = candidate.fit_predict(frame.iloc[:10].drop(columns=["y"]), frame["y"].iloc[:10], features, 1)
    assert list(got) == list(expected)  # the predictions are unchanged
    assert not (folder / "analytics").exists()
    assert "only development rows" in capsys.readouterr().err
    with pytest.raises(ValidationError):
        real.data(np.array([holdout_row]))


# 10
def test_a_table_over_the_row_cap_is_recorded_as_too_large_without_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(analytics_record, "MAX_ROWS", 5)
    candidate, frame, protocol, folder = unit(tmp_path)
    drive(candidate, frame, protocol)
    lines = [json.loads(x) for x in (folder / "analytics" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    oof = next(x for x in lines if x["kind"] == "oof")
    assert oof["status"] == "too_large" and "file" not in oof
    assert not list((folder / "analytics").glob("oof/*"))


def test_a_full_analytics_budget_is_recorded_as_budget_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(analytics_record, "MAX_FOLDER_BYTES", 0)
    candidate, frame, protocol, folder = unit(tmp_path)
    drive(candidate, frame, protocol)
    lines = [json.loads(x) for x in (folder / "analytics" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["status"] for x in lines] == ["budget_exhausted"]


# 12
@pytest.mark.parametrize("fault", ["folder_is_a_file", "disk_full", "bug"])
def test_a_recorder_problem_never_changes_the_predictions_or_fails_the_attempt(tmp_path, monkeypatch, capsys, fault):
    candidate, frame, protocol, folder = unit(tmp_path)
    expected = drive(FakeInner(), frame, protocol)
    if fault == "folder_is_a_file":
        (folder / "analytics").write_text("not a folder", encoding="utf-8")
    elif fault == "disk_full":
        def full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(analytics_record.os, "replace", full)
    else:
        def broken(*args, **kwargs):
            raise RuntimeError("recorder bug")
        monkeypatch.setattr(analytics_record, "encode_oof", broken)
    got = drive(candidate, frame, protocol)
    assert [list(x) for x in got] == [list(x) for x in expected]
    assert "analytics:" in capsys.readouterr().err
    assert not list(folder.glob("analytics/**/.*tmp*"))


# 12
def test_a_bundle_run_error_of_the_inner_runner_propagates_unchanged(tmp_path):
    error = BundleRunError("crashed", "code_error", "the child crashed")

    class Crashing:
        def fit_predict(self, *args):
            raise error

    candidate, frame, protocol, folder = unit(tmp_path, Crashing())
    with pytest.raises(BundleRunError) as caught:
        drive(candidate, frame, protocol)
    assert caught.value is error and not (folder / "analytics").exists()


# 13
def test_a_crashed_bundle_writes_no_file_and_the_fix_bundle_is_recorded_under_its_own_id(failure_world):
    world = failure_world
    lines = last_lines(world)
    baseline, failed, good = attempts_of(world)
    assert ("oof", f"{baseline}-b1") not in lines  # crashed on its second fold: nothing written
    assert lines[("oof", f"{baseline}-b1-f1")]["status"] == "ok"
    assert not [k for k in lines if k[1] and k[1].startswith(f"{failed}-")]
    client = client_of(world)
    ok = client.get(f"/api/attempts/{baseline}/analytics").get_json()["data"]
    assert ok["bundle_id"] == f"{baseline}-b1-f1" and ok["recorded"]["oof"]["status"] == "ok"
    bad = client.get(f"/api/attempts/{failed}/analytics").get_json()["data"]
    assert bad["not_recorded"]["code"] == "run_failed" and "failed" in bad["not_recorded"]["reason"]
    page = client.get(f"/attempts/{failed}/analytics").get_data(as_text=True)
    assert "Not recorded for this attempt" in page and "run of this attempt failed" in page
    assert good in attempts_of(world) and lines[("oof", f"{good}-b1")]["status"] == "ok"


class CountingSandbox(InProcessSandbox):
    pass


def _scores_and_stages(world):
    registry = world.registry()
    try:
        cycle = cycle_of(world)
        out = {}
        for summary in registry.list_attempts(cycle):
            detail = registry.get_attempt(summary.attempt_id)
            out[summary.attempt_id] = [(e.id, e.outcome, tuple((m.partition, m.mean) for m in e.metrics))
                                       for e in detail.executions]
        return out, list(world.sandbox.runs)
    finally:
        registry.close()


# 5
def test_wrapping_changes_neither_scores_nor_executions_nor_sandbox_calls(tmp_path, monkeypatch):
    with_wrapper = finished_world(tmp_path / "with", sandbox=CountingSandbox())
    monkeypatch.setattr(build, "RecordingCandidate", lambda inner, *args, **kwargs: inner)
    without = finished_world(tmp_path / "without", sandbox=CountingSandbox())
    scores_a, runs_a = _scores_and_stages(with_wrapper)
    scores_b, runs_b = _scores_and_stages(without)
    assert scores_a == scores_b and runs_a == runs_b and runs_a  # fit then predict per partition
    assert runs_a[:2] == ["fit", "predict"] and runs_a.count("fit") == runs_a.count("predict")
    assert (with_wrapper.project / "analytics" / "index.jsonl").exists()
    assert not (without.project / "analytics").exists()  # only the analytics files differ


# 43
@pytest.mark.parametrize("supervised", [False, True])
def test_the_real_flow_records_every_successful_bundle_and_results_equal_the_unwrapped_run(tmp_path, monkeypatch, supervised):
    def run_flow(base):
        world = make_world(base)
        flag = ("--supervised",) if supervised else ()
        assert world.run("start", *flag)[0] == 10
        world.lead('[plan]\ndecision = "approve"\n')
        assert world.run("resume", *flag)[0] == 10
        return world

    recorded = run_flow(tmp_path / "recorded")
    monkeypatch.setattr(build, "RecordingCandidate", lambda inner, *args, **kwargs: inner)
    plain = run_flow(tmp_path / "plain")
    lines = index_lines(recorded)
    registry = recorded.registry()
    try:
        cycle = cycle_of(recorded)
        bundles = {b.id for b in registry.list_bundles(cycle)}
    finally:
        registry.close()
    oof = {x["bundle_id"] for x in lines if x["kind"] == "oof" and x["status"] == "ok"}
    assert oof and oof <= bundles
    for x in lines:
        if x.get("file"):
            assert hashlib.sha256((recorded.project / "analytics" / x["file"]).read_bytes()).hexdigest() == x["sha256"]
    assert any(x["kind"] == "describe" for x in lines) and sum(x["kind"] == "development" for x in lines) == 1
    assert _scores_and_stages(recorded)[0] == _scores_and_stages(plain)[0]
    if supervised:
        def kinds(world):
            records, _ = store.read_jsonl(world.project / "cockpit" / "events.jsonl")
            return [r["type"] for r in records if r["type"] != "heartbeat"]
        assert kinds(recorded) == kinds(plain)


# 43
def test_a_second_recording_of_the_same_bundle_does_not_duplicate_files_and_the_latest_line_wins(tmp_path):
    candidate, frame, protocol, folder = unit(tmp_path)
    drive(candidate, frame, protocol)
    before = sorted(p.name for p in (folder / "analytics").rglob("*") if p.is_file())
    again, _, _, _ = RecordingCandidate(FakeInner(), frame, protocol, "B1", folder), 0, 0, 0
    drive(again, frame, protocol)  # a resume after a kill records the bundle once more
    with open(folder / "analytics" / "index.jsonl", "ab") as handle:
        handle.write(b'{"v":1,"kind":"oof","bundle_id":"B1","sha2')  # torn tail of a killed writer
    after = sorted(p.name for p in (folder / "analytics").rglob("*") if p.is_file())
    assert before == after
    latest, skipped = store.analytics_index(folder)
    assert skipped == 1 and sorted(k for k in latest if k[0] == "oof") == [("oof", "B1")]
    blob, reason = store.read_analytics_file(folder, latest[("oof", "B1")])
    assert reason is None and decode_oof(blob, classification=True).frame["row"].nunique() == 48
