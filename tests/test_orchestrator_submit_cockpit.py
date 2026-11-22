"""What the submission cockpit needs from the submit command (cases 15, 16, 17, 18, 19)."""

import hashlib
import json

import pytest
from orchestrator_helpers import competition
from overrides_helpers import stopped_world

from kaggle_agent.orchestrator import submission
from kaggle_agent.orchestrator.cockpit import CockpitFiles, RunLock


def sends(world):
    return [c for c in world.kaggle.calls if c[0] == "send"]


def submissions(world):
    registry = world.registry()
    try:
        return registry.list_submissions("toy-1")
    finally:
        registry.close()


def marker(world):
    return json.loads((world.project / "out/submissions/toy-1.pending.json").read_text(encoding="utf-8"))


def submit(world, tmp_path, *args):
    out = tmp_path / "submit.json"
    out.unlink(missing_ok=True)
    code, text = world.run("submit", "--out", str(out), *args)
    return code, json.loads(out.read_text(encoding="utf-8")), text


def crash_after_send(world, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(submission, "record_submission", broken)
    assert world.run("submit", "--confirm-submit")[0] == 1
    monkeypatch.undo()


# 15
def test_the_dry_run_result_has_the_structured_preview_and_sends_nothing(tmp_path):
    world = stopped_world(tmp_path)
    code, result, _ = submit(world, tmp_path)
    data = result["data"]
    assert code == 0 and result["code"] == 0
    sha = hashlib.sha256((world.project / "out/submissions/toy-1.csv").read_bytes()).hexdigest()
    assert data["sha256"] == sha and data["rows"] == 10 and data["columns"] == ["id", "y"]
    assert data["message"].startswith("kaggle-agent toy-1") and data["file"].endswith("toy-1.csv")
    assert data["daily_limit"]["left"] == 5 and data["daily_limit"]["limit"] == 5 and data["daily_limit"]["resets_at"]
    assert data["chosen"] == {"attempt_id": "toy-1-R0-c1", "bundle_id": "toy-1-R0-c1-b1", "by": "the program",
                              "program_choice": "toy-1-R0-c1"}
    assert data["cv"] == pytest.approx(0.7291666666666666) and data["metric"] == "accuracy"
    assert data["marker"] is None
    assert data["holdout"]["status"] == "succeeded" and data["holdout"]["score"] is None  # sealed in a job file
    assert sends(world) == [] and submissions(world) == []
    assert not (world.project / "out/submissions/toy-1.pending.json").exists()
    assert "sealed" in " ".join(result["lines"])  # the job file never carries the holdout number


# 16
def test_a_confirmed_send_needs_the_hash_of_the_preview_and_sends_exactly_once(tmp_path):
    world = stopped_world(tmp_path)
    sha = submit(world, tmp_path)[1]["data"]["sha256"]
    code, result, text = submit(world, tmp_path, "--confirm-submit", "--expect-sha256", "0" * 64)
    assert code == 21 and "preview it again" in " ".join(result["lines"])
    assert sends(world) == [] and submissions(world) == []
    code, result, _ = submit(world, tmp_path, "--confirm-submit", "--expect-sha256", sha)
    assert code == 0 and len(sends(world)) == 1 and [s.submission_id for s in submissions(world)] == ["fake-1"]
    assert submit(world, tmp_path, "--confirm-submit", "--expect-sha256", sha)[0] == 0
    assert len(sends(world)) == 1  # the same file is never sent twice


# 17
def test_the_daily_limit_shows_zero_left_with_the_reset_and_refuses_with_22(tmp_path):
    world = stopped_world(tmp_path, comp=competition(daily_limit=1))
    world.kaggle.submit("toy", world.project / "data/sample_submission.csv", "earlier")
    code, result, _ = submit(world, tmp_path)
    assert code == 22 and result["data"]["daily_limit"]["left"] == 0 and result["data"]["daily_limit"]["resets_at"]
    code, result, _ = submit(world, tmp_path, "--confirm-submit")
    assert code == 22 and len(sends(world)) == 1 and submissions(world) == []
    assert "nothing was sent" in " ".join(result["lines"])


# 18
def test_a_pending_score_is_reported_with_23_and_a_later_call_records_the_score_without_a_second_send(tmp_path):
    world = stopped_world(tmp_path, comp=competition(score_script=(None,)))
    code, result, _ = submit(world, tmp_path, "--confirm-submit")
    assert code == 23 and len(sends(world)) == 1 and submissions(world)[0].public_score is None
    world.kaggle.competitions["toy"] = competition(score_script=(None, 0.7))
    code, result, _ = submit(world, tmp_path, "--confirm-submit")
    assert code == 0 and submissions(world)[0].public_score == 0.7 and len(sends(world)) == 1


# 19
def test_marker_recovery_by_not_sent_sends_once_and_by_adopt_sends_nothing(tmp_path, monkeypatch):
    world = stopped_world(tmp_path / "a")
    crash_after_send(world, monkeypatch)
    assert marker(world)["state"] == "sending"
    code, result, _ = submit(world, tmp_path, "--confirm-submit")
    assert code == 24 and len(sends(world)) == 1 and marker(world)["state"] == "sending"
    assert "Kaggle submissions page" in " ".join(result["lines"])
    code, _, _ = submit(world, tmp_path, "--confirm-submit", "--assume-not-sent")
    assert code == 0 and len(sends(world)) == 2 and len(submissions(world)) == 1
    assert marker(world)["state"] == "recorded"
    # the other way: the DS checked Kaggle and the file WAS sent (fake-1 exists there)
    world = stopped_world(tmp_path / "b")
    crash_after_send(world, monkeypatch)
    code, result, text = submit(world, tmp_path, "--adopt-id", "fake-1")
    assert code == 0, (result, text)
    assert len(sends(world)) == 1 and [s.submission_id for s in submissions(world)] == ["fake-1"]
    assert marker(world)["state"] == "recorded" and marker(world)["submission_id"] == "fake-1"
    # an id Kaggle does not know is refused and records nothing
    world = stopped_world(tmp_path / "c")
    crash_after_send(world, monkeypatch)
    code, _, _ = submit(world, tmp_path, "--adopt-id", "nope-404")
    assert code != 0 and submissions(world) == [] and marker(world)["state"] == "sending"


# 16
def test_a_confirmed_send_takes_the_run_lock_and_is_refused_while_a_run_is_active(tmp_path):
    world = stopped_world(tmp_path)
    lock = RunLock(CockpitFiles(world.project))
    assert lock.acquire()
    try:
        code, result, _ = submit(world, tmp_path, "--confirm-submit")
        assert code == 1 and "a run is active" in " ".join(result["lines"])
        assert sends(world) == [] and submissions(world) == []
        assert submit(world, tmp_path)[0] == 0  # a dry run needs no lock
    finally:
        lock.release()
    assert submit(world, tmp_path, "--confirm-submit")[0] == 0 and len(sends(world)) == 1
