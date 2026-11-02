"""submit: refusals, dry run, confirmed send, bounded wait and the crash-safe marker (cases 21 to 33)."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
from orchestrator_helpers import REVIEW_ACCEPT, World, competition, drive_to_review, drive_to_stopped, sample_csv

from kaggle_agent.kaggle import SubmissionSpec
from kaggle_agent.orchestrator import submission
from kaggle_agent.orchestrator.errors import SubmitRefused
from kaggle_agent.web import create_app


def stopped_world(tmp_path, **kwargs):
    world = World(tmp_path, **kwargs)
    drive_to_stopped(world)
    return world


def sends(world):
    return [c for c in world.kaggle.calls if c[0] == "send"]


def marker(world):
    return world.project / "out/submissions/toy-1.pending.json"


def submissions(world):
    registry = world.registry()
    try:
        return registry.list_submissions("toy-1")
    finally:
        registry.close()


# 21
def test_nothing_is_submitted_before_the_review_is_accepted(tmp_path):
    world = World(tmp_path)
    drive_to_review(world)
    code, text = world.run("submit", "--confirm-submit")
    assert code == 20 and "refused" in text and "nothing was sent" in text and "AWAITING_REVIEW" in text
    assert sends(world) == [] and submissions(world) == []


# 22
def test_dry_run_prints_what_would_be_sent_and_sends_nothing(tmp_path):
    world = stopped_world(tmp_path)
    code, text = world.run("submit")
    assert code == 0 and "DRY RUN" in text
    for word in ("file:", "sha256:", "rows: 10", "columns: id, y", "message:", "chosen bundle:", "holdout",
                 "daily limit: 5 of 5"):
        assert word in text, word
    assert sends(world) == [] and submissions(world) == []


# 23
def test_nothing_is_sent_without_the_flag_and_a_submit_key_in_the_lead_file_is_an_error(tmp_path):
    world = World(tmp_path)
    drive_to_review(world)
    world.lead(REVIEW_ACCEPT + "\n[submit]\nconfirm = true\n")
    code, text = world.run("resume")
    assert code == 12 and "submit" in text and world.state() == "AWAITING_REVIEW"
    world.lead(REVIEW_ACCEPT + "submit = true\n")
    assert world.run("resume")[0] == 12 and world.state() == "AWAITING_REVIEW"
    assert sends(world) == []
    world.lead(REVIEW_ACCEPT)
    assert world.run("resume")[0] == 0 and world.state() == "STOPPED"
    assert sends(world) == [] and world.run("submit")[0] == 0 and sends(world) == []


# 24
def test_an_invalid_submission_file_is_refused(tmp_path):
    world = stopped_world(tmp_path / "width")
    spec = world.kaggle.get_submission_spec("toy")
    wide = SubmissionSpec(columns=spec.columns + ("z",), id_column=spec.id_column, expected_ids=spec.expected_ids)
    world.kaggle.get_submission_spec = lambda competition: wide
    code, text = world.run("submit", "--confirm-submit")
    assert code == 21 and "1 value(s) per row" in text and sends(world) == []

    world = stopped_world(tmp_path / "noid")
    test = world.project / "data/test.csv"
    test.write_text(test.read_text(encoding="utf-8").replace("id,", "ident,", 1), encoding="utf-8")
    code, text = world.run("submit", "--confirm-submit")
    assert code == 21 and "id column" in text and sends(world) == []

    world = stopped_world(tmp_path / "ids")
    spec = world.kaggle.get_submission_spec("toy")
    other = SubmissionSpec(columns=spec.columns, id_column=spec.id_column, expected_ids=("1", "2"))
    world.kaggle.get_submission_spec = lambda competition: other
    code, text = world.run("submit", "--confirm-submit")
    assert code == 21 and "invalid" in text and sends(world) == []
    assert not marker(world).exists()


# 25
def test_the_daily_limit_refuses_with_the_reset_time(tmp_path):
    world = stopped_world(tmp_path / "kaggle", comp=competition(daily_limit=1))
    world.kaggle.submit("toy", world.project / "data/sample_submission.csv", "earlier")
    code, text = world.run("submit", "--confirm-submit")
    assert code == 22 and "daily limit" in text and "resets at" in text
    assert len(sends(world)) == 1 and submissions(world) == []


def test_recorded_submissions_fill_the_guard_even_when_kaggle_reports_room(tmp_path):
    world = stopped_world(tmp_path, comp=competition(daily_limit=1))
    assert world.run("submit", "--confirm-submit")[0] == 0
    entries = submissions(world)
    fresh = World(tmp_path / "fresh", comp=competition(daily_limit=1))  # Kaggle itself knows of no send
    fresh.clock.time = world.clock.time
    with pytest.raises(SubmitRefused) as refusal:
        submission.check_daily_limit(fresh.deps, "toy", entries)
    assert refusal.value.code == 22


# 26
def test_a_confirmed_submission_is_sent_once_recorded_scored_and_shown(tmp_path):
    world = stopped_world(tmp_path, comp=competition(score_script=(None, None, 0.81)))
    code, text = world.run("submit", "--confirm-submit")
    assert code == 0 and "submitted: fake-1" in text and "public score: 0.81" in text
    assert len(sends(world)) == 1
    [entry] = submissions(world)
    assert entry.status == "complete" and entry.public_score == 0.81
    report = (world.project / "out/reports/toy-1.md").read_text(encoding="utf-8")
    assert "## Submission" in report and "fake-1" in report and "0.81000" in report
    registry = world.registry()
    try:
        html = create_app(registry).test_client().get("/cycles/toy-1").get_data(as_text=True)
    finally:
        registry.close()
    assert "fake-1" in html and "0.81" in html
    assert world.run("submit", "--confirm-submit")[0] == 0 and len(sends(world)) == 1
    _, status = world.run("status")
    assert "fake-1" in status


# 27
def test_the_wait_is_bounded_and_a_rerun_resumes_without_sending_again(tmp_path):
    world = stopped_world(tmp_path, comp=competition(score_script=(None,)))
    code, text = world.run("submit", "--confirm-submit")
    assert code == 23 and "still pending" in text and "nothing is sent twice" in text
    assert len(sends(world)) == 1 and submissions(world)[0].public_score is None
    world.kaggle.competitions["toy"] = competition(score_script=(None, 0.7))
    code, text = world.run("submit", "--confirm-submit")
    assert code == 0 and "public score: 0.7" in text
    assert len(sends(world)) == 1 and submissions(world)[0].public_score == 0.7


def crash_after_send(world, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(submission, "record_submission", broken)
    code, text = world.run("submit", "--confirm-submit")
    assert code == 1 and "pending marker" in text
    monkeypatch.undo()


# 28
def test_a_crash_between_send_and_record_leaves_a_marker_and_the_rerun_refuses(tmp_path, monkeypatch):
    world = stopped_world(tmp_path)
    crash_after_send(world, monkeypatch)
    assert json.loads(marker(world).read_text(encoding="utf-8"))["state"] == "sending"
    assert submissions(world) == [] and len(sends(world)) == 1
    code, text = world.run("submit", "--confirm-submit")
    assert code == 24 and "Kaggle submissions page" in text and "--assume-not-sent" in text
    assert len(sends(world)) == 1
    assert world.run("submit")[0] == 0  # a dry run is still possible


# 29
@pytest.mark.parametrize("how", ["flag", "delete"])
def test_the_lead_can_proceed_after_checking_by_hand(tmp_path, monkeypatch, how):
    world = stopped_world(tmp_path)
    crash_after_send(world, monkeypatch)
    if how == "delete":
        marker(world).unlink()
        code, text = world.run("submit", "--confirm-submit")
    else:
        code, text = world.run("submit", "--confirm-submit", "--assume-not-sent")
    # the fake Kaggle client has the same file: the registry has no row, so a new send happens
    assert code == 0, text
    [entry] = submissions(world)
    assert entry.public_score == 0.8 and len(sends(world)) == 2
    assert json.loads(marker(world).read_text(encoding="utf-8"))["state"] == "recorded"


# 30
def test_the_marker_is_written_before_the_send_and_marked_recorded_after(tmp_path, monkeypatch):
    world = stopped_world(tmp_path)
    seen = {}
    original = world.kaggle.submit

    def spy(slug, file, message):
        seen["marker"] = json.loads(marker(world).read_text(encoding="utf-8"))
        seen["entries"] = len(submissions(world))
        return original(slug, file, message)

    world.kaggle.submit = spy
    assert world.run("submit", "--confirm-submit")[0] == 0
    before = seen["marker"]
    sha = hashlib.sha256((world.project / "out/submissions/toy-1.csv").read_bytes()).hexdigest()
    assert before["state"] == "sending" and before["sha256"] == sha and before["message"].startswith("kaggle-agent")
    assert before["at"].endswith("+00:00") and seen["entries"] == 0
    after = json.loads(marker(world).read_text(encoding="utf-8"))
    assert after["state"] == "recorded" and after["submission_id"] == "fake-1" and after["sha256"] == sha
    assert world.run("submit", "--confirm-submit")[0] == 0 and len(sends(world)) == 1
    assert not list(marker(world).parent.glob("*.tmp*"))


# 31
def test_the_dry_run_never_writes_a_marker_or_a_row(tmp_path):
    world = stopped_world(tmp_path)
    before = (world.project / "registry.db").read_bytes()
    assert world.run("submit")[0] == 0
    assert not marker(world).exists() and submissions(world) == []
    assert (world.project / "registry.db").read_bytes() == before


# 32
def test_the_submission_comes_from_the_stored_model_through_predict_without_pickle(tmp_path):
    world = stopped_world(tmp_path, comp=competition(sample=sample_csv(), test=None), real_sandbox=True)
    assert world.run("submit", "--confirm-submit")[0] == 0
    lines = (world.project / "out/submissions/toy-1.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,y" and len(lines) == 11 and lines[1].startswith("100,")
    assert {line.split(",")[1] for line in lines[1:]} == {"0"}  # the majority class
    assert "command.predict" in Path(submission.__file__).read_text(encoding="utf-8")
    for path in Path(submission.__file__).parent.glob("*.py"):
        imported = {n.names[0].name for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                    if isinstance(n, ast.Import)}
        assert not imported & {"pickle", "cloudpickle", "joblib"}, path


def test_a_matrix_of_probabilities_fills_several_columns(tmp_path, monkeypatch):
    world = stopped_world(tmp_path, comp=competition(sample=sample_csv(columns="id,y"), test=None))
    spec = SubmissionSpec(columns=("id", "p0", "p1"), id_column="id",
                          expected_ids=world.kaggle.get_submission_spec("toy").expected_ids)

    class Result:
        code, lines = 0, ()

    def fake_predict(persisted, test, output, sandbox, timeout):
        output.write_text(json.dumps([[0.25, 0.75]] * 10), encoding="utf-8")
        return Result()

    monkeypatch.setattr(submission.command, "predict", fake_predict)
    target = world.project / "out/m.csv"
    rows, columns = submission.build_submission(world.project, world.project / "data/test.csv", spec, target,
                                                world.deps, 10)
    assert rows == 10 and columns == ["id", "p0", "p1"]
    assert target.read_text(encoding="utf-8").splitlines()[1] == "100,0.25,0.75"


# 33
def test_the_holdout_is_read_once_across_resume_and_submit(tmp_path):
    world = stopped_world(tmp_path)
    world.run("status")
    world.run("submit")
    world.run("submit", "--confirm-submit")
    registry = world.registry()
    try:
        assert len(registry.list_holdout_reads("toy-1")) == 1
        assert registry.get_holdout_outcome("toy-1") is not None
    finally:
        registry.close()
