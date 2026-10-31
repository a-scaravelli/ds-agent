"""The fake client: download, info, rules, submission, limit and score, all offline."""

import hashlib
import logging
from datetime import timedelta

import pytest

from kaggle_agent.kaggle import (
    CompetitionNotFoundError,
    Credentials,
    DailyLimitReachedError,
    FakeKaggleClient,
    KaggleClient,
    KaggleError,
    RulesNotAcceptedError,
    ScoreTimeoutError,
    SubmissionFormatError,
)
from kaggle_helpers import DAY, SAMPLE_CSV, SECRET, TEST, TRAIN, Clock, make_fake, write_submission


def test_download_writes_files_with_hashes(tmp_path):
    client, _ = make_fake()
    first = client.download("comp", tmp_path / "one")
    second = client.download("comp", tmp_path / "two")
    by_name = {f.name: f for f in first.files}
    assert (tmp_path / "one" / "train.csv").read_bytes() == TRAIN
    assert by_name["train.csv"].sha256 == hashlib.sha256(TRAIN).hexdigest()
    assert by_name["train.csv"].size == len(TRAIN)
    assert first.data_hash == second.data_hash
    changed, _ = make_fake(files={"train.csv": TRAIN + b"x", "test.csv": TEST})
    assert changed.download("comp", tmp_path / "three").data_hash != first.data_hash


def test_download_defaults_to_data_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, _ = make_fake()
    result = client.download("comp")
    assert result.directory.replace("\\", "/") == "data/comp"
    assert (tmp_path / "data" / "comp" / "test.csv").is_file()


def test_unknown_competition(tmp_path):
    client, _ = make_fake()
    with pytest.raises(CompetitionNotFoundError):
        client.download("x", tmp_path)
    with pytest.raises(CompetitionNotFoundError):
        client.get_competition_info("x")


def test_competition_info_returns_the_three_texts():
    client, _ = make_fake(description="D text", rules="R text", evaluation="E text")
    info = client.get_competition_info("comp")
    assert (info.description, info.rules, info.evaluation) == ("D text", "R text", "E text")


def test_rules_not_accepted_is_a_manual_step(tmp_path):
    client, _ = make_fake(rules_accepted=False)
    path = write_submission(tmp_path / "s.csv")
    for call in (
        lambda: client.download("comp", tmp_path / "d"),
        lambda: client.get_competition_info("comp"),
        lambda: client.submit("comp", path, "m"),
    ):
        with pytest.raises(RulesNotAcceptedError, match="website"):
            call()
    public = {n for n in dir(KaggleClient) if not n.startswith("_")}
    assert not any("accept" in name for name in public)
    assert client.sent == []


def test_invalid_submission_is_not_sent(tmp_path):
    client, _ = make_fake()
    bad = write_submission(tmp_path / "bad.csv", ["a,1"])
    with pytest.raises(SubmissionFormatError):
        client.submit("comp", bad, "m")
    assert client.sent == [] and not any(c[0] == "send" for c in client.calls)
    assert client.daily_limit("comp").used == 0


def test_daily_limit_is_respected(tmp_path):
    client, _ = make_fake(daily_limit=2)
    path = write_submission(tmp_path / "s.csv")
    client.submit("comp", path, "one")
    client.submit("comp", path, "two")
    status = client.daily_limit("comp")
    assert (status.used, status.remaining) == (2, 0)
    with pytest.raises(DailyLimitReachedError):
        client.submit("comp", path, "three")
    assert len(client.sent) == 2


def test_daily_limit_resets_next_day(tmp_path):
    client, clock = make_fake(daily_limit=1)
    path = write_submission(tmp_path / "s.csv")
    client.submit("comp", path, "one")
    with pytest.raises(DailyLimitReachedError):
        client.submit("comp", path, "two")
    clock.now += timedelta(days=1)
    assert client.daily_limit("comp").remaining == 1
    client.submit("comp", path, "two")
    assert len(client.sent) == 2


def test_client_knows_limit_and_submissions_left():
    earlier = [DAY - timedelta(hours=1), DAY - timedelta(minutes=5), DAY - timedelta(days=2)]
    client, _ = make_fake(daily_limit=5, earlier=earlier)
    status = client.daily_limit("comp")
    assert (status.limit, status.used, status.remaining) == (5, 2, 3)
    assert status.resets_at == DAY.replace(hour=0, minute=0) + timedelta(days=1)
    with pytest.raises(ValueError):  # a limit nobody reports is a construction error
        make_fake(daily_limit=None)


def test_wait_for_score_polls_until_complete(tmp_path):
    client, clock = make_fake(score_script=(None, None, 0.87))
    record = client.submit("comp", write_submission(tmp_path / "s.csv"), "msg")
    assert record.status == "pending" and record.public_score is None
    final = client.wait_for_score("comp", record.submission_id, timeout=100, poll_interval=5)
    assert (final.status, final.public_score) == ("complete", 0.87)
    assert final.submission_id == record.submission_id
    assert clock.sleeps == [5, 5]


def test_wait_for_score_times_out_on_the_injected_clock(tmp_path):
    client, clock = make_fake(score_script=(None,))
    record = client.submit("comp", write_submission(tmp_path / "s.csv"), "msg")
    start = clock.now
    with pytest.raises(ScoreTimeoutError):
        client.wait_for_score("comp", record.submission_id, timeout=30, poll_interval=10)
    assert (clock.now - start).total_seconds() == 30


def test_failed_scoring_returns_error_status_without_score(tmp_path):
    client, _ = make_fake(score_script=(None, "error"))
    record = client.submit("comp", write_submission(tmp_path / "s.csv"), "msg")
    final = client.wait_for_score("comp", record.submission_id, 60, 1)
    assert final.status == "error" and final.public_score is None


def test_unknown_submission_id():
    client, _ = make_fake()
    with pytest.raises(KaggleError):
        client.get_submission("comp", "nope")


def test_secret_is_absent_from_logs_and_errors(tmp_path, caplog):
    creds = Credentials("env", token=SECRET)
    assert SECRET not in repr(creds) + str(creds) + f"{creds}"
    client, _ = make_fake(score_script=(None, 0.5), rules_accepted=True)
    messages = []
    with caplog.at_level(logging.DEBUG):
        client.download("comp", tmp_path / "d")
        client.get_competition_info("comp")
        record = client.submit("comp", write_submission(tmp_path / "s.csv"), "m")
        client.wait_for_score("comp", record.submission_id, 60, 1)
        for call in (
            lambda: client.submit("comp", write_submission(tmp_path / "b.csv", ["a,1"]), "m"),
            lambda: client.download("zzz", tmp_path / "z"),
        ):
            with pytest.raises(KaggleError) as info:
                call()
            messages.append(str(info.value))
    assert SECRET not in caplog.text and not any(SECRET in m for m in messages)


def test_fake_has_no_client_state_beyond_its_configuration():
    client, _ = make_fake()
    assert isinstance(client, KaggleClient) and isinstance(client, FakeKaggleClient)
    assert SAMPLE_CSV  # the sample submission is the single source of the spec
    spec = client.get_submission_spec("comp")
    assert spec.columns == ("id", "target") and spec.expected_ids == ("a", "b", "c")


def test_competition_info_has_data_description_and_every_page():
    client, _ = make_fake(
        description="D", rules="R", evaluation="E",
        data_description="DD", extra_pages={"Frequently Asked Questions": "FAQ"},
    )
    info = client.get_competition_info("comp")
    assert info.data_description == "DD"
    assert dict(info.pages) == {
        "description": "D", "rules": "R", "evaluation": "E",
        "data-description": "DD", "Frequently Asked Questions": "FAQ",
    }
    plain, _ = make_fake()
    assert plain.get_competition_info("comp").data_description == ""
    assert "data-description" not in plain.get_competition_info("comp").pages


def test_data_hash_and_file_hashes_are_exposed_without_a_snapshot_accessor(tmp_path):
    import dataclasses

    client, _ = make_fake()
    one = client.download("comp", tmp_path / "a")
    two = client.download("comp", tmp_path / "b")
    assert one.data_hash == two.data_hash and len(one.data_hash) == 64
    assert {"train.csv", "test.csv"} <= {f.name for f in one.files}
    assert all(f.name and len(f.sha256) == 64 for f in one.files)
    assert not hasattr(one, "snapshot")
    assert "snapshot" not in {f.name for f in dataclasses.fields(type(one))}


def test_kaggle_code_does_not_assume_a_single_attempt_iteration():
    import ast

    import kaggle_agent.kaggle as package
    from pathlib import Path

    forbidden = {"attempt_id", "iteration"}
    for path in Path(package.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = set()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {node.name, *(a.arg for a in node.args.args + node.args.kwonlyargs)}
            elif isinstance(node, ast.arg):
                names = {node.arg}
            elif isinstance(node, ast.Name):
                names = {node.id}
            elif isinstance(node, ast.Attribute):
                names = {node.attr}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = {node.target.id}
            assert not names & forbidden, f"{path.name}: {names & forbidden}"
