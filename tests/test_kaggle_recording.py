"""Recording submissions and scores in the registry, and the guards around it."""

import ast
import sqlite3
from pathlib import Path

import pytest
from registry_helpers import make_cycle

from kaggle_agent.kaggle import Credentials, record_score, record_submission
from kaggle_agent.registry import NotFoundError, RegistryError, SqliteRegistry
from kaggle_helpers import SECRET, make_fake, write_submission

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"


def _flow(registry, tmp_path):
    client, _ = make_fake(score_script=(None, 0.91))
    sent = client.submit("comp", write_submission(tmp_path / "s.csv"), "first try")
    record_submission(registry, "C1", sent)
    final = client.wait_for_score("comp", sent.submission_id, 60, 1)
    record_score(registry, final)
    return sent, final


def test_submission_and_score_are_recorded_together(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as registry:
        registry.add_cycle(make_cycle("C1"))
        sent, final = _flow(registry, tmp_path)
        entries = registry.list_submissions("C1")
        assert len(entries) == 1
        entry = entries[0]
        assert (entry.submission_id, entry.competition, entry.file_sha256, entry.message) == (
            sent.submission_id, "comp", sent.file_sha256, "first try",
        )
        assert entry.submitted_at == sent.submitted_at.isoformat()
        assert (entry.status, entry.public_score) == ("complete", 0.91) and entry.completed_at
        with pytest.raises(RegistryError):
            record_score(registry, final)  # accepted once
        with pytest.raises(NotFoundError):
            record_submission(registry, "unknown-cycle", sent)
    with SqliteRegistry(path) as registry:
        assert registry.list_submissions("C1")[0].public_score == 0.91


def test_pending_record_cannot_be_recorded_as_a_result(tmp_path):
    client, _ = make_fake()
    sent = client.submit("comp", write_submission(tmp_path / "s.csv"), "m")
    with SqliteRegistry() as registry:
        registry.add_cycle(make_cycle("C1"))
        record_submission(registry, "C1", sent)
        with pytest.raises(RegistryError):
            record_score(registry, sent)


def test_secret_is_nowhere_in_the_registry_file(tmp_path):
    creds = Credentials("env", token=SECRET)
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as registry:
        registry.add_cycle(make_cycle("C1"))
        _flow(registry, tmp_path)
        assert SECRET not in repr(registry.list_submissions("C1")) + repr(creds)
    db = sqlite3.connect(path)
    dump = "\n".join(db.iterdump())
    db.close()
    assert SECRET not in dump and SECRET not in path.read_bytes().decode("latin-1")


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield "." * node.level + (node.module or "")


def test_only_kaggle_modules_import_credentials_and_results_carry_no_secret():
    import dataclasses

    from kaggle_agent import kaggle

    for path in SRC.rglob("*.py"):
        if "kaggle" in path.relative_to(SRC).parts[:1]:
            continue
        assert not any("credentials" in name.split(".") for name in _imports(path)), path
    for name in ("DownloadedFile", "DownloadResult", "CompetitionInfo", "SubmissionSpec",
                 "SubmissionRecord", "DailyLimit"):
        fields = {f.name for f in dataclasses.fields(getattr(kaggle, name))}
        assert not {f for f in fields if any(w in f for w in ("token", "key", "secret", "credential"))}


def test_the_score_never_changes_the_protocol():
    for path in sorted((SRC / "kaggle").glob("*.py")):
        names = list(_imports(path))
        assert not any("evaluation" in n.split(".") for n in names), path
        if path.name != "recording.py":
            assert not any(n.split(".")[-1] == "registry" or ".registry" in n for n in names), path
    text = (SRC / "kaggle" / "recording.py").read_text(encoding="utf-8")
    for forbidden in ("set_protocol_hash", "add_cycle", "set_cycle_state", "add_comparison"):
        assert forbidden not in text


def test_recording_binds_to_the_cycle_only():
    import ast
    from pathlib import Path

    import kaggle_agent.kaggle.recording as recording

    tree = ast.parse(Path(recording.__file__).read_text(encoding="utf-8"))
    names = {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "cycle_id" in names
    assert not names & {"attempt_id", "round_id", "iteration", "candidate_id"}
