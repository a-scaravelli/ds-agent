"""The submission cockpit through the page: preview, confirm, limit, score, marker recovery, token scan (15-21)."""

import json
import re
from pathlib import Path

from cockpit_finish_helpers import FinishWorld
from orchestrator_helpers import competition

from kaggle_agent.orchestrator import submission

WEB = Path(__file__).resolve().parents[1] / "src" / "kaggle_agent" / "web"
TOKEN = "KGAT_fake_secret_token_0123456789abcdef"
WORD = {"confirm": True, "typed": "SEND"}


def sends(fw):
    return [c for c in fw.world.kaggle.calls if c[0] == "send"]


def preview(fw):
    done = fw.wait_job(fw.ok("/api/submission/preview")["job"])
    return done["result"]


def confirm(fw, sha, **body):
    return fw.post("/api/submission/confirm", {"sha256": sha, **WORD, **body})


def submissions(fw):
    reg = fw.registry()
    try:
        return reg.list_submissions("toy-1")
    finally:
        reg.close()


# 15
def test_the_dry_run_preview_shows_the_file_hash_rows_columns_message_and_the_daily_limit(tmp_path):
    fw = FinishWorld(tmp_path)
    before = fw.registry_hash()
    result = preview(fw)
    data = result["data"]
    assert result["code"] == 0 and data["rows"] == 10 and data["columns"] == ["id", "y"] and len(data["sha256"]) == 64
    assert data["message"].startswith("kaggle-agent toy-1") and data["daily_limit"]["left"] == 5
    assert sends(fw) == [] and fw.registry_hash() == before
    assert not (fw.project / "out/submissions/toy-1.pending.json").exists()
    page = fw.page("/cycles/toy-1/submission")
    assert "dry run" in page.lower() and "Send" in page  # the preview itself is drawn from the job by the page script
    assert [a["what"] for a in fw.lines("audit.jsonl")] == ["submission.preview"]


# 16
def test_the_confirm_needs_the_word_the_hash_and_sends_exactly_once(tmp_path):
    fw = FinishWorld(tmp_path)
    sha = preview(fw)["data"]["sha256"]
    assert fw.post("/api/submission/confirm", {"sha256": sha, "typed": "SEND"}).status_code == 400  # no confirm
    assert fw.post("/api/submission/confirm", {"sha256": sha, "confirm": True, "typed": "send"}).status_code == 400
    assert fw.post("/api/submission/confirm", {"sha256": "nothex", **WORD}).status_code == 400
    assert sends(fw) == []
    wrong = fw.wait_job(confirm(fw, "0" * 64).get_json()["data"]["job"])
    assert wrong["result"]["code"] == 21 and sends(fw) == [] and submissions(fw) == []
    fw.wait_job(confirm(fw, sha).get_json()["data"]["job"])
    assert len(sends(fw)) == 1 and [s.submission_id for s in submissions(fw)] == ["fake-1"]
    assert fw.controller.argvs[-1][1] == "submit"


# 17
def test_a_daily_limit_of_zero_is_shown_with_the_reset_and_the_confirm_is_refused_with_22(tmp_path):
    fw = FinishWorld(tmp_path, comp=competition(daily_limit=1))
    fw.world.kaggle.submit("toy", fw.project / "data/sample_submission.csv", "earlier")
    data = preview(fw)["data"]
    assert data["daily_limit"]["left"] == 0 and data["daily_limit"]["resets_at"]
    done = fw.wait_job(confirm(fw, data["sha256"]).get_json()["data"]["job"])
    assert done["result"]["code"] == 22 and "nothing was sent" in " ".join(done["result"]["lines"])
    assert len(sends(fw)) == 1 and submissions(fw) == []  # only the earlier one
    assert "0" in fw.page("/cycles/toy-1/submission")


# 18
def test_a_pending_score_is_followed_by_a_refresh_and_the_same_file_is_never_sent_twice(tmp_path):
    fw = FinishWorld(tmp_path, comp=competition(score_script=(None,)))
    sha = preview(fw)["data"]["sha256"]
    done = fw.wait_job(confirm(fw, sha).get_json()["data"]["job"])
    assert done["result"]["code"] == 23 and len(sends(fw)) == 1
    history = fw.get_json("/api/runs/toy-1/submissions/preview")["history"]
    assert len(history) == 1 and history[0]["public_score"] is None
    fw.world.kaggle.competitions["toy"] = competition(score_script=(None, 0.7))
    done = fw.wait_job(confirm(fw, sha).get_json()["data"]["job"])
    assert done["result"]["code"] == 0 and len(sends(fw)) == 1
    history = fw.get_json("/api/runs/toy-1/submissions/preview")["history"]
    assert history[0]["public_score"] == 0.7 and history[0]["submission_id"] == "fake-1"


def crash_after_send(fw, monkeypatch, sha):
    def broken(*args, **kwargs):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(submission, "record_submission", broken)
    fw.wait_job(confirm(fw, sha).get_json()["data"]["job"])
    monkeypatch.undo()


# 19
def test_a_pending_marker_shows_both_recovery_options_and_each_is_audited(tmp_path, monkeypatch):
    fw = FinishWorld(tmp_path / "a")
    sha = preview(fw)["data"]["sha256"]
    crash_after_send(fw, monkeypatch, sha)
    marker = fw.get_json("/api/runs/toy-1/submissions/preview")["marker"]
    assert marker["state"] == "sending" and marker["recovery_needed"] is True
    page = fw.page("/cycles/toy-1/submission")
    assert "NOT sent" in page and "WAS sent" in page
    done = fw.wait_job(confirm(fw, sha).get_json()["data"]["job"])
    assert done["result"]["code"] == 24 and len(sends(fw)) == 1
    assert fw.post("/api/submission/recover", {"action": "not_sent", "confirm": True}).status_code == 400  # reason
    assert fw.post("/api/submission/recover", {"action": "adopt", "submission_id": "a b;c", "confirm": True}
                   ).status_code == 400
    fw.wait_job(fw.ok("/api/submission/recover", {"action": "not_sent", "reason": "the page was empty",
                                                  "confirm": True})["job"])
    assert len(sends(fw)) == 2 and len(submissions(fw)) == 1
    assert fw.get_json("/api/runs/toy-1/submissions/preview")["marker"]["recovery_needed"] is False
    assert fw.post("/api/submission/recover", {"action": "not_sent", "reason": "x", "confirm": True}
                   ).status_code == 409  # nothing left to recover
    assert "submission.recover.not_sent" in [a["what"] for a in fw.lines("audit.jsonl")]
    # the other way: Kaggle has it, so it is recorded and nothing is sent
    fw = FinishWorld(tmp_path / "b")
    sha = preview(fw)["data"]["sha256"]
    crash_after_send(fw, monkeypatch, sha)
    fw.wait_job(fw.ok("/api/submission/recover", {"action": "adopt", "submission_id": "fake-1", "confirm": True})["job"])
    assert len(sends(fw)) == 1 and [s.submission_id for s in submissions(fw)] == ["fake-1"]
    assert "submission.recover.adopt" in [a["what"] for a in fw.lines("audit.jsonl")]


# 20
def test_the_token_is_never_read_or_shown_by_the_web(tmp_path, monkeypatch):
    new_sources = [WEB / n for n in ("override_writes.py", "override_api.py", "override_pages.py", "submission_data.py",
                                     "bundle_data.py", "labels_data.py", "history_data.py", "report_export.py",
                                     "cycles_data.py", "controller.py", "static/js/overrides.js")]
    new_sources += sorted((WEB / "templates").glob("*.html")) + sorted((WEB / "static/css").glob("*.css"))
    pattern = re.compile(r"KAGGLE_|access_token|kaggle\.json|os\.environ|getenv|KGAT")
    environment = WEB / "environment.py"  # the one module that only checks the PRESENCE of a credential
    for path in new_sources:
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name
    assert "open(" not in environment.read_text(encoding="utf-8").split("def _file_state")[0]
    monkeypatch.setenv("KAGGLE_API_TOKEN", TOKEN)
    fw = FinishWorld(tmp_path)
    sha = preview(fw)["data"]["sha256"]
    fw.wait_job(confirm(fw, sha).get_json()["data"]["job"])
    bodies = [fw.client.get(p).get_data(as_text=True) for p in (
        "/cycles/toy-1/submission", "/api/runs/toy-1/submissions/preview", "/api/jobs", "/cycles/toy-1/gap",
        "/api/runs/toy-1/gap", "/")]
    for body in bodies:
        assert TOKEN not in body
    for path in fw.project.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md", ".log", ".txt") and "registry" not in path.name:
            assert TOKEN not in path.read_text(encoding="utf-8", errors="ignore"), path


# 21
def test_the_history_shows_cv_and_the_holdout_sealed_until_it_is_unsealed(tmp_path):
    fw = FinishWorld(tmp_path, comp=competition(score_script=(0.7, 0.71)))
    sha = preview(fw)["data"]["sha256"]
    fw.wait_job(confirm(fw, sha).get_json()["data"]["job"])
    rows = fw.get_json("/api/runs/toy-1/submissions/preview")["history"]
    assert rows[0]["cv"] == 0.7291666666666666 and rows[0]["holdout"] == "sealed" and rows[0]["public_score"] == 0.7
    assert "sealed" in fw.page("/cycles/toy-1/submission") and "0.4167" not in fw.page("/cycles/toy-1/submission")
    fw.ok("/api/holdout/unseal", {"kind": "first", "confirm": True})
    rows = fw.get_json("/api/runs/toy-1/submissions/preview")["history"]
    assert abs(rows[0]["holdout"] - 0.4166666666666667) < 1e-9
    assert "data-render" in fw.page("/cycles/toy-1/submission")  # the table is drawn from the API by the page script
    assert json.dumps(rows)
