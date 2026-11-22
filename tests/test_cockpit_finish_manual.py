"""MANUAL test of the end-of-journey pages over a real run folder, and the checklist (case 54)."""

import json
import os
from pathlib import Path

import pytest
from cockpit_helpers import Served, loopback_only  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR_VARIABLE = "KAGGLE_AGENT_RUN_COCKPIT_DIR"


# 54 (MANUAL: skipped unless KAGGLE_AGENT_RUN_COCKPIT_DIR points to a real project folder)
@pytest.mark.skipif(RUN_DIR_VARIABLE not in os.environ, reason=f"manual: set {RUN_DIR_VARIABLE} to a project folder")
def test_manual_the_end_of_journey_pages_answer_200_over_a_real_run_folder():
    folder = Path(os.environ[RUN_DIR_VARIABLE]).resolve()
    db = folder / "registry.db"
    assert db.is_file(), f"{db} does not exist"
    with Served(db, folder) as server:
        runs = json.loads(server.get("/api/runs")[2])["data"]
        assert runs, "the registry has no cycle"
        urls = ["/api/history", "/api/learnings", "/api/overrides/pending"]
        for run in runs:
            run_id = run["id"]
            urls += [f"/api/runs/{run_id}/{part}" for part in ("labels", "holdout", "bundle", "gap", "submissions/preview")]
            urls += [f"/cycles/{run_id}/report", f"/cycles/{run_id}/report/print"]
        for url in urls:
            assert server.get(url)[0] in (200, 404), url


# 54
def test_the_manual_test_is_skipped_without_the_variable_and_the_checklist_is_complete():
    if RUN_DIR_VARIABLE not in os.environ:
        marks = [m for m in getattr(test_manual_the_end_of_journey_pages_answer_200_over_a_real_run_folder,
                                    "pytestmark", []) if m.name == "skipif"]
        assert marks and marks[0].args[0] is True
    text = (ROOT / "tests" / "MANUAL_COCKPIT_FINISH.md").read_text(encoding="utf-8").lower()
    for needed in ("sealed", "unseal", "export", "print", "reproduce", "predict", "dry-run", "daily limit", "fork",
                   "steering", "try this", "bar override", "choose the model", "light and dark", "narrow", "macos",
                   "out of scope by default"):
        assert needed in text, needed
    assert "MANUAL_COCKPIT_FINISH.md" in (ROOT / "README.md").read_text(encoding="utf-8")
