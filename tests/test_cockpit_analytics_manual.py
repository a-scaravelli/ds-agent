"""MANUAL test of the analytics pages over a real run folder, and the portability checks (cases 44, 47)."""

import ast
import json
import os
import re
from pathlib import Path

import pytest
from cockpit_helpers import Served, loopback_only  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR_VARIABLE = "KAGGLE_AGENT_RUN_COCKPIT_DIR"
NEW_TESTS = sorted(ROOT.glob("tests/test_analytics_*.py")) + [ROOT / "tests" / "analytics_helpers.py",
                                                              ROOT / "tests" / "test_cockpit_analytics_manual.py"]
NOT_PORTABLE = ["/" + "tmp", "/" + "proc", "/" + "dev/shm", "strace ", "lsof", "xdg-open", "/" + "sys/", "pgrep", "systemctl"]


# 47 (MANUAL: skipped unless KAGGLE_AGENT_RUN_COCKPIT_DIR points to a real project folder)
@pytest.mark.skipif(RUN_DIR_VARIABLE not in os.environ, reason=f"manual: set {RUN_DIR_VARIABLE} to a project folder")
def test_manual_every_analytics_page_and_api_answers_200_over_a_real_run_folder():
    folder = Path(os.environ[RUN_DIR_VARIABLE]).resolve()
    db = folder / "registry.db"
    assert db.is_file(), f"{db} does not exist"
    with Served(db, folder) as server:
        runs = json.loads(server.get("/api/runs")[2])["data"]
        assert runs, "the registry has no cycle"
        for run in runs:
            run_id = run["id"]
            attempts = [a["id"] for a in json.loads(server.get(f"/api/runs/{run_id}/attempts")[2])["data"]]
            urls = [f"/runs/{run_id}/compare", f"/api/runs/{run_id}/history", f"/api/runs/{run_id}/compare",
                    f"/api/runs/{run_id}/ablation"]
            if len(attempts) >= 2:
                urls += [f"/runs/{run_id}/paired?a={attempts[0]}&b={attempts[-1]}",
                         f"/api/runs/{run_id}/paired?a={attempts[0]}&b={attempts[-1]}"]
            for attempt in attempts:
                urls.append(f"/attempts/{attempt}/analytics")
                urls += [f"/api/attempts/{attempt}/{part}" for part in
                         ("analytics", "folds", "panels", "segments", "hardest", "flags", "features")]
            for url in urls:
                assert server.get(url)[0] == 200, url


# 47
def test_the_manual_test_is_skipped_without_the_variable_and_the_checklist_is_complete():
    if RUN_DIR_VARIABLE not in os.environ:
        marks = [m for m in getattr(test_manual_every_analytics_page_and_api_answers_200_over_a_real_run_folder,
                                    "pytestmark", []) if m.name == "skipif"]
        assert marks and marks[0].args[0] is True
    text = (ROOT / "tests" / "MANUAL_COCKPIT_ANALYTICS.md").read_text(encoding="utf-8")
    for needed in ("1280", "narrow", "light", "dark", "permutation importance", "macOS", "real_process"):
        assert needed.lower() in text.lower(), needed
    assert "MANUAL_COCKPIT_ANALYTICS.md" in (ROOT / "README.md").read_text(encoding="utf-8")


# 44
def test_the_new_tests_are_portable_between_macos_and_linux():
    assert len(NEW_TESTS) >= 12
    for path in NEW_TESTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for bad in NOT_PORTABLE:
                    assert bad not in node.value or path.name == Path(__file__).name, (path.name, node.lineno, bad)
        text = path.read_text(encoding="utf-8")
        bounds = [float(x) for x in re.findall(r"time\.monotonic\(\) - \w+ < ([\d.]+)", text)]
        assert all(b >= 5 for b in bounds), (path.name, bounds)  # no timing tighter than a generous bound
        assert "mkdtemp(" not in text.replace("tmp_path_factory.mktemp(", "") or path.name == Path(__file__).name
    for path in NEW_TESTS:
        text = path.read_text(encoding="utf-8")
        if "tmp_path" in text and "tmp_path_factory" not in text and path.name != Path(__file__).name:
            # a test folder is always resolved (macOS reaches its temp folder through a symbolic link)
            assert ".resolve()" in text or "World(" in text or "finished_world" in text or "halted_world" in text, path.name
    marked = (ROOT / "tests" / "MANUAL_COCKPIT_ANALYTICS.md").read_text(encoding="utf-8")
    real = re.findall(r"^real_process = pytest\.mark\.real_process.*$", (ROOT / "tests" / "test_analytics_importance_job.py")
                      .read_text(encoding="utf-8"), re.M)
    assert real and "test_the_importance_job_runs_through_the_controller_and_the_sandbox_with_progress_events" in marked
