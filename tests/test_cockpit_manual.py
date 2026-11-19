"""MANUAL test of the cockpit on a real run folder, and the portability checks (macOS and Linux)."""

import os
import re
from pathlib import Path

import pytest
from cockpit_helpers import Served, loopback_only  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR_VARIABLE = "KAGGLE_AGENT_RUN_COCKPIT_DIR"
COCKPIT_TESTS = sorted(ROOT.glob("tests/test_cockpit_*.py")) + [
    ROOT / "tests" / "cockpit_helpers.py", ROOT / "tests" / "test_web_readonly.py",
    ROOT / "tests" / "test_web_server.py", ROOT / "tests" / "test_web_cli.py"]
NOT_PORTABLE = ["/" + "tmp", "/" + "proc", "/" + "dev/shm", "strace ", "lsof", "xdg-open", "/" + "sys/"]


# 31 (MANUAL: skipped unless KAGGLE_AGENT_RUN_COCKPIT_DIR points to a real project folder)
@pytest.mark.skipif(RUN_DIR_VARIABLE not in os.environ, reason=f"manual: set {RUN_DIR_VARIABLE} to a project folder")
def test_manual_every_page_answers_200_over_a_real_run_folder():
    folder = Path(os.environ[RUN_DIR_VARIABLE]).resolve()
    db = folder / "registry.db"
    assert db.is_file(), f"{db} does not exist"
    with Served(db, folder) as server:
        status, _, body = server.get("/api/runs")
        assert status == 200
        import json

        runs = json.loads(body)["data"]
        assert runs, "the registry has no cycle"
        for run in runs:
            run_id = run["id"]
            urls = ["/", f"/cycles/{run_id}", f"/cycles/{run_id}/rounds", f"/cycles/{run_id}/attempts",
                    f"/cycles/{run_id}/traces", f"/cycles/{run_id}/report", f"/cycles/{run_id}/guidance",
                    "/suggestions", f"/api/runs/{run_id}", f"/api/runs/{run_id}/chart"]
            for rid in json.loads(server.get(f"/api/runs/{run_id}/rounds")[2])["data"]:
                urls.append(f"/cycles/{run_id}/rounds/{rid['id']}")
            for attempt in json.loads(server.get(f"/api/runs/{run_id}/attempts")[2])["data"]:
                urls.append(f"/attempts/{attempt['id']}")
            for url in urls:
                assert server.get(url)[0] == 200, url


# 31
def test_the_manual_test_is_skipped_by_default_and_the_readme_has_the_checklist():
    assert RUN_DIR_VARIABLE in (ROOT / "README.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Manual checklist for a real run folder" in readme and "tests/test_cockpit_manual.py" in readme


# 31
def test_the_cockpit_tests_use_resolved_paths_and_no_linux_only_tool():
    for path in COCKPIT_TESTS:
        if path == Path(__file__).resolve():  # this file holds the list of words
            continue
        text = path.read_text(encoding="utf-8")
        for word in NOT_PORTABLE:
            assert word not in text, (path.name, word)
        assert "tempfile.gettempdir" not in text, path.name
    helpers = (ROOT / "tests" / "cockpit_helpers.py").read_text(encoding="utf-8")
    assert "tmp_path.resolve()" in (ROOT / "tests" / "test_cockpit_concurrency.py").read_text(encoding="utf-8")
    assert re.search(r"tmp_path\.resolve\(\) / \"registry.db\"", helpers)
    # the macOS /var -> /private/var symlink: compare resolved paths on both sides
    for name in ("test_cockpit_api.py", "test_cockpit_pages.py"):
        assert ".resolve()" in (ROOT / "tests" / name).read_text(encoding="utf-8")
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src/kaggle_agent/web").glob("*.py"))
    for word in NOT_PORTABLE[:3]:
        assert word not in src


# 51 (MANUAL: the real orchestrator through the page, default controller commands)
@pytest.mark.skipif(RUN_DIR_VARIABLE not in os.environ, reason=f"manual: set {RUN_DIR_VARIABLE} to a project folder")
def test_manual_start_watch_decide_pause_stop_and_resume_through_the_page():
    """Needs a project folder with a valid config.toml, real Kaggle credentials and a logged-in Claude.

    Follows tests/MANUAL_COCKPIT_RUN.md: start, read the live events, approve the plan, pause, resume, stop.
    """
    import http.client
    import json
    import threading
    import time

    from kaggle_agent.web import create_app
    from kaggle_agent.web.__main__ import NoRegistry, make_web_server
    from kaggle_agent.registry import SqliteRegistry
    from kaggle_agent.web.controller import RunController

    folder = Path(os.environ[RUN_DIR_VARIABLE]).resolve()
    assert (folder / "config.toml").is_file(), "the folder needs a config.toml (use the Setup page)"
    db = folder / "registry.db"
    app = create_app(lambda: SqliteRegistry(db, read_only=True) if db.exists() else NoRegistry(),
                     project_dir=folder, controller=RunController(folder))  # default commands: the real run
    server = make_web_server(app, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port

    def call(method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        headers = {"Host": f"127.0.0.1:{port}"}
        if method == "POST":
            headers.update({"X-CSRF-Token": app.config["CSRF_TOKEN"], "Content-Type": "application/json",
                            "Origin": f"http://127.0.0.1:{port}"})
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def wait_state(*states, timeout=1800):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = call("GET", "/api/run")[1]["data"]["run"]["state"]
            if state in states:
                return state
            time.sleep(2)
        raise AssertionError(f"timed out waiting for {states}")

    try:
        assert call("POST", "/api/run/start", {})[0] in (200, 202)
        assert wait_state("waiting for you", "failed", "suspended") == "waiting for you"
        assert call("GET", "/api/events?after=0")[1]["data"]["events"], "no live events"
        call("POST", "/api/run/pause", {"reason": "manual test"})
        wait_state("paused", "waiting for you", "finished")
        status, _ = call("POST", "/api/run/resume", {})
        assert status in (200, 202, 409)
        wait_state("running", "waiting for you", "finished", "paused")
        call("POST", "/api/run/stop", {"confirm": True, "reason": "manual test"})
        wait_state("failed", "paused", "idle", "waiting for you", "finished", timeout=120)
    finally:
        server.shutdown()
        server.server_close()
