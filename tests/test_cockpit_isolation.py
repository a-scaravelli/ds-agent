"""Single writer of registry.db, old projects, and the frozen packages."""

import hashlib
import os

from cockpit_live_helpers import live, live_factory  # noqa: F401  (fixtures)
from cockpit_helpers import all_page_urls, all_api_urls  # noqa: F401

from kaggle_agent.sandbox.protected import hash_path
from kaggle_agent.web.writes import WRITE_ENDPOINTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def tree(folder):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


# 41
def test_the_registry_file_is_written_only_by_the_orchestrator(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    db = live.registry_path()
    before = (digest(db), digest(db.with_name("registry.db-wal")))
    live.decide_plan(address=1)
    for rule in sorted(WRITE_ENDPOINTS):
        if rule in ("/api/run/start", "/api/run/resume"):
            continue  # these start the orchestrator, which is the one writer
        live.post(rule, {})  # most are refused; none may reach the registry
    for url in ("/", "/cycles/toy-1", "/cycles/toy-1/attempts", "/decisions", "/audit", "/live", "/api/run",
                "/api/decisions/pending", "/api/environment", "/api/runs/toy-1/attempts"):
        live.get(url)
    assert (digest(db), digest(db.with_name("registry.db-wal"))) == before
    assert not db.with_name("registry.db-journal").exists()


# 42
def test_a_project_without_the_new_files_opens_and_nothing_is_created(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    import shutil

    shutil.rmtree(live.dir / "cockpit")
    live.stop_server()
    live.controller = type(live.controller)(live.dir, orchestrator_argv=live.controller.orchestrator_argv,
                                            claude_executable=live.claude, grace_seconds=1.0)
    live.start_server()
    before = tree(live.dir)
    for url in ("/", "/live", "/decisions", "/audit", "/environment", "/setup", "/cycles/toy-1",
                "/api/run", "/api/events?after=0", "/api/audit", "/api/notes", "/api/decisions/pending",
                "/api/console", "/api/runs"):
        status, _ = live.get(url)
        assert status in (200, 404), (url, status)
    assert tree(live.dir) == before  # a GET creates nothing, not even the cockpit folder
    assert live.run_state()["state"] in ("idle", "failed", "waiting")


# 43
def test_evaluation_and_the_other_frozen_packages_keep_their_hashes():
    import test_orchestrator_guards as guards

    for name, expected in guards.PINNED.items():
        assert hash_path(guards.SRC / name) == expected, name
    assert "evaluation" in guards.PINNED
