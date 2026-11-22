"""Permutation importance as a background job through the controller and the sandbox (cases 31, 32, 33).

The real-process tests (marked ``real_process``) start child processes: run them also on macOS (see
MANUAL_COCKPIT_ANALYTICS.md).
"""

import hashlib
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest
from analytics_helpers import (  # noqa: F401  (fixtures)
    attempts_of,
    cycle_of,
    failure_world,
    halted_world,
)
from cockpit_helpers import loopback_only  # noqa: F401  (autouse fixture)

from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web import create_app, store
from kaggle_agent.web.controller import RunController

real_process = pytest.mark.real_process

X2_MODEL = '''\
import json
import os


def fit(train):
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "m.json"), "w") as handle:
        json.dump({"ok": True}, handle)


def predict(bundle_id, X):
    return [1 if float(v) == 0 else 0 for v in X["x2"]]
'''
ORIGIN = {"Origin": "http://localhost"}


def app_for(world, controller):
    app = create_app(lambda: SqliteRegistry(str(world.project / "registry.db"), read_only=True),
                     project_dir=world.project, controller=controller)
    return app, app.test_client()


def post(app, client, body, token=True, headers=None):
    head = {**ORIGIN, **(headers or {})}
    if token:
        head["X-CSRF-Token"] = app.config["CSRF_TOKEN"]
    return client.post("/api/jobs/importance", json=body, headers=head)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_done(controller, job_id, seconds=240):
    record = controller.wait_job(job_id, timeout=seconds)
    assert record is not None and record["status"] != "running", record
    return record


class SpyController(RunController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.argvs = []

    def _spawn(self, argv, stdout=None, new_session=False):
        self.argvs.append(tuple(argv))
        return super()._spawn(argv, stdout, new_session)


GATE = textwrap.dedent('''\
    import json, os, sys, time
    argv = sys.argv[1:]
    out = argv[argv.index("--out") + 1]
    job = argv[argv.index("--job") + 1]
    config = argv[argv.index("--config") + 1]
    gate = os.environ.get("GATE_FILE") or os.path.join(os.path.dirname(config), "gate")
    folder = os.path.join(os.path.dirname(config), "project", "cockpit", "jobs")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, job + ".progress.jsonl"), "a") as handle:
        handle.write(json.dumps({"type": "job_start", "total": 4}) + "\\n")
        handle.write(json.dumps({"type": "job_progress", "done": 2, "total": 4, "feature": "x2", "fold": "r0/f0"}) + "\\n")
    while not os.path.exists(gate):
        time.sleep(0.05)
    with open(out, "w") as handle:
        json.dump({"ok": True}, handle)
''')


@pytest.fixture(scope="module")
def importance_world(tmp_path_factory):
    return halted_world(tmp_path_factory.mktemp("importance"), model=lambda n, b: X2_MODEL, real_sandbox=True)


@pytest.fixture
def gate_script(tmp_path):
    path = tmp_path.resolve() / "gate_job.py"
    path.write_text(GATE, encoding="utf-8")
    return path


# 31
@real_process
def test_the_importance_job_runs_through_the_controller_and_the_sandbox_with_progress_events(importance_world):
    world = importance_world
    db = world.project / "registry.db"
    before = sha(db)
    controller = SpyController(world.project, config_path=world.config, heavy_job_timeout=600)
    app, client = app_for(world, controller)
    attempt = attempts_of(world)[-1]
    response = post(app, client, {"attempt": attempt})
    assert response.status_code == 202, response.get_data(as_text=True)
    job = response.get_json()["data"]
    assert job["kind"] == "importance" and job["attempt"] == attempt
    record = wait_done(controller, job["id"])
    assert record["status"] == "done", record
    assert record["result"]["ok"] is True and record["result"]["features"] == 3
    events, _ = store.read_jsonl(world.project / "cockpit" / "jobs" / f"{job['id']}.progress.jsonl")
    kinds = [e["type"] for e in events]
    assert kinds[0] == "job_start" and kinds[-1] == "job_end" and events[-1]["status"] == "done"
    done = [e["done"] for e in events if e["type"] == "job_progress"]
    assert done == sorted(done) and done[-1] == done[0] + len(done) - 1 == events[0]["total"]
    assert all(e["total"] == events[0]["total"] and e["feature"] and e["fold"] for e in events if e["type"] == "job_progress")
    bundle = f"{attempt}-b1"
    result = json.loads((world.project / "analytics" / "importance" / f"{bundle}.json").read_text(encoding="utf-8"))
    ranked = [r["feature"] for r in result["features"]]
    assert ranked[0] == "x2" and result["features"][0]["importance"] > 0.2
    unused = {r["feature"]: r["importance"] for r in result["features"]}
    assert unused["x1"] == pytest.approx(0.0) and unused["id"] == pytest.approx(0.0)
    assert not list((world.project / "analytics" / "importance").glob(".*"))  # atomic: no temp file is left
    assert not (world.project / "cockpit" / "events.jsonl").exists()  # the event log of a run is not touched
    assert sha(db) == before
    argv = controller.argvs[-1]
    assert argv[:3] == (sys.executable, "-m", "kaggle_agent.orchestrator")
    assert argv[3:] == ("importance", "--config", str(world.config), "--attempt", attempt,
                        "--out", argv[9], "--job", job["id"])
    # the job status carries the progress, and the features page shows the permutation importance
    status = client.get(f"/api/jobs/{job['id']}").get_json()["data"]
    assert status["progress"]["percent"] == 100.0 and status["progress"]["last"]["type"] == "job_end"
    features = client.get(f"/api/attempts/{attempt}/features").get_json()["data"]
    assert features["permutation"]["features"][0]["feature"] == "x2"
    assert features["permutation"]["groups"][0]["family"] == "x2"
    audit, _ = store.read_jsonl(world.project / "cockpit" / "audit.jsonl")
    assert audit[-1]["what"] == "job.importance" and audit[-1]["payload"] == {"attempt": attempt}


# 32
def test_a_running_job_never_blocks_the_pages_and_shows_its_progress(importance_world, gate_script):
    world = importance_world
    gate = world.base / "gate"
    gate.unlink(missing_ok=True)
    controller = RunController(world.project, config_path=world.config, orchestrator_argv=(sys.executable, str(gate_script)),
                               heavy_job_timeout=120)
    app, client = app_for(world, controller)
    attempt = attempts_of(world)[-1]
    job = post(app, client, {"attempt": attempt}).get_json()["data"]
    try:
        for url in (f"/attempts/{attempt}/analytics", f"/runs/{cycle_of(world)}/compare", f"/attempts/{attempt}"):
            started = time.monotonic()
            assert client.get(url).status_code == 200, url
            assert time.monotonic() - started < 10, url  # a generous bound: the job is still running
        deadline = time.monotonic() + 20
        status = client.get(f"/api/jobs/{job['id']}").get_json()["data"]
        while status["progress"]["percent"] is None and time.monotonic() < deadline:
            time.sleep(0.1)
            status = client.get(f"/api/jobs/{job['id']}").get_json()["data"]
        assert status["status"] == "running" and status["progress"]["percent"] == 50.0
        assert status["progress"]["last"]["feature"] == "x2"
    finally:
        gate.write_text("go", encoding="utf-8")
    assert wait_done(controller, job["id"])["status"] == "done"
    gate.unlink()


# 33
def test_bad_requests_start_no_process_and_a_missing_token_changes_nothing(importance_world, failure_world, tmp_path):
    world = importance_world
    controller = SpyController(world.project, config_path=world.config)
    app, client = app_for(world, controller)
    good = attempts_of(world)[-1]
    audit_path = world.project / "cockpit" / "audit.jsonl"
    audit_before = audit_path.read_bytes() if audit_path.exists() else b""
    assert post(app, client, {"attempt": "toy-1-R9-c9"}).status_code == 404
    for bad in ("../../etc/passwd", "a/b", "x y", "", None, 5):
        assert post(app, client, {"attempt": bad}).status_code == 400, bad
    assert post(app, client, {}).status_code == 400
    denied = post(app, client, {"attempt": good}, token=False)
    assert denied.status_code == 403
    assert client.post("/api/jobs/importance", json={"attempt": good}).status_code == 403  # no Origin either
    assert controller.argvs == []
    assert (audit_path.read_bytes() if audit_path.exists() else b"") == audit_before
    # a project whose attempt has no succeeded run has no recorded bundle: refused with the reason
    failed = attempts_of(failure_world)[1]
    app2, client2 = app_for(failure_world, SpyController(failure_world.project, config_path=failure_world.config))
    refused = post(app2, client2, {"attempt": failed})
    assert refused.status_code == 400 and "failed" in refused.get_json()["error"]["message"]
    assert app2.config["CONTROLLER"].argvs == []


# 33
def test_a_job_over_its_timeout_ends_as_timeout_and_the_argv_is_the_closed_list(importance_world, gate_script):
    world = importance_world
    gate = world.base / "gate"
    gate.unlink(missing_ok=True)
    controller = SpyController(world.project, config_path=world.config, orchestrator_argv=(sys.executable, str(gate_script)),
                               heavy_job_timeout=1.0)
    app, client = app_for(world, controller)
    attempt = attempts_of(world)[-1]
    job = post(app, client, {"attempt": attempt}).get_json()["data"]
    record = wait_done(controller, job["id"], 60)
    assert record["status"] == "timeout" and "did not finish" in record["error"]
    argv = controller.argvs[-1]
    assert argv[2:4] == ("importance", "--config") and argv[argv.index("--attempt") + 1] == attempt
    assert set(argv[4:]) - {str(world.config), attempt, argv[argv.index("--out") + 1], job["id"]} == {
        "--attempt", "--out", "--job"}
    with pytest.raises(ValueError):
        controller.start_job("importance", attempt="../x")
    with pytest.raises(ValueError):
        controller.start_job("importance")
    with pytest.raises(ValueError):
        controller.start_job("rm")
