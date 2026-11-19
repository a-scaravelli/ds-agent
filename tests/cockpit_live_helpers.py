"""Helpers of the live cockpit tests: a project folder, the real werkzeug server on port 0 and real HTTP.

The run controller is built the only way tests can build it: through its constructor, with
``orchestrator_argv=(sys.executable, tests/fake_run_main.py, "--scenario", <json>)`` (the fakes of the
orchestrator World helpers) and ``claude_executable=<absolute path of a script written in tmp_path>``. No
``PATH`` change, no subprocess import, no process filesystem, no fixed temporary folder.
"""

import http.client
import json
import os
import signal
import socket
import stat
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web import create_app
from kaggle_agent.web.__main__ import NoRegistry, make_web_server
from kaggle_agent.web.controller import RunController

from cockpit_helpers import loopback_only  # noqa: F401  (autouse fixture)

FAKE_RUN = Path(__file__).resolve().parent / "fake_run_main.py"
TIMEOUT = 40

CONFIG_TOML = """\
[competition]
slug = "toy"
target = "from_submission_spec"
metric = "from_competition_info"

[project]
folder = "."

[limits]
rounds = 2
candidates = 1
stagnation = 2
pause_every = 1
fix_retries = 1

[protocol]
validation_params = { n_folds = 3 }

[runtime]
wait_seconds = 60
poll_interval = 5
"""

FORM = {
    "competition": {"slug": "toy", "target": "from_submission_spec", "metric": "from_competition_info"},
    "limits": {"rounds": 2, "candidates": 1, "stagnation": 2, "pause_every": 1, "fix_retries": 1},
    "protocol": {"validation_params": {"n_folds": 3}},
}


def fake_claude(tmp_path, mode: str = "in", name: str = "fake-claude") -> str:
    """An executable script that prints a scripted ``claude auth status`` answer (or hangs)."""
    body = {"in": 'print(\'{"loggedIn": true, "authMethod": "oauth"}\')',
            "out": 'print(\'{"loggedIn": false}\')',
            "garbage": "print('hello, not json')",
            "hang": "import time\ntime.sleep(60)"}[mode]
    path = Path(tmp_path) / name
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path.resolve())


class SSE:
    """A held server-sent stream read with real HTTP."""

    def __init__(self, port, path="/api/events/stream", headers=None, timeout=20):
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        self.conn.request("GET", path, headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
        self.response = self.conn.getresponse()
        self.status = self.response.status
        self.headers = dict(self.response.getheaders())

    def frames(self, count=None, stop=None):
        """Yield (kind, payload): ('event', dict), ('comment', text) until closed, count or stop(frame)."""
        seen = 0
        pending: dict = {}
        while True:
            try:
                line = self.response.fp.readline()
            except (OSError, ValueError):
                return
            if not line:
                return
            text = line.decode("utf-8").rstrip("\n")
            if text.startswith(":"):
                yield ("comment", text[1:].strip())
            elif text == "":
                if "data" in pending:
                    frame = ("event", json.loads(pending["data"]))
                    yield frame
                    seen += 1
                    if (count is not None and seen >= count) or (stop and stop(frame[1])):
                        return
                pending = {}
            elif ":" in text:
                key, _, value = text.partition(":")
                pending[key] = value.strip()

    def abort(self):
        """Reset the connection abruptly (SO_LINGER 0), as a crashed browser would."""
        try:
            sock = self.conn.sock or self.response.fp.raw._sock  # the response owns the socket once headers are read
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.response.close()
            self.conn.close()
        except OSError:
            pass

    def close(self):
        try:
            self.response.close()  # the response holds its own file object on the socket
            self.conn.close()
        except OSError:
            pass


class LiveProject:
    def __init__(self, tmp_path, scenario=None, grace=1.0, claude="in", sse=None, write_config=True,
                 controller_kwargs=None, app_kwargs=None):
        self.tmp = Path(tmp_path)
        self.dir = self.tmp / "proj"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.scenario_file = self.tmp / "scenario.json"
        self.set_scenario(scenario or {})
        if write_config:
            (self.dir / "config.toml").write_text(CONFIG_TOML, encoding="utf-8")
        self.claude = fake_claude(self.tmp, claude)
        self.controller = RunController(
            self.dir, orchestrator_argv=(sys.executable, str(FAKE_RUN), "--scenario", str(self.scenario_file)),
            claude_executable=self.claude, grace_seconds=grace, **(controller_kwargs or {}))
        self.start_server(sse=sse, **(app_kwargs or {}))

    # -- server ------------------------------------------------------------------------------

    def registry_path(self) -> Path:
        return self.dir / "registry.db"

    def _factory(self):
        self.registry_opens = getattr(self, "registry_opens", 0) + 1
        path = self.registry_path()
        return SqliteRegistry(path, read_only=True) if path.exists() else NoRegistry()

    def start_server(self, sse=None, controller=None, **kwargs):
        self.app = create_app(self._factory, project_dir=self.dir, controller=controller or self.controller,
                              sse=sse or {"SSE_POLL": 0.05, "SSE_HEARTBEAT": 0.3, "SSE_MAX_SECONDS": 3.0},
                              **kwargs)
        self.server = make_web_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop_server(self):
        started = time.monotonic()
        self.server.shutdown()
        self.thread.join(5)
        self.server.server_close()
        return time.monotonic() - started

    def set_scenario(self, scenario: dict):
        self.scenario_file.write_text(json.dumps(scenario), encoding="utf-8")

    @property
    def token(self) -> str:
        return self.app.config["CSRF_TOKEN"]

    # -- http --------------------------------------------------------------------------------

    def request(self, method, path, body=None, headers=None, token=True, origin=True, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)
        base = {"Host": host or f"127.0.0.1:{self.port}"}
        if method == "POST":
            if token is True:
                base["X-CSRF-Token"] = self.token
            elif token:
                base["X-CSRF-Token"] = token
            if origin is True:
                base["Origin"] = f"http://127.0.0.1:{self.port}"
            elif origin:
                base["Origin"] = origin
            base["Content-Type"] = "application/json"
        payload = None if body is None else json.dumps(body)
        try:
            conn.request(method, path, body=payload, headers={**base, **(headers or {})})
            response = conn.getresponse()
            raw = response.read()
        finally:
            conn.close()
        try:
            data = json.loads(raw) if raw and response.getheader("Content-Type", "").startswith("application/json") else None
        except ValueError:
            data = None
        return response.status, dict(response.getheaders()), (data if data is not None else raw.decode("utf-8", "replace"))

    def get(self, path, **kwargs):
        status, headers, data = self.request("GET", path, **kwargs)
        return status, data

    def api(self, path):
        status, data = self.get(path)
        assert status == 200, (path, status, data)
        return data["data"]

    def post(self, path, body=None, **kwargs):
        status, headers, data = self.request("POST", path, body if body is not None else {}, **kwargs)
        return status, data

    def ok(self, path, body=None):
        status, data = self.post(path, body)
        assert status in (200, 202), (path, status, data)
        return data["data"]

    # -- waiting -----------------------------------------------------------------------------

    def wait_for(self, predicate, timeout=TIMEOUT, what="the condition"):
        deadline = time.monotonic() + timeout
        value = predicate()
        while not value:
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting for {what}; console: {self.console()[-1500:]}")
            time.sleep(0.05)
            value = predicate()
        return value

    def events(self, after=0):
        return self.api(f"/api/events?after={after}&limit=1000")["events"]

    def event_types(self):
        return [e["type"] for e in self.events()]

    def run_state(self):
        return self.api("/api/run")["run"]

    def wait_state(self, *states, timeout=TIMEOUT):
        return self.wait_for(lambda: (s := self.run_state())["state"] in states and s, timeout,
                             f"state {states} (now {self.run_state()['state']})")

    def console(self) -> str:
        path = self.dir / "cockpit" / "console.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def sse(self, **kwargs) -> SSE:
        return SSE(self.port, **kwargs)

    def cockpit_file(self, name):
        return self.dir / "cockpit" / name

    def registry_state(self) -> str:
        registry = SqliteRegistry(self.registry_path(), read_only=True)
        try:
            return registry.list_cycles()[0].state
        finally:
            registry.close()

    # -- driving the cycle -------------------------------------------------------------------

    def wait_exit(self, code, timeout=TIMEOUT):
        state = self.wait_for(lambda: (s := self.run_state())["exit"] and not s["alive"] and s, timeout,
                              f"exit code {code}")
        assert state["exit"]["code"] == code, (state, self.console()[-1500:])
        return state

    def decide_plan(self, action="approve", address=1, **extra):
        return self.ok("/api/decisions/plan", {"action": action, "address": address, **extra})

    def run_to_waiting(self, state_name: str, command="resume"):
        self.ok(f"/api/run/{command}")
        self.wait_exit(10)
        self.wait_for(lambda: self.registry_state() == state_name, what=state_name)

    def cleanup(self):
        try:
            info = self.controller._run_info()
            if info and info["pgid"] != os.getpgid(0) and self.controller.is_alive(info["pid"]):  # never our own group
                try:
                    os.killpg(info["pgid"], signal.SIGKILL)
                except OSError:
                    pass
        finally:
            try:
                self.stop_server()
            except Exception:
                pass


@pytest.fixture
def live(tmp_path):
    project = LiveProject(tmp_path)
    yield project
    project.cleanup()


@pytest.fixture
def live_factory(tmp_path):
    made = []

    def make(**kwargs):
        project = LiveProject(Path(tmp_path) / f"project{len(made)}", **kwargs)  # one folder per project
        made.append(project)
        return project

    yield make
    for project in made:
        project.cleanup()


# -- in-process runs of the real orchestrator with the World fakes (no controller) ----------------

def drive_supervised(world, to="STOPPED", extra=()):
    """start -> approve -> continue after each round -> accept, every command with ``--supervised``."""
    from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, REVIEW_ACCEPT

    flag = ("--supervised", *extra)
    assert world.run("start", *flag)[0] == 10
    world.lead(PLAN_APPROVE)
    assert world.run("resume", *flag)[0] == 10
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume", *flag)[0] == 10
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    assert world.run("resume", *flag)[0] == 10
    if to == "AWAITING_REVIEW":
        return
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2) + REVIEW_ACCEPT)
    code, text = world.run("resume", *flag)
    assert code == 0 and world.state() == "STOPPED", text


def read_events(project_dir):
    from kaggle_agent.orchestrator.cockpit import read_jsonl

    return read_jsonl(Path(project_dir) / "cockpit" / "events.jsonl")[0]


# -- shared steps of the controller, decision and flow tests ----------------------------------------

def registry_summary(path):
    """What the registry holds, without ids and times: the state, attempts, rounds, guidance and calls."""
    registry = SqliteRegistry(Path(path), read_only=True)
    try:
        cycle = registry.list_cycles()[0]
        return {
            "state": cycle.state,
            "attempts": [(a.iteration, a.phase, a.outcome, a.failure_kind)
                         for a in registry.list_attempts(cycle.id)],
            "rounds": len(registry.list_rounds_with_candidates(cycle.id)),
            "candidates": [len(r.candidates) for r in registry.list_rounds_with_candidates(cycle.id)],
            "guidance": [g.text for g in registry.list_guidance(cycle.id)],
            "calls": sorted(str(c.role) for c in registry.list_llm_calls(cycle.id)),
        }
    finally:
        registry.close()


def wait_for_stage(project, after_id=0, timeout=TIMEOUT):
    """Wait until the orchestrator is inside a stage (a stage_start newer than ``after_id``)."""
    return project.wait_for(
        lambda: next((e for e in project.events(after_id) if e["type"] == "stage_start"), None), timeout,
        "a stage to start")


def finish_cycle(project, first_round=1, rounds=2):
    """From the question of ``first_round`` to the end: continue after each round, accept the review."""
    for n in range(first_round, rounds + 1):
        project.ok("/api/decisions/guidance", {"action": "continue", "address": n})
        project.ok("/api/run/resume")
        project.wait_exit(10)
    project.ok("/api/decisions/review", {"confirm": True})
    project.ok("/api/run/resume")
    project.wait_exit(0)
