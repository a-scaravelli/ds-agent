"""Helpers of the web tests of ticket 30: the Flask test client over the project of a real fake cycle.

``FinishWorld`` drives the orchestrator ``World`` (fake Claude, fake Kaggle, in-process sandbox) to STOPPED and puts the
web app over the same project folder. Its controller is a ``RunController`` whose ``_spawn`` runs the orchestrator
command LINE in this process over the same fakes (so the fake Kaggle client counts what was sent), instead of starting
a process; liveness is never reported for the fake pid. Production code has no such switch: the test only overrides
``_spawn``, the one place the controller starts something.
"""

import contextlib
import hashlib
import io
import json
import sys
import threading
from pathlib import Path

from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, REVIEW_ACCEPT, World, drive_to_stopped

from kaggle_agent.orchestrator.__main__ import main
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web import create_app
from kaggle_agent.web.controller import RunController

FAKE_PID = 4194000  # no such process: the controller never sees it alive
_RUN = threading.Lock()


class FakeProc:
    """What ``subprocess.Popen`` gives back, for a command that already ran in this process."""

    def __init__(self, code: int, output: str) -> None:
        self.pid, self.returncode, self._output = FAKE_PID, code, output.encode()

    def communicate(self, timeout=None):
        return self._output, None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class InProcessController(RunController):
    def __init__(self, world: World, **kwargs) -> None:
        super().__init__(world.project, orchestrator_argv=("ORCH",), config_path=world.config, **kwargs)
        self.world = world
        self.argvs: list[list[str]] = []

    def _spawn(self, argv, stdout=None, new_session=False):
        argv = list(argv)
        if argv[0] != "ORCH":
            return super()._spawn(argv, stdout, new_session)
        self.argvs.append(argv)
        out = io.StringIO()
        with _RUN, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(argv[1:], deps=self.world.deps)
        return FakeProc(code, out.getvalue())

    def active(self) -> bool:
        return self.lock_held()


class FinishWorld:
    def __init__(self, tmp_path, stop: bool = True, **world_kwargs) -> None:
        self.world = World(tmp_path, **world_kwargs)
        if stop:
            drive_to_stopped(self.world)
        self.project = self.world.project
        self.controller = InProcessController(self.world)
        self.app = create_app(lambda: SqliteRegistry(self.project / "registry.db", read_only=True),
                              project_dir=self.project, controller=self.controller)
        self.client = self.app.test_client()

    # -- http --------------------------------------------------------------------------------------------

    @property
    def token(self) -> str:
        return self.app.config["CSRF_TOKEN"]

    def post(self, path, body=None, token=True, origin=True, **kwargs):
        headers = {"Origin": "http://localhost" if origin is True else origin} if origin else {}
        if token is True:
            headers["X-CSRF-Token"] = self.token
        elif token:
            headers["X-CSRF-Token"] = token
        return self.client.post(path, json=body if body is not None else {}, headers=headers, **kwargs)

    def ok(self, path, body=None, status=(200, 202)):
        response = self.post(path, body)
        assert response.status_code in status, (path, response.status_code, response.get_data(as_text=True))
        return response.get_json()["data"]

    def get_json(self, path, status=200):
        response = self.client.get(path)
        assert response.status_code == status, (path, response.status_code, response.get_data(as_text=True)[:300])
        return response.get_json()["data"]

    def page(self, path, status=200) -> str:
        response = self.client.get(path)
        assert response.status_code == status, (path, response.status_code)
        return response.get_data(as_text=True)

    def wait_job(self, job: dict, timeout: float = 120.0) -> dict:
        done = self.controller.wait_job(job["id"], timeout=timeout)
        assert done is not None and done["status"] != "running", done
        return done

    # -- files and the registry ---------------------------------------------------------------------------

    def cockpit(self, name: str) -> Path:
        return self.project / "cockpit" / name

    def lines(self, name: str) -> list[dict]:
        path = self.cockpit(name)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def registry_hash(self) -> str:
        return hashlib.sha256((self.project / "registry.db").read_bytes()).hexdigest()

    def registry(self):
        return SqliteRegistry(self.project / "registry.db", read_only=True)

    def cycle(self, cycle_id: str = "toy-1"):
        reg = self.registry()
        try:
            return reg.get_cycle(cycle_id)
        finally:
            reg.close()

    def report(self, cycle_id: str = "toy-1") -> dict:
        reg = self.registry()
        try:
            return reg.get_final_report(cycle_id).content
        finally:
            reg.close()

    def attempts(self, cycle_id: str = "toy-1") -> list[str]:
        reg = self.registry()
        try:
            return [a.attempt_id for a in reg.list_attempts(cycle_id)]
        finally:
            reg.close()

    def resume(self, lead: str | None = None):
        """Run ``resume`` through the web controller (the decisions of the page are what the cycle reads)."""
        if lead is not None:
            self.world.lead(lead)
        return self.world.run("resume", "--supervised")


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def iterating_world(tmp_path, to_round: int = 1, **kwargs) -> FinishWorld:
    """A cycle waiting for the guidance after round ``to_round`` (pause_every is 1)."""
    fw = FinishWorld(tmp_path, stop=False, **kwargs)
    world = fw.world
    assert world.run("start", "--supervised")[0] == 10
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    for n in range(1, to_round):
        world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=n))
        assert world.run("resume", "--supervised")[0] == 10
    return fw


def labelled_world(tmp_path) -> FinishWorld:
    """A finished cycle with every override of ticket 30 on it: a re-read, a bar override, a steering, a seed re-run
    and a pick of the DS, all made through the page (the pick is finalized by its job)."""
    fw = FinishWorld(tmp_path)
    fw.ok("/api/holdout/unseal", {"kind": "first", "confirm": True})
    fw.ok("/api/holdout/unseal", {"kind": "second", "confirm": True, "reason": "look again"})
    fw.ok("/api/decisions/bar-override", {"scope": "round", "round": 2, "attempt": "toy-1-R2-c1",
                                          "reason": "the gain is small but real", "confirm": True})
    fw.ok("/api/decisions/steering", {"steering": {"focus": "feature_engineering", "pin_features": ["x1"]}})
    fw.ok("/api/decisions/choose-model", {"attempt": "toy-1-R1-c1", "reason": "simpler model", "confirm": True})
    fw.wait_job(fw.ok("/api/jobs/rerun", {"attempt": "toy-1-R1-c1", "seed_offset": 2})["job"])
    fw.wait_job(fw.ok("/api/jobs/finalize")["job"])
    return fw


LABELS = ("chosen by the DS", "bar overridden by the DS", "holdout re-read", "steered by the DS",
          "re-run with another seed")
