"""The run controller: the ONLY web module that imports ``subprocess``.

It starts, watches, pauses and stops the orchestrator as a detached child process and runs the short
background jobs of the cockpit. The rule that replaces the old "no subprocess outside llm and sandbox"
is a CLOSED ALLOWLIST, checked on this source by a test:

1. every ``subprocess.Popen`` / ``subprocess.run`` call is inside the single function ``_spawn`` and
   none passes ``shell`` (the argv is always a list or tuple);
2. ``_spawn`` is only ever given one of three argv shapes:
   ``orchestrator_argv`` + (``start`` | ``resume`` | ``validate-config`` | ``profile`` | ``preview`` | ``importance``,
   ``--config``, <path>, ``--supervised`` or ``--out``, <path> [``--attempt``, <validated id>, ``--job``, <job id>]); ``("ps", "-o", "lstart=", "-p", str(int(pid)))``;
   ``(claude_executable, "auth", "status")``;
3. ``DEFAULT_ORCHESTRATOR_ARGV`` is exactly ``(sys.executable, "-m", "kaggle_agent.orchestrator")``;
4. ``orchestrator_argv`` and ``claude_executable`` are constructor parameters only: this file reads no
   environment variable and no request, and ``web/__main__.py`` and the HTTP handlers never pass them.
   Tests reach their fakes only through these two constructor parameters.

Processes: the child gets ``start_new_session=True`` (its own session and process group, so a restart of
the web server does not kill the run), stdout and stderr go to ``cockpit/console.log``. Liveness is
``os.kill(pid, 0)`` plus the ``ps -o lstart=`` start-time string recorded in ``run.json`` (so a reused pid
is detected and never signalled); nothing reads the proc filesystem (macOS and Linux behave the same). The real
guarantee of "one run at a time" is the ``flock`` the orchestrator holds on ``run.lock``: a crash needs no
cleanup.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import store
from .codes_text import describe_code

DEFAULT_ORCHESTRATOR_ARGV = (sys.executable, "-m", "kaggle_agent.orchestrator")
RUN_COMMANDS = ("start", "resume")
JOB_COMMANDS = ("validate-config", "profile", "preview", "importance")
ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
KEEP_JOBS = 20

WAITING_TEXT = {
    "AWAITING_PLAN_APPROVAL": "your approval of the plan",
    "AWAITING_GUIDANCE": "your guidance after a round",
    "AWAITING_REVIEW": "your review of the final report",
}


class Conflict(Exception):
    """The request cannot be done now (a run is already active, nothing to pause, ...): HTTP 409."""


def _parse_time(text) -> float | None:
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


class RunController:
    """Controls the one orchestrator run of a project folder."""

    def __init__(self, project_dir, *, orchestrator_argv=DEFAULT_ORCHESTRATOR_ARGV,
                 claude_executable="claude", grace_seconds: float = 20.0, config_path=None,
                 heartbeat_stale_seconds: float = 60.0, spawn_wait: float = 15.0,
                 job_timeout: float = 120.0, claude_timeout: float = 15.0,
                 heavy_job_timeout: float = 3600.0) -> None:
        self.project_dir = Path(project_dir)
        self.orchestrator_argv = tuple(orchestrator_argv)
        self.claude_executable = str(claude_executable)
        self.grace_seconds = grace_seconds
        self.config_path = Path(config_path) if config_path else store.project_path(project_dir, "config.toml")
        self.heartbeat_stale_seconds = heartbeat_stale_seconds
        self.spawn_wait = spawn_wait
        self.job_timeout = job_timeout
        self.heavy_job_timeout = heavy_job_timeout
        self.claude_timeout = claude_timeout
        self._lock = threading.RLock()
        self._children: dict[int, subprocess.Popen] = {}

    # -- the one function that spawns ---------------------------------------------------------

    def _spawn(self, argv, stdout=None, new_session: bool = False):
        """Start a process from an argv list (never a shell); the closed list is in the module docstring."""
        return subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=stdout,
                                stderr=subprocess.STDOUT if stdout is not None else subprocess.DEVNULL,
                                start_new_session=new_session)

    # -- helpers on processes --------------------------------------------------------------------

    def process_start_time(self, pid) -> str | None:
        """The ``ps -o lstart=`` string of a process (same on macOS and Linux), None if it is gone."""
        try:
            proc = self._spawn(("ps", "-o", "lstart=", "-p", str(int(pid))), stdout=subprocess.PIPE)
            out, _ = proc.communicate(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
        text = out.decode("utf-8", "replace").strip()
        return text or None

    @staticmethod
    def is_alive(pid) -> bool:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OverflowError, ValueError):
            return True
        return True

    def _run_info(self) -> dict | None:
        info = store.read_json(store.cockpit_path(self.project_dir, "run.json"))
        return info if isinstance(info, dict) and isinstance(info.get("pid"), int) else None

    def _verified(self, info: dict | None) -> bool:
        """The recorded process is alive AND is the same process (start time matches)."""
        if info is None or not self.is_alive(info["pid"]):
            return False
        for attempt in range(3):  # `ps` can fail or come back empty under load: ask again before giving up
            found = self.process_start_time(info["pid"])
            if found is not None:
                return found == info.get("start_time")
            if not self.is_alive(info["pid"]):
                return False
            time.sleep(0.1)
        return False

    def lock_held(self) -> bool:
        return store.lock_is_held(store.cockpit_path(self.project_dir, "run.lock"))

    def active(self) -> bool:
        return self.lock_held() or self._verified(self._run_info())

    # -- the state shown to the DS ---------------------------------------------------------------

    def _events_tail(self) -> list[dict]:
        records, _ = store.read_jsonl(store.cockpit_path(self.project_dir, "events.jsonl"))
        return records[-400:]

    @staticmethod
    def _current_stage(events: list[dict]) -> str | None:
        open_stages: list[str] = []
        for event in events:
            data = event.get("data") or {}
            if event.get("type") == "stage_start":
                open_stages.append(data.get("role") or data.get("stage"))
            elif event.get("type") == "stage_end" and (data.get("role") or data.get("stage")) in open_stages:
                open_stages.remove(data.get("role") or data.get("stage"))
            elif event.get("type") == "run_start":
                open_stages = []
        return open_stages[-1] if open_stages else None

    def heartbeat_age(self) -> float | None:
        beat = store.read_json(store.cockpit_path(self.project_dir, "heartbeat.json"))
        stamp = _parse_time(beat.get("at")) if isinstance(beat, dict) else None
        return None if stamp is None else max(0.0, time.time() - stamp)

    def state(self) -> dict:
        """The state of the run, derived from files and liveness only (it creates nothing)."""
        info = self._run_info()
        events = self._events_tail()
        last_run = store.read_json(store.project_path(self.project_dir, "last-run.json"), {}) or {}
        out = {"state": "idle", "label": "idle", "text": "Nothing is running.", "stage": None, "waiting_for": None,
               "pid": None, "alive": False, "responsive": True, "heartbeat_age": None, "exit": None,
               "suspended_kind": None, "command": None, "started_at": None, "stop_requested": False,
               "pause_requested": store.exists(store.cockpit_path(self.project_dir, "pause-request.json"))}
        alive = self._verified(info)
        held = self.lock_held()
        if info is not None:
            out.update(pid=info["pid"], command=info.get("command"), started_at=info.get("started_at"),
                       stop_requested=bool(info.get("stop_requested")))
        if alive or held:
            stage = self._current_stage(events)
            age = self.heartbeat_age()
            out.update(alive=True, state="running", stage=stage, heartbeat_age=age)
            out["label"] = f"running ({stage})" if stage else "running"
            out["text"] = "The run is working." + (f" Now: {stage}." if stage else "")
            if age is not None and age > self.heartbeat_stale_seconds:
                out.update(responsive=False, label="not responding",
                           text="The run process is alive but its heartbeat is old: it may be stuck.")
            if out["stop_requested"]:
                out.update(state="stopping", label="stopping", text="The run is being stopped.")
            return out
        exit_info = store.read_json(store.cockpit_path(self.project_dir, "exit.json"))
        if isinstance(exit_info, dict) and info is not None and exit_info.get("pid") != info["pid"]:
            exit_info = None
        if info is not None and not isinstance(exit_info, dict):
            out.update(state="failed", label="failed",
                       text="The run process disappeared without a final status (it was killed or the machine "
                            "stopped). The registry is intact: press Resume.")
            return out
        if isinstance(exit_info, dict):
            code = exit_info.get("code")
            state, text = describe_code(code)
            out.update(state=state, label=state, text=text, exit={"code": code, "at": exit_info.get("at"),
                                                                  "meaning": exit_info.get("meaning")})
            self._refine(out, events, last_run)
            return out
        outcome = last_run.get("outcome")  # a run started from a terminal: last-run.json only
        if outcome in ("finished", "waiting", "suspended", "paused"):
            out.update(state=outcome, label=outcome, text=f"The last run (from a terminal) ended: {outcome}.")
            self._refine(out, events, last_run)
        return out

    @staticmethod
    def _refine(out: dict, events: list[dict], last_run: dict) -> None:
        if out["state"] == "waiting":
            what = WAITING_TEXT.get(last_run.get("state"))
            out["waiting_for"] = what
            out["label"] = f"waiting for you ({what})" if what else "waiting for you"
        elif out["state"] == "suspended":
            kinds = [e["data"].get("kind") for e in events if e.get("type") == "suspended"]
            kind = kinds[-1] if kinds else ("login" if "login" in str(last_run.get("reason", "")).lower() else "quota")
            out["suspended_kind"] = kind
            out["label"] = f"suspended ({kind})"
            if kind == "login":
                out["text"] += " Run `claude auth login` in a normal Terminal, then press Resume."

    # -- start, pause, stop -------------------------------------------------------------------------

    def launch(self, command: str) -> dict:
        """Start ``start`` or ``resume`` as a detached child; a second request gets ``Conflict``."""
        if command not in RUN_COMMANDS:
            raise ValueError(command)
        with self._lock, store.FileLock(store.cockpit_path(self.project_dir, "controller.lock")):
            if self.active():
                raise Conflict("a run is already active in this project; no second run is started")
            store.remove(store.cockpit_path(self.project_dir, "pause-request.json"))
            console = store.console_handle(store.cockpit_path(self.project_dir, "console.log"))
            try:
                proc = self._spawn((*self.orchestrator_argv, command, "--config", str(self.config_path),
                                    "--supervised"), stdout=console, new_session=True)
            finally:
                console.close()
            start_time = None
            for _ in range(5):  # `ps` can fail or come back empty under load: ask again
                start_time = self.process_start_time(proc.pid)
                if start_time is not None or not self.is_alive(proc.pid):
                    break
                time.sleep(0.1)
            store.write_json(store.cockpit_path(self.project_dir, "run.json"), {
                "pid": proc.pid, "pgid": proc.pid, "start_time": start_time, "command": command,
                "config": str(self.config_path), "started_at": store.utc_text(), "stop_requested": False,
                "who": store.os_user()})
            self._children[proc.pid] = proc
            threading.Thread(target=self._reap, args=(proc,), daemon=True, name="reaper").start()
            deadline = time.monotonic() + self.spawn_wait
            while time.monotonic() < deadline and proc.poll() is None and not self.lock_held():
                time.sleep(0.05)
        return {"pid": proc.pid, "command": command}

    def _reap(self, proc) -> None:
        code = proc.wait()
        self._children.pop(proc.pid, None)
        if code != 14:  # a pause request that was still pending at a different exit is stale
            store.remove(store.cockpit_path(self.project_dir, "pause-request.json"))

    def request_pause(self) -> None:
        with self._lock:
            if not self.active():
                raise Conflict("nothing is running, so there is nothing to pause")
            store.write_json(store.cockpit_path(self.project_dir, "pause-request.json"),
                             {"at": store.utc_text(), "who": store.os_user()})

    def stop(self) -> dict:
        """SIGTERM to the process group, SIGKILL after the grace period (in a background thread)."""
        with self._lock:
            info = self._run_info()
            if not self._verified(info):
                raise Conflict("no run started by this cockpit is alive (or its pid was reused), so nothing is signalled")
            store.write_json(store.cockpit_path(self.project_dir, "run.json"), {**info, "stop_requested": True})
            try:
                os.killpg(int(info["pgid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass
            threading.Thread(target=self._kill_later, args=(info,), daemon=True, name="killer").start()
        return {"pid": info["pid"], "grace_seconds": self.grace_seconds}

    def _kill_later(self, info: dict) -> None:
        deadline = time.monotonic() + self.grace_seconds
        while time.monotonic() < deadline:
            if not self.is_alive(info["pid"]):  # a failed `ps` under load must not cancel the kill
                return
            time.sleep(0.05)
        # still alive after the grace period: confirm it is the same process (ps can fail under load, so ask again)
        if any(self._verified(info) for _ in range(5)):
            try:
                os.killpg(int(info["pgid"]), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def resume_if_idle(self) -> bool:
        """After a decision: resume unless a run is active. True if a process was started."""
        try:
            self.launch("resume")
        except Conflict:
            return False
        return True

    # -- background jobs ---------------------------------------------------------------------------

    def start_job(self, kind: str, config_path=None, attempt: str | None = None) -> dict:
        """Run ``validate-config``, ``profile``, ``preview`` or ``importance`` in the background.

        ``importance`` needs a validated ``attempt`` id and uses the longer heavy-job timeout.
        """
        if kind not in JOB_COMMANDS:
            raise ValueError(kind)
        if kind == "importance" and not (isinstance(attempt, str) and ATTEMPT_ID.fullmatch(attempt)):
            raise ValueError("bad attempt id")
        job_id = store.new_id()[:16]
        record_path, out_path = store.job_paths(self.project_dir, job_id)
        record = {"id": job_id, "kind": kind, "status": "running", "started_at": store.utc_text(),
                  "finished_at": None, "error": None}
        if kind == "importance":
            record["attempt"] = attempt
        store.write_json(record_path, record)
        self._trim_jobs()
        cfg = Path(config_path) if config_path else self.config_path
        threading.Thread(target=self._job, args=(kind, cfg, record, record_path, out_path, attempt),
                         daemon=True, name=f"job-{kind}").start()
        return record

    def _job(self, kind, cfg, record, record_path, out_path, attempt=None) -> None:
        status, error = "done", None
        limit = self.heavy_job_timeout if kind == "importance" else self.job_timeout
        try:
            job_id = record["id"]
            if kind == "importance":  # attempt and job_id were validated by regex before this thread started
                proc = self._spawn((*self.orchestrator_argv, kind, "--config", str(cfg), "--attempt", attempt,
                                    "--out", str(out_path), "--job", job_id), stdout=subprocess.PIPE)
            else:
                proc = self._spawn((*self.orchestrator_argv, kind, "--config", str(cfg), "--out", str(out_path)),
                                   stdout=subprocess.PIPE)
            try:
                proc.communicate(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                status, error = "timeout", f"the job did not finish within {limit:g} seconds"
            else:
                if not store.exists(out_path):
                    status, error = "failed", f"the job exited with code {proc.returncode} and no result"
                elif proc.returncode not in (0, None):
                    result = store.read_json(out_path, {}) or {}
                    status, error = "failed", result.get("error") or f"the job exited with code {proc.returncode}"
        except OSError as exc:
            status, error = "failed", f"cannot start the job: {exc}"
        store.write_json(record_path, {**record, "status": status, "error": error,
                                       "finished_at": store.utc_text()})

    def job(self, job_id: str) -> dict | None:
        try:
            record_path, out_path = store.job_paths(self.project_dir, job_id)
        except ValueError:
            return None
        record = store.read_json(record_path)
        if not isinstance(record, dict):
            return None
        result = store.read_json(out_path) if record.get("status") in ("done", "failed") else None
        out = {**record, "result": result}
        if record.get("kind") == "importance":
            out["progress"] = store.job_progress(self.project_dir, job_id)
        return out

    def wait_job(self, job_id: str, timeout: float | None = None) -> dict | None:
        deadline = time.monotonic() + (max(self.job_timeout, self.heavy_job_timeout) + 5 if timeout is None else timeout)
        while time.monotonic() < deadline:
            record = self.job(job_id)
            if record is not None and record["status"] != "running":
                return record
            time.sleep(0.05)
        return self.job(job_id)

    def _trim_jobs(self) -> None:
        paths = store.list_jobs(self.project_dir)
        paths.sort(key=lambda p: store.mtime(p) or 0)
        for path in paths[:-KEEP_JOBS]:
            store.remove(path)
            store.remove(path.with_name(path.stem + ".out.json"))

    # -- Claude login check ------------------------------------------------------------------------

    def claude_status(self) -> dict:
        """Run ``claude auth status`` and parse ``loggedIn``; anything unexpected is ``unknown``."""
        try:
            proc = self._spawn((self.claude_executable, "auth", "status"), stdout=subprocess.PIPE)
            try:
                out, _ = proc.communicate(timeout=self.claude_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return {"state": "unknown", "detail": "`claude auth status` did not answer in time"}
        except OSError as exc:
            return {"state": "unknown", "detail": f"cannot run `claude`: {exc.strerror or exc}"}
        try:
            data = json.loads(out.decode("utf-8", "replace"))
            logged = data["loggedIn"]
        except (ValueError, KeyError, TypeError):
            return {"state": "unknown", "detail": "the answer of `claude auth status` could not be read"}
        if not isinstance(logged, bool):
            return {"state": "unknown", "detail": "the answer of `claude auth status` could not be read"}
        return {"state": "logged_in" if logged else "logged_out", "detail": None}
