"""Live and cockpit GET endpoints: events (polling and server-sent), console, run state, pending decisions,
audit, notes, environment, jobs and the current config.

Events come from ``cockpit/events.jsonl`` (written by the orchestrator) and are read from the file, so a
held stream never holds a registry connection. A server-sent stream is bounded: a heartbeat comment every
15 s, at most 60 s of life (the browser reconnects with ``Last-Event-ID``), an immediate close when the
app's shutdown event is set, and at most 8 streams at once (the next gets 429 with ``Retry-After`` and
the page falls back to polling).

A stream slot is a lease held only while the connection is open. It is freed in a ``finally`` when the
stream ends, when the client closes (the generator probes its socket at every poll, so no heartbeat write
is needed to notice a closed peer: on macOS a write to a closed peer does not always fail), or at the
latest when the lease expires.
"""

import json
import re
import select
import threading
import time

from flask import Response, abort, current_app, jsonify, request
from kaggle_agent.proposals import ALLOWED_LIBRARIES, ValidationError
from kaggle_agent.registry import RegistryError

from . import audit, data, decisions, store
from .controller import RunController

API_VERSION = 1
DEFAULT_LIMIT, MAX_LIMIT = 200, 1000
CONSOLE_DEFAULT, CONSOLE_MAX = 8192, 65536
HISTORY_ON_CONNECT = 100
_REDACT = [
    re.compile(r"(?i)\b(bearer|token|secret|password|passwd|api[_-]?key|kaggle_key|authorization)\b(\s*[=:]\s*|\s+)\S+"),
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
]

LEASE_GRACE = 10.0
MSG_PEEK = 0x2  # same value on Linux and macOS; the web package may not import socket


class StreamSlots:
    """Per-app stream slots: token -> lease deadline. A lease past its deadline is reaped at the next acquire."""

    def __init__(self, clock=time.monotonic):
        self._lock = threading.Lock()
        self._leases: dict[int, float] = {}
        self._next = 0
        self._clock = clock

    def acquire(self, maximum: int, ttl: float):
        with self._lock:
            now = self._clock()
            for token in [t for t, deadline in self._leases.items() if deadline <= now]:
                del self._leases[token]
            if len(self._leases) >= maximum:
                return None
            self._next += 1
            self._leases[self._next] = now + ttl + LEASE_GRACE
            return self._next

    def release(self, token) -> None:
        with self._lock:
            self._leases.pop(token, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._leases)


def _peer_closed(sock) -> bool:
    """Non-blocking probe: True when the client has closed the connection. Data is peeked, never consumed."""
    if sock is None:
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            return False
        return sock.recv(1, MSG_PEEK) == b""
    except BlockingIOError:
        return False
    except (OSError, ValueError):  # BrokenPipe, ConnectionReset, EPROTOTYPE, EBADF, closed fd
        return True


def redact(text: str) -> str:
    """Token-like strings are replaced before the console output leaves the server."""
    for pattern in _REDACT:
        text = pattern.sub(lambda m: (m.group(1) + " [redacted]") if m.lastindex else "[redacted]", text)
    return text


def _ok(payload):
    return jsonify({"api_version": API_VERSION, "data": payload, "warnings": []})


def _project():
    return current_app.config["PROJECT_DIR"]


def _int_arg(name: str, default: int, low: int, high: int) -> int:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValidationError(name, "must be an integer") from None
    return max(low, min(high, value))


def events_after(project, after: int, limit: int) -> list[dict]:
    if project is None:
        return []
    records, _ = store.read_jsonl(store.cockpit_path(project, "events.jsonl"))
    return [r for r in records if isinstance(r.get("id"), int) and r["id"] > after][:limit]


def _same_origin() -> None:
    origin, host = request.headers.get("Origin"), request.headers.get("Host")
    if origin and origin not in (f"http://{host}", f"https://{host}"):
        abort(403, description="The event stream is same-origin only.")


def _sse_frame(event: dict) -> str:
    return f"id: {event['id']}\nevent: {event.get('type', 'message')}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


def register(get, source) -> None:
    @get("/events/stream")
    def event_stream():
        _same_origin()
        project = _project()
        cfg = current_app.config
        shutdown = cfg["SHUTDOWN"]
        slots = current_app.extensions["sse_slots"]
        poll, beat, lifetime = cfg["SSE_POLL"], cfg["SSE_HEARTBEAT"], cfg["SSE_MAX_SECONDS"]
        token = slots.acquire(cfg["SSE_MAX_STREAMS"], lifetime)
        if token is None:
            response = jsonify({"api_version": API_VERSION, "error": {
                "status": 429, "title": "Too many streams",
                "message": "Too many live streams are open; the page falls back to polling."}})
            response.status_code = 429
            response.headers["Retry-After"] = "2"
            return response
        try:
            sock = request.environ.get("werkzeug.socket")
            last_header = request.headers.get("Last-Event-ID") or request.args.get("after")
            try:
                after = int(last_header) if last_header not in (None, "") else None
            except ValueError:
                after = None
            if after is None:
                tail = events_after(project, 0, 10 ** 9)[-HISTORY_ON_CONNECT:]
                after = tail[0]["id"] - 1 if tail else 0

            def generate():
                cursor, started, last_send = after, time.monotonic(), time.monotonic()
                try:
                    yield "retry: 2000\n\n"
                    while not shutdown.is_set() and time.monotonic() - started < lifetime:
                        if _peer_closed(sock):
                            return
                        batch = events_after(project, cursor, 500)
                        for event in batch:
                            cursor = event["id"]
                            last_send = time.monotonic()
                            yield _sse_frame(event)
                        if not batch:
                            if time.monotonic() - last_send >= beat:
                                last_send = time.monotonic()
                                yield ": heartbeat\n\n"
                            shutdown.wait(poll)
                            if _peer_closed(sock):
                                return
                except OSError:
                    return
                finally:
                    slots.release(token)

            response = Response(generate(), mimetype="text/event-stream")
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Accel-Buffering"] = "no"
            response.call_on_close(lambda: slots.release(token))
            return response
        except BaseException:
            slots.release(token)
            raise

    @get("/events")
    def events():
        after = _int_arg("after", 0, 0, 10 ** 12)
        limit = _int_arg("limit", DEFAULT_LIMIT, 1, MAX_LIMIT)
        batch = events_after(_project(), after, limit)
        return _ok({"events": batch, "last_id": batch[-1]["id"] if batch else after})

    @get("/console")
    def console():
        project = _project()
        size = _int_arg("tail", CONSOLE_DEFAULT, 1, CONSOLE_MAX)
        raw = b"" if project is None else store.tail_bytes(store.cockpit_path(project, "console.log"), size)
        return _ok({"text": redact(raw.decode("utf-8", "replace")), "bytes": len(raw)})

    @get("/run")
    def run_state():
        project, controller = _project(), current_app.config.get("CONTROLLER")
        state = controller.state() if controller is not None else {"state": "idle", "label": "idle"}
        limits = None
        try:
            src = source()
        except RegistryError:  # the orchestrator is still creating the registry file: the state is still shown
            src = None
        if project is not None and src is not None:
            cycle_id = decisions.cycle_id_of(src)
            if cycle_id:
                try:
                    limits = data.limits_view(src, src.registry.get_cycle(cycle_id))
                except Exception:
                    limits = None
        history = [r for r in (audit.timeline(project) if project is not None else [])
                   if str(r.get("what", "")).startswith("run.") or r.get("what") in (
                       "finished", "paused", "stopped", "suspended", "error", "run_start")][-30:]
        settings = (store.read_json(store.cockpit_path(project, "settings.json"), {}) or {}) if project else {}
        return _ok({"run": state, "limits": limits, "history": history,
                    "settings": {"auto_resume": bool(settings.get("auto_resume"))},
                    "project_started": project is not None and store.exists(store.project_path(project, "project.json"))})

    @get("/decisions/pending")
    def pending():
        project = _project()
        src = source()
        question = decisions.current_question(src) if project is not None else None
        return _ok({"question": question,
                    "decisions": decisions.decision_statuses(project) if project is not None else [],
                    "conflicts": decisions.conflicts(project, question) if project is not None else [],
                    "lead_file": store.lead_toml_summary(project) if project is not None else None,
                    "notices": store.read_notices(project) if project is not None else []})

    @get("/audit")
    def audit_log():
        project = _project()
        return _ok({"entries": audit.timeline(project, _int_arg("limit", 500, 1, 5000)) if project else []})

    @get("/notes")
    def notes():
        src = source()
        out = [{"type": kind, "id": ident, **values} for (kind, ident), values in src.notes().items()]
        return _ok({"targets": out})

    @get("/environment")
    def environment_cards():
        env = current_app.config.get("ENVIRONMENT")
        force = request.args.get("refresh") == "1"
        return _ok(env.cards(force=force) if env is not None else {})

    @get("/jobs/<job_id>")
    def job(job_id):
        controller: RunController | None = current_app.config.get("CONTROLLER")
        record = controller.job(job_id) if controller is not None else None
        if record is None:
            abort(404, description="There is no such job.")
        return _ok(record)

    @get("/config")
    def config():
        project = _project()
        controller = current_app.config.get("CONTROLLER")
        path = controller.config_path if controller is not None else None
        raw = store.read_toml(path) if path is not None else None
        started = project is not None and store.exists(store.project_path(project, "project.json"))
        frozen = ("competition.slug", "competition.target", "competition.metric", "competition.train_file",
                  "competition.test_file", "protocol") if started else ()
        return _ok({"exists": raw is not None, "config": raw, "started": started, "frozen": list(frozen),
                    "allowed_libraries": list(ALLOWED_LIBRARIES)})
