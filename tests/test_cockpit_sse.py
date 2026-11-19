"""Server-sent events, polling fallback, console and the live UI (real server, real HTTP)."""

import re
import threading
import time
from pathlib import Path

from cockpit_live_helpers import live, live_factory  # noqa: F401  (fixtures)

from kaggle_agent.orchestrator.events import Emitter, EventLog

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "src" / "kaggle_agent" / "web" / "static" / "js"


def emitter(project) -> Emitter:
    return Emitter(EventLog(project.cockpit_file("events.jsonl")), "toy-1")


def ids(frames):
    return [payload["id"] for kind, payload in frames if kind == "event"]


# 5
def test_sse_delivers_live_events_and_resumes_with_last_event_id(live_factory):
    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 5, "SSE_MAX_SECONDS": 20})
    emit = emitter(project)
    for n in range(3):
        emit.emit("stage_start", stage=f"s{n}")
    first = project.sse()
    assert first.status == 200 and first.headers["Content-Type"].startswith("text/event-stream")
    assert first.headers["Cache-Control"] == "no-store"
    seen = []
    threading.Timer(0.3, lambda: [emit.emit("stage_end", stage=f"s{n}") for n in range(2)]).start()
    for kind, payload in first.frames(count=5):
        seen.append((kind, payload))
    first.close()
    assert ids(seen) == [1, 2, 3, 4, 5]  # three old, two live
    for n in range(2):
        emit.emit("finished", reason=f"r{n}")  # while the client is away
    again = project.sse(headers={"Last-Event-ID": "4"})
    got = ids(list(again.frames(count=3)))
    again.close()
    assert got == [5, 6, 7]  # exactly n+1 onward: no loss, no duplicate


# 6
def test_sse_heartbeat_and_bounded_lifetime(live_factory):
    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 0.2, "SSE_MAX_SECONDS": 1.0})
    stream = project.sse()
    started = time.monotonic()
    frames = list(stream.frames())  # ends by itself: the server closes the stream
    stream.close()
    assert 0.8 <= time.monotonic() - started < 4
    beats = [p for kind, p in frames if kind == "comment" and p == "heartbeat"]
    assert len(beats) >= 2


# 7
def test_many_sse_clients_do_not_exhaust_workers_and_hold_no_registry_connection(live_factory):
    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 0.3, "SSE_MAX_SECONDS": 30, "SSE_MAX_STREAMS": 8})
    opened_before = getattr(project, "registry_opens", 0)
    streams = [project.sse() for _ in range(20)]
    try:
        statuses = [s.status for s in streams]
        assert statuses.count(200) == 8 and statuses.count(429) == 12
        assert all(s.headers.get("Retry-After") for s in streams if s.status == 429)
        assert getattr(project, "registry_opens", 0) == opened_before  # a stream never opens a registry
        started = time.monotonic()
        status, body = project.get("/")
        assert status == 200 and "<h1>Runs</h1>" in body
        assert project.get("/api/runs")[0] == 200 and project.get("/api/events?after=0")[0] == 200
        assert time.monotonic() - started < 10
    finally:
        for s in streams:
            s.close()
    # a closed stream frees its slot
    for _ in range(100):  # a dead peer is noticed at the next heartbeat write
        again = project.sse()
        if again.status == 200:
            again.close()
            break
        again.close()
        time.sleep(0.1)
    else:
        raise AssertionError("the slots were not released")


# 8
def test_the_server_stops_cleanly_with_held_sse_connections(live_factory):
    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 5, "SSE_MAX_SECONDS": 60})
    streams = [project.sse() for _ in range(3)]
    results = []
    readers = [threading.Thread(target=lambda s=s: results.append(list(s.frames())), daemon=True) for s in streams]
    for reader in readers:
        reader.start()
    time.sleep(0.3)
    elapsed = project.stop_server()
    assert elapsed < 3 and not project.thread.is_alive()
    for reader in readers:
        reader.join(3)
        assert not reader.is_alive()  # every stream was closed by the server
    assert len(results) == 3
    for s in streams:
        s.close()


# 9
def test_the_polling_endpoint_returns_events_in_order_and_is_not_cached(live):
    emit = emitter(live)
    for n in range(30):
        emit.emit("stage_start", stage=f"s{n}")
    status, headers, body = live.request("GET", "/api/events?after=10&limit=5")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert [e["id"] for e in body["data"]["events"]] == [11, 12, 13, 14, 15] and body["data"]["last_id"] == 15
    last = live.api("/api/events?after=25&limit=5")
    assert [e["id"] for e in last["events"]] == [26, 27, 28, 29, 30] and last["last_id"] == 30
    assert live.api("/api/events?after=30") == {"events": [], "last_id": 30}
    assert live.get("/api/events?after=x")[0] == 400


# 10
def test_the_console_returns_the_bounded_redacted_tail(live):
    path = live.cockpit_file("console.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old line\n" * 2000 + "last line with password=hunter2hunter2\n", encoding="utf-8")
    body = live.api("/api/console?tail=2048")
    assert body["bytes"] <= 2048 and "last line" in body["text"] and "hunter2" not in body["text"]
    assert live.api("/api/console?tail=999999999")["bytes"] <= 65536
    assert live.api("/api/console")["text"].count("old line") > 100


# 11
def test_the_live_ui_has_ticker_timeline_progress_calls_console_and_opt_in_notifications(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    page = live.get("/live")[1]
    for needle in ('id="live-ticker"', 'data-stream-url="/api/events/stream"', 'id="timeline"', 'id="progress-list"',
                   'id="call-log"', "Tokens in", "Tokens out", 'id="console"', 'id="enable-notifications"',
                   'id="notify-sound"'):
        assert needle in page, needle
    assert "waiting for you (your approval of the plan)" in live.get("/run")[1]
    chip = live.get("/")[1]
    assert 'id="run-chip"' in chip
    notify = (STATIC / "notify.js").read_text(encoding="utf-8")
    assert notify.count("Notification.requestPermission") == 1
    at = notify.index("Notification.requestPermission")
    handler = notify.rfind('addEventListener("click"', 0, at)
    assert handler != -1 and "function" in notify[handler:at]
    assert not re.search(r"^\s*Notification\.requestPermission", notify, re.M)  # never at load
    assert 'localStorage.getItem("notify-sound")' in notify  # the sound is optional
    script = (STATIC / "live.js").read_text(encoding="utf-8")
    assert "new EventSource" in script and "/events" not in script  # the URLs come from the page
    assert "setInterval(poll" in script  # the polling fallback
    for name in ("live.js", "notify.js", "cockpit.js"):
        assert f"js/{name}" in page


def _accepts_stream(project, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        again = project.sse()
        status = again.status
        again.close()
        if status == 200:
            return True
        time.sleep(0.1)
    return False


def _wait_free(project, seconds=5):
    slots = project.app.extensions["sse_slots"]
    deadline = time.monotonic() + seconds
    while len(slots) and time.monotonic() < deadline:
        time.sleep(0.05)
    return len(slots) == 0


# 2
def test_the_slot_is_released_on_every_exit_path(live_factory, monkeypatch):
    from kaggle_agent.web import live as live_module

    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 5, "SSE_MAX_SECONDS": 30, "SSE_MAX_STREAMS": 2})
    for _ in range(5):  # closed without reading the body
        assert _accepts_stream(project)
    real = live_module.events_after
    calls = []

    def flaky(*args, **kwargs):
        if not calls:
            calls.append(1)
            raise RuntimeError("forced")
        return real(*args, **kwargs)

    monkeypatch.setattr(live_module, "events_after", flaky)
    assert _wait_free(project)
    assert project.sse().status == 500  # the forced error between acquire and the response
    monkeypatch.setattr(live_module, "events_after", real)
    assert _wait_free(project)
    first, second = project.sse(), project.sse()
    try:
        assert (first.status, second.status) == (200, 200)
    finally:
        first.close()
        second.close()


# 3
def test_a_closed_peer_is_detected_on_an_idle_stream_without_a_heartbeat_write(live_factory):
    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 60, "SSE_MAX_SECONDS": 60, "SSE_MAX_STREAMS": 8})
    streams = [project.sse() for _ in range(8)]
    assert [s.status for s in streams] == [200] * 8
    assert project.sse().status == 429
    for s in streams:
        s.abort()
    assert _accepts_stream(project, 2)


class FakeSock:
    def __init__(self, recv=b"", error=None):
        self._recv, self._error = recv, error
        self.peeked = []

    def fileno(self):
        return 99

    def recv(self, n, flags=0):
        self.peeked.append(flags)
        if self._error:
            raise self._error
        return self._recv


# 4
def test_the_peer_probe_handles_the_macos_errors_and_runs_in_every_iteration(monkeypatch, live_factory):
    import select as select_module

    from kaggle_agent.web import live as live_module

    def readable(socks, *_):
        return (socks, [], [])

    monkeypatch.setattr(live_module.select, "select", readable)
    probe = live_module._peer_closed
    assert probe(FakeSock(b"")) is True
    assert probe(FakeSock(error=OSError(41, "EPROTOTYPE"))) is True
    assert probe(FakeSock(error=OSError(9, "EBADF"))) is True
    assert probe(FakeSock(error=ValueError("closed fd"))) is True
    assert probe(FakeSock(error=BlockingIOError())) is False
    data = FakeSock(b"G")
    assert probe(data) is False and data.peeked == [0x2]  # peeked, never consumed
    monkeypatch.setattr(live_module.select, "select", lambda *a: ([], [], []))
    assert probe(FakeSock(b"")) is False
    assert probe(None) is False
    monkeypatch.setattr(live_module.select, "select", select_module.select)

    # the probe runs in an iteration with no batch and no heartbeat due
    calls = []
    monkeypatch.setattr(live_module, "_peer_closed", lambda sock: calls.append(1) or len(calls) >= 3)
    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 60, "SSE_MAX_SECONDS": 30})
    stream = project.sse()
    try:
        assert list(stream.frames()) == []  # the fake says closed at the third probe: the stream ends
    finally:
        stream.close()
    assert len(calls) >= 3


# 4 (write failure)
def test_a_failing_write_with_errno_41_ends_the_generator_cleanly(live_factory, monkeypatch):
    from kaggle_agent.web import live as live_module

    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 0.05, "SSE_MAX_SECONDS": 30, "SSE_MAX_STREAMS": 1})
    slots = project.app.extensions["sse_slots"]
    calls = []

    def frame(event):
        calls.append(1)
        raise OSError(41, "EPROTOTYPE")

    monkeypatch.setattr(live_module, "_sse_frame", frame)
    emitter(project).emit("stage_start", stage="x")
    stream = project.sse()
    try:
        for _ in range(100):
            if calls and len(slots) == 0:
                break
            time.sleep(0.05)
        assert calls and len(slots) == 0
    finally:
        stream.close()


# 5
def test_slots_are_per_app_and_an_expired_lease_is_reaped(live_factory):
    from kaggle_agent.web import live as live_module

    now = [0.0]
    slots = live_module.StreamSlots(clock=lambda: now[0])
    assert slots.acquire(2, 60) == 1 and slots.acquire(2, 5) == 2
    assert slots.acquire(2, 60) is None
    now[0] = 14.0  # lease 2 (5 s + 10 s grace) is not expired yet at 14.0
    assert slots.acquire(2, 60) is None
    now[0] = 15.5
    third = slots.acquire(2, 60)
    assert third is not None and len(slots) == 2  # the expired one was reaped, the live one kept
    slots.release(third)
    slots.release(third)  # idempotent
    assert len(slots) == 1

    one = live_factory(sse={"SSE_MAX_STREAMS": 1})
    other = live_factory(sse={"SSE_MAX_STREAMS": 1})
    held = one.sse()
    try:
        assert held.status == 200
        again = other.sse()
        assert again.status == 200
        again.close()
        assert one.app.extensions["sse_slots"] is not other.app.extensions["sse_slots"]
    finally:
        held.close()


# 6
def test_thirty_opens_and_closes_still_deliver_an_event_each_time(live_factory):
    project = live_factory(sse={"SSE_POLL": 0.05, "SSE_HEARTBEAT": 5, "SSE_MAX_SECONDS": 30, "SSE_MAX_STREAMS": 8})
    emitter(project).emit("stage_start", stage="s")
    for _ in range(30):
        assert _wait_free(project)
        stream = project.sse()
        try:
            assert stream.status == 200
            assert ids(stream.frames(count=1))
        finally:
            stream.close()
