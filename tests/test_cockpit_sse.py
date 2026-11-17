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
