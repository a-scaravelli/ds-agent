"""The real threaded server with several clients, a writer, and a large registry (no network)."""

import json
import threading
import time

from cockpit_helpers import (  # noqa: F401  (fixtures)
    RUN_A,
    RUN_B,
    Served,
    build_large_registry,
    cockpit_db,
    loopback_only,
    sha256_of,
)
from registry_helpers import make_attempt, make_execution

from kaggle_agent.registry import SqliteRegistry

TIMEOUT = 60
SLOW_BOUND_SECONDS = 2.0  # generous: includes the per-request connection open and its SELECT 1


def run_threads(targets):
    errors = []

    def wrap(fn):
        def inner():
            try:
                fn()
            except BaseException as exc:  # reported in the main thread
                errors.append(exc)
        return inner

    threads = [threading.Thread(target=wrap(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    assert not any(t.is_alive() for t in threads), "a thread did not finish"
    assert not errors, errors


# 1
def test_the_server_handles_requests_from_16_threads_at_once(cockpit_db):
    before = sha256_of(cockpit_db)
    urls = ["/", f"/cycles/{RUN_A}", f"/cycles/{RUN_A}/attempts", "/attempts/a1",
            f"/api/runs/{RUN_A}", f"/api/runs/{RUN_A}/attempts", f"/api/runs/{RUN_A}/chart",
            f"/cycles/{RUN_A}/traces", "/api/meta"]
    statuses = []
    lock = threading.Lock()

    with Served(cockpit_db, cockpit_db.parent) as server:
        def client(index):
            def work():
                for i in range(20):
                    status, _, body = server.get(urls[(index + i) % len(urls)])
                    with lock:
                        statuses.append(status)
            return work

        run_threads([client(n) for n in range(16)])
    assert len(statuses) == 16 * 20
    assert set(statuses) == {200}
    assert sha256_of(cockpit_db) == before


# 2
def test_pages_work_while_the_orchestrator_writes_and_the_writer_is_never_blocked(cockpit_db):
    done = threading.Event()
    written = []
    seen = []
    statuses = []
    lock = threading.Lock()

    def writer():
        reg = SqliteRegistry(cockpit_db)  # write mode: WAL on, the orchestrator's way
        try:
            assert reg._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            for n in range(1, 61):
                aid = f"w-{n}"
                reg.add_attempt(RUN_B, make_attempt(aid, hypothesis=f"written {n}"), f"fp-w-{n}")
                reg.add_execution(make_execution(aid))
                reg.close_attempt(aid, "done")
                written.append(aid)
                time.sleep(0.005)
        finally:
            reg.close()
            done.set()

    with Served(cockpit_db, cockpit_db.parent) as server:
        def reader(index):
            def work():
                last = 0
                while not done.is_set() or last < 61:
                    for url in ("/", f"/cycles/{RUN_B}", "/attempts/b-base"):
                        status, _, _ = server.get(url)
                        with lock:
                            statuses.append(status)
                    status, _, body = server.get(f"/api/runs/{RUN_B}/attempts")
                    with lock:
                        statuses.append(status)
                    count = len(json.loads(body)["data"])
                    assert count >= last, "the attempt count went down"
                    last = count
                    with lock:
                        seen.append(count)
                    if done.is_set():
                        break
            return work

        run_threads([writer] + [reader(n) for n in range(4)])
    assert len(written) == 60  # the writer finished without 'database is locked'
    assert set(statuses) == {200} and len(statuses) > 8
    assert max(seen) <= 61


# 29
def test_a_large_registry_answers_each_page_in_under_two_seconds(tmp_path):
    db = build_large_registry(tmp_path.resolve() / "big.db", attempts=500, rounds=50)
    with Served(db, db.parent) as server:
        for url in ("/cycles/big", "/cycles/big/attempts", "/api/runs/big/chart",
                    "/cycles/big/traces", "/cycles/big/rounds", "/"):
            started = time.perf_counter()
            status, _, body = server.get(url)
            elapsed = time.perf_counter() - started
            assert status == 200, url
            assert elapsed < SLOW_BOUND_SECONDS, (url, elapsed)
        status, _, body = server.get("/api/runs/big/attempts")
        assert status == 200 and len(json.loads(body)["data"]) == 500
        assert len(json.loads(server.get("/api/runs/big/chart")[2])["data"]["points"]) > 400
