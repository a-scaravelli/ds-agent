"""The cockpit only reads: GET routes, read-only connections, the file is never changed (no network)."""

import ast
import inspect
import sqlite3
from pathlib import Path

import pytest
from cockpit_helpers import (  # noqa: F401  (fixtures)
    RUN_A,
    all_api_urls,
    all_page_urls,
    client_for,
    cockpit_client,
    cockpit_db,
    loopback_only,
    page,
    sha256_of,
)

from kaggle_agent.registry import Cycle, Registry, RegistryError, SqliteRegistry
from kaggle_agent.web import create_app

WEB = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "web"
WRITE_PREFIXES = ("add_", "set_", "close_", "link_", "update_", "open_")
READ_PREFIXES = ("list_", "get_", "metric_series", "cycles_of_suggestion", "count_")


def _rows(path) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in sorted(tables)}
    finally:
        db.close()


# 6
def test_every_rule_accepts_only_get(cockpit_client):
    rules = list(cockpit_client.application.url_map.iter_rules())
    assert rules
    # ticket 27: only the endpoints of web/writes.py accept POST (see test_cockpit_security)
    for rule in rules:
        methods = (rule.methods or set()) - {"HEAD", "OPTIONS"}
        assert methods == ({"POST"} if rule.endpoint.startswith("writes.") else {"GET"}), rule.rule


# 6
def test_other_methods_answer_405_on_every_rule_and_leave_the_file_unchanged(cockpit_client, cockpit_db):
    before = sha256_of(cockpit_db)
    cockpit_urls = ["/live", "/run", "/decisions", "/setup", "/environment", "/audit", "/api/events",
                    "/api/events/stream", "/api/console", "/api/run", "/api/decisions/pending", "/api/audit",
                    "/api/environment", "/api/jobs/x", "/api/config",
                    # ticket 30: the routes of the final steps, memory and overrides
                    "/fork", "/learnings", "/history", "/api/history", "/api/learnings", "/api/overrides/pending",
                    f"/api/runs/{RUN_A}/labels", f"/api/runs/{RUN_A}/origin", f"/api/runs/{RUN_A}/bundle",
                    f"/api/runs/{RUN_A}/bundle/files/persisted.json", f"/api/runs/{RUN_A}/report.md",
                    f"/api/runs/{RUN_A}/gap", f"/api/runs/{RUN_A}/bar-override",
                    f"/api/runs/{RUN_A}/submissions/preview", "/api/jobs/" + "a" * 16 + "/predictions.csv",
                    f"/cycles/{RUN_A}/bundle", f"/cycles/{RUN_A}/predict", f"/cycles/{RUN_A}/submission",
                    f"/cycles/{RUN_A}/gap", f"/cycles/{RUN_A}/report/print"]
    cockpit_urls.remove("/api/events/stream")  # a held stream is covered by test_cockpit_sse
    urls = all_page_urls() + all_api_urls() + cockpit_urls
    covered = {"api.event_stream", "api.notes"}
    for url in urls:
        adapter = cockpit_client.application.url_map.bind("localhost")
        covered.add(adapter.match(url.split("?")[0], method="GET")[0])
    rules = {r.endpoint for r in cockpit_client.application.url_map.iter_rules()
             if not r.endpoint.startswith("writes.")}  # the write rules are tested in test_cockpit_writes_security
    # every rule except static and the two redirects has a url above
    assert rules - covered <= {"static", "metrics", "reports"}, rules - covered
    for url in urls + [f"/cycles/{RUN_A}/metrics", f"/cycles/{RUN_A}/reports"]:
        for method in ("post", "put", "patch", "delete"):
            response = getattr(cockpit_client, method)(url, data={"text": "x"})
            assert response.status_code == 405, (method, url)
            if url.startswith("/api/"):
                body = response.get_json()
                assert body["api_version"] == 1 and body["error"]["status"] == 405
            else:
                assert 'class="error-state"' in response.get_data(as_text=True)
    assert sha256_of(cockpit_db) == before


class SpyRegistry:
    """Delegates to a real registry and records the name of every method called."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = set()

    def __getattribute__(self, name):
        if name.startswith("_") or name in ("calls",):
            return object.__getattribute__(self, name)
        inner = object.__getattribute__(self, "_inner")
        calls = object.__getattribute__(self, "calls")
        target = getattr(inner, name)
        if callable(target):
            calls.add(name)
        return target


# 5
def test_all_pages_and_endpoints_call_only_read_methods_and_leave_the_file_unchanged(cockpit_db):
    before_hash, before_rows = sha256_of(cockpit_db), _rows(cockpit_db)
    with SqliteRegistry(cockpit_db, read_only=True) as inner:
        spy = SpyRegistry(inner)
        client = create_app(spy, project_dir=cockpit_db.parent).test_client()
        for url in all_page_urls() + all_api_urls() + ["/cycles/run-b", "/api/runs/run-b"]:
            assert client.get(url).status_code == 200, url
        assert spy.calls
        for name in spy.calls:
            assert not name.startswith(WRITE_PREFIXES), name
            assert name.startswith(READ_PREFIXES), name
    assert sha256_of(cockpit_db) == before_hash and _rows(cockpit_db) == before_rows


# 5
def test_every_request_opens_its_own_connection_with_mode_ro_and_query_only(cockpit_db, monkeypatch):
    uris, query_only = [], []
    real_connect = sqlite3.connect

    def connect(target, *args, **kwargs):
        uris.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr("kaggle_agent.registry.sqlite.sqlite3.connect", connect)

    def factory():
        reg = SqliteRegistry(cockpit_db, read_only=True)
        query_only.append(reg._db.execute("PRAGMA query_only").fetchone()[0])
        return reg

    client = create_app(factory, project_dir=cockpit_db.parent).test_client()
    urls = all_page_urls() + all_api_urls()
    for url in urls:
        assert client.get(url).status_code == 200
    assert len(uris) == len(urls) and all(u.endswith("?mode=ro") for u in uris)
    assert set(query_only) == {1}
    assert client.get("/static/css/app.css").status_code == 200
    assert len(uris) == len(urls)  # a static file opens no registry


def test_the_read_only_instance_refuses_writes(cockpit_db):
    from registry_helpers import make_cycle

    before = sha256_of(cockpit_db)
    with SqliteRegistry(cockpit_db, read_only=True) as reg:
        page(create_app(reg).test_client(), "/")
        with pytest.raises(RegistryError):
            reg.add_cycle(make_cycle("new"))
    assert sha256_of(cockpit_db) == before
    # WAL mode: a reader may leave -shm and -wal files, never a changed main file
    assert {p.name for p in cockpit_db.parent.iterdir()} <= {
        "registry.db", "registry.db-shm", "registry.db-wal", "project.json"}


def test_code_inspection_finds_no_write_call_and_no_request_body_use():
    for path in sorted(WEB.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr != "add_argument":  # argparse
                assert not node.attr.startswith(WRITE_PREFIXES), f"{path.name}: {node.attr}"
                if path.name not in ("security.py", "writes.py", "override_writes.py"):  # the CSRF guard and the write endpoints read the body
                    assert node.attr not in {"form", "json", "files", "get_json"}, path.name
            if isinstance(node, ast.ImportFrom) and node.module:
                imports_sqlite = node.module.endswith("registry.sqlite")
                assert not imports_sqlite or path.name == "__main__.py", path


def test_metric_direction_is_not_readable_through_the_registry():
    protocol_words = ("protocol", "direction")
    for name, member in inspect.getmembers(Registry, inspect.isfunction):
        if name.startswith(("get_", "list_")):
            assert not any(w in name for w in protocol_words), name
    assert not any(
        "direction" in name or "protocol" in name and name != "protocol_hash"
        for name in Cycle.__dataclass_fields__
    )
    web_text = "\n".join(p.read_text(encoding="utf-8") for p in WEB.glob("*.py"))
    assert "kaggle_agent.evaluation" not in web_text
