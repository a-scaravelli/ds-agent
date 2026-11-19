"""Local-only protections: Host check, headers, CSRF structure, and no read of bundle folders."""

import builtins
import os
import re
from pathlib import Path

import pytest
from cockpit_helpers import (  # noqa: F401  (fixtures)
    RUN_A,
    Served,
    all_api_urls,
    all_page_urls,
    client_for,
    cockpit_client,
    cockpit_db,
    html_tags,
    loopback_only,
    page,
)
from werkzeug.exceptions import Forbidden

from kaggle_agent.registry import BundleRecord, SqliteRegistry
from kaggle_agent.web import create_app, security

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; "
       "frame-ancestors 'none'")


# 23
def test_the_host_header_is_checked_on_pages_api_and_static_files(cockpit_db):
    paths = ["/", "/api/runs", "/static/css/app.css", f"/cycles/{RUN_A}", "/nothing-here"]
    with Served(cockpit_db, cockpit_db.parent) as server:
        for good in ("localhost:8000", "127.0.0.1", f"127.0.0.1:{server.port}", "LOCALHOST"):
            for path in paths[:3]:
                assert server.get(path, host=good)[0] == 200, (good, path)
        for bad in ("evil.example", "127.0.0.1.evil.example", "", "localhost.evil.example:80",
                    "[::1]", "127.0.0.1:99999x"):
            for path in paths:
                status, _, body = server.get(path, host=bad)
                assert status == 403, (bad, path, status)
        status, headers, body = server.get("/api/runs", host="evil.example")
        assert status == 403 and '"error"' in body  # JSON error under /api


def test_host_allowed_unit():
    for ok in ("localhost", "localhost:5000", "127.0.0.1", "127.0.0.1:8000"):
        assert security.host_allowed(ok)
    for bad in (None, "", "example.com", "127.0.0.1.evil.com", "localhost:abc", "0.0.0.0",
                "127.0.0.2", "a.localhost"):
        assert not security.host_allowed(bad)


# 24
def test_security_headers_and_no_inline_script_style_or_handler(cockpit_client):
    for url in all_page_urls() + all_api_urls() + ["/cycles/nope", "/api/runs/nope"]:
        response = cockpit_client.get(url)
        h = response.headers
        assert h["Content-Security-Policy"] == CSP, url
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["Referrer-Policy"] == "no-referrer"
        assert h["X-Frame-Options"] == "DENY"
        assert h["Cross-Origin-Resource-Policy"] == "same-origin"
        if url.startswith("/api/"):
            assert h["Cache-Control"] == "no-store", url
        body = response.get_data(as_text=True)
        if not url.startswith("/api/"):
            for name, attrs in html_tags(body):
                if name == "script":
                    assert "src" in attrs or attrs.get("type") == "application/json", (url, attrs)
                assert "style" not in attrs, (url, name)
                assert not [a for a in attrs if a.startswith("on")], (url, name, attrs)
                for key in ("href", "src", "action"):
                    assert not (attrs.get(key) or "").lower().startswith("javascript:"), (url, name)
    assert cockpit_client.get("/static/css/app.css").headers["Content-Security-Policy"] == CSP


# 26
def test_the_csrf_guard_rejects_bad_requests_and_accepts_a_valid_pair(cockpit_client):
    app = cockpit_client.application
    token = app.config["CSRF_TOKEN"]
    assert len(token) >= 32
    good = {"Host": "127.0.0.1:8000", "Origin": "http://127.0.0.1:8000", "X-CSRF-Token": token}

    def check(headers=None, data=None):
        with app.test_request_context("/x", method="POST", headers=headers, data=data):
            security.check_write_request()

    check(good)  # a valid pair is accepted
    check({k: v for k, v in good.items() if k != "X-CSRF-Token"}, data={"csrf_token": token})
    with pytest.raises(Forbidden):  # missing token
        check({k: v for k, v in good.items() if k != "X-CSRF-Token"})
    with pytest.raises(Forbidden):  # wrong token
        check({**good, "X-CSRF-Token": "x" * len(token)})
    with pytest.raises(Forbidden):  # foreign Origin
        check({**good, "Origin": "http://evil.example"})
    with pytest.raises(Forbidden):  # missing Origin
        check({k: v for k, v in good.items() if k != "Origin"})
    with pytest.raises(Forbidden):  # Host not local
        check({**good, "Host": "evil.example", "Origin": "http://evil.example"})

    @security.write_guard
    def endpoint():
        return "done"

    with app.test_request_context("/x", method="POST", headers=good):
        assert endpoint() == "done"
    with app.test_request_context("/x", method="POST", headers={**good, "Origin": "http://e.x"}):
        with pytest.raises(Forbidden):
            endpoint()


# 26
def test_the_shell_carries_the_csrf_meta_tag_and_no_route_uses_the_guard(cockpit_client):
    app = cockpit_client.application
    html = page(cockpit_client, "/")
    assert f'<meta name="csrf-token" content="{app.config["CSRF_TOKEN"]}">' in html
    assert html.count(app.config["CSRF_TOKEN"]) == 1  # only in the meta tag
    # ticket 27: exactly the routes of WRITE_ENDPOINTS allow POST, each wrapped in write_guard; the rest is GET only
    from kaggle_agent.web.writes import WRITE_ENDPOINTS
    posted = set()
    for rule in app.url_map.iter_rules():
        view = app.view_functions[rule.endpoint]
        methods = (rule.methods or set()) - {"HEAD", "OPTIONS"}
        if "POST" in methods:  # a path may also have a GET rule (/api/notes lists, POST writes)
            assert rule.rule in WRITE_ENDPOINTS and methods == {"POST"}, rule.rule
            assert getattr(view, "write_guarded", False), rule.rule
            posted.add(rule.rule)
        else:
            assert "check_write_request" not in view.__code__.co_names
            assert not getattr(view, "write_guarded", False) and methods == {"GET"}, rule.rule
    assert posted == set(WRITE_ENDPOINTS) and len(posted) >= 15


def test_a_new_app_gets_a_new_random_token(cockpit_db):
    a = client_for(cockpit_db).application.config["CSRF_TOKEN"]
    b = client_for(cockpit_db).application.config["CSRF_TOKEN"]
    assert a != b


# 32
def test_no_page_or_endpoint_opens_or_lists_a_bundle_folder(cockpit_db, monkeypatch):
    workspace = cockpit_db.parent / "workspace" / "bundles" / "a1-b1"
    workspace.mkdir(parents=True)
    (workspace / "sentinel.py").write_text("print('never executed')\n", encoding="utf-8")
    with SqliteRegistry(cockpit_db) as reg:
        reg.add_bundle(BundleRecord("sentinel-b", RUN_A, "hash", str(workspace), {"in": "csv"}))
    touched = []
    real_open, real_scandir, real_listdir = builtins.open, os.scandir, os.listdir

    def record(target):
        try:
            touched.append(Path(os.fsdecode(target)).resolve())
        except (TypeError, ValueError):  # a file descriptor
            pass

    def spy_open(file, *args, **kwargs):
        record(file)
        return real_open(file, *args, **kwargs)

    def spy_scandir(path=".", *args, **kwargs):
        record(path)
        return real_scandir(path, *args, **kwargs)

    def spy_listdir(path=".", *args, **kwargs):
        record(path)
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "scandir", spy_scandir)
    monkeypatch.setattr(os, "listdir", spy_listdir)
    client = client_for(cockpit_db, cockpit_db.parent)
    for url in all_page_urls() + all_api_urls() + [f"/api/runs/{RUN_A}/bundles", "/attempts/a1"]:
        assert client.get(url).status_code == 200, url
    body = client.get(f"/api/runs/{RUN_A}/bundles").get_json()["data"]
    assert {b["id"] for b in body} >= {"sentinel-b", "a1-b1"}
    assert any(b["reference"] == str(workspace) for b in body)  # shown as text only
    monkeypatch.undo()
    assert touched, "the spy must see the template and project file reads"
    root = workspace.parent.parent.resolve()
    assert not [p for p in touched if root == p or root in p.parents], touched
