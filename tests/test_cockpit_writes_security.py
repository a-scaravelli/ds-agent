"""CSRF and Origin on every write endpoint, GET-only for the rest, the confirmation step."""

import hashlib
import re

from cockpit_live_helpers import live, live_factory, wait_for_stage  # noqa: F401  (fixtures)

from kaggle_agent.web.writes import WRITE_ENDPOINTS

GOOD_BODY = {"auto_resume": True, "action": "approve", "address": 1, "confirm": True, "reason": "x",
             "values": {"rounds": 3}, "config": {"limits": {"rounds": 3}},
             "target": {"type": "run", "id": "toy-1"}, "tags": ["a"], "text": "t"}


def cockpit_snapshot(project):
    """Every file under the project folder with its hash: a rejected write must change none of them."""
    out = {}
    for path in sorted(project.dir.rglob("*")):
        if path.is_file() and path.name not in {"heartbeat.json"}:
            out[str(path.relative_to(project.dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


# 36
def test_every_write_endpoint_refuses_missing_wrong_foreign_and_bad_host_requests(live):
    assert len(WRITE_ENDPOINTS) >= 15 and set(WRITE_ENDPOINTS) <= {r.rule for r in live.app.url_map.iter_rules()}
    before = cockpit_snapshot(live)
    port = live.port
    attempts = {
        "no token": dict(token=False),
        "wrong token": dict(token="wrong-token"),
        "foreign origin": dict(origin="http://evil.example"),
        "origin of another port": dict(origin=f"http://127.0.0.1:{port + 1}"),
        "no origin": dict(origin=False),
        "bad host": dict(host="evil.example"),
        "other host with the right origin": dict(host="192.168.1.5:8000", origin="http://192.168.1.5:8000"),
    }
    for path in WRITE_ENDPOINTS:
        for name, options in attempts.items():
            status, _headers, body = live.request("POST", path, dict(GOOD_BODY), **options)
            assert status == 403, (path, name, status, body)
    assert cockpit_snapshot(live) == before  # nothing was created or changed
    assert not (live.dir / "cockpit").exists()


# 36
def test_a_form_token_is_accepted_like_the_header_and_a_get_never_writes(live):
    before = cockpit_snapshot(live)
    for path in ("/", "/live", "/run", "/decisions", "/setup", "/environment", "/audit", "/api/run",
                 "/api/events", "/api/console", "/api/audit", "/api/notes", "/api/decisions/pending",
                 "/api/config"):
        assert live.get(path)[0] == 200, path
    assert cockpit_snapshot(live) == before  # no GET creates a file (the cockpit folder stays absent)


# 37
def test_everything_else_is_get_only_and_every_write_rule_is_guarded(live):
    app = live.app
    for rule in app.url_map.iter_rules():
        methods = (rule.methods or set()) - {"HEAD", "OPTIONS"}
        view = app.view_functions[rule.endpoint]
        if rule.endpoint.startswith("writes."):
            assert methods == {"POST"} and rule.rule in WRITE_ENDPOINTS and view.write_guarded, rule.rule
        else:
            assert methods == {"GET"} and not getattr(view, "write_guarded", False), rule.rule
            path = re.sub(r"<[^>]+>", "x", rule.rule)
            for method in ("POST", "PUT", "DELETE", "PATCH"):
                if method == "POST" and rule.rule in WRITE_ENDPOINTS:
                    continue  # a path that has a GET rule and a guarded POST rule (/api/notes)
                status, _h, _b = live.request(method, path, {}, headers={"X-CSRF-Token": live.token,
                                                                         "Origin": f"http://127.0.0.1:{live.port}"})
                assert status == 405, (method, path, status)


# 38
def test_stop_limits_and_accept_need_the_confirmation_and_leave_no_trace_without_it(live):
    block = live.tmp / "block"
    live.set_scenario({"block_file": str(block)})
    block.write_text("x", encoding="utf-8")
    live.ok("/api/run/start")
    wait_for_stage(live)
    pid = live.run_state()["pid"]
    audit_before = len(live.api("/api/audit")["entries"])
    for path, body in (("/api/run/stop", {"reason": "because"}),
                       ("/api/run/stop", {"reason": "because", "confirm": False}),
                       ("/api/run/stop", {"reason": "because", "confirm": "yes"}),
                       ("/api/limits", {"values": {"rounds": 3}, "reason": "because"}),
                       ("/api/limits", {"values": {"rounds": 3}, "reason": "because", "confirm": False}),
                       ("/api/decisions/review", {}),
                       ("/api/decisions/review", {"confirm": False})):
        status, answer = live.post(path, body)
        assert status == 400 and "confirm" in answer["error"]["message"], (path, body, answer)
    assert len(live.api("/api/audit")["entries"]) == audit_before  # no audit entry
    assert live.controller.is_alive(pid) and live.run_state()["stop_requested"] is False
    assert not (live.dir / "cockpit" / "limit-changes.jsonl").exists()
    live.ok("/api/limits", {"values": {"rounds": 3}, "reason": "because", "confirm": True})
    live.ok("/api/run/stop", {"reason": "because", "confirm": True})
    block.unlink()
    live.wait_exit(15)
