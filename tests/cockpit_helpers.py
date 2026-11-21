"""Helpers of the cockpit tests: a rich fixture registry, a large one, a real threaded server and
an old-format registry file. Everything is offline; only 127.0.0.1 is ever used."""

import hashlib
from html import parser as html_parser
import http.client
import json
import shutil
import socket
import sqlite3
import threading
from pathlib import Path

import pytest
from registry_helpers import (
    T0,
    add_round,
    crash,
    make_attempt,
    make_cycle,
    make_execution,
    make_trace,
)

from kaggle_agent.proposals import Guidance, ReviewerReport, Suggestion
from kaggle_agent.registry import (
    BundleRecord,
    ComparisonRecord,
    HoldoutOutcome,
    HoldoutRead,
    LlmCallEntry,
    PartitionMetric,
    ReportRecord,
    SqliteRegistry,
    SubmissionEntry,
)
from kaggle_agent.web import create_app
from kaggle_agent.web.__main__ import make_web_server

RUN_A = "run-a"
RUN_B = "run-b"
PROJECT = "proj-1"
SCRIPT = "<script>alert(1)</script>"
BUNDLE_REFERENCE = "/abs/path/<b>x</b>/bundle"
TIMEOUT = 15

PROJECT_JSON = {
    "cycle_id": RUN_A, "competition": "toy", "target": "y", "metric": "accuracy",
    "direction": "maximize", "input_kind": "labels",
    "protocol": {"metric": "accuracy", "direction": "maximize", "validation_scheme": "kfold",
                 "repetitions": 3, "threshold_k": 1.5, "seed": 42},
    "protocol_config": {}, "train_file": "train.csv", "test_file": "test.csv", "files": {},
    "version": 1,
}


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    """Any socket connection to something other than 127.0.0.1 fails the test."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost"):
            raise AssertionError(f"the cockpit tests may only use 127.0.0.1, not {host!r}")

    def connect(self, address):
        check(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ok(reg, attempt_id, mean, dispersion=0.02, **kwargs):
    reg.add_execution(make_execution(
        attempt_id, metrics=[PartitionMetric("validation", mean, dispersion)], **kwargs))
    reg.close_attempt(attempt_id, f"{attempt_id} concluded")


def _failed(reg, attempt_id, kind, reason):
    reg.add_execution(make_execution(attempt_id, outcome="failed", failure_kind=kind, reason=reason))
    reg.close_attempt(attempt_id, f"{attempt_id} failed")


def build_cockpit_registry(path, write_project_file=True):
    """Two cycles of one project; ``run-a`` is finished with holdout, report and submission.

    Returns the path. ``project.json`` is written next to the registry (unless told not to).
    """
    path = Path(path)
    reg = SqliteRegistry(path)
    reg.add_cycle(make_cycle(RUN_A, project_id=PROJECT, iterations=4, stagnation_k=2,
                             state="STOPPED"))
    reg.set_cycle_parameters(RUN_A, {"candidates": 2, "pause_every": 1,
                                     "max_consecutive_failures": 3, "max_rejections": 3})
    reg.set_protocol_hash(RUN_A, "protohash-run-a-0123456789")
    reg.add_cycle(make_cycle(RUN_B, project_id=PROJECT, iterations=5, state="ITERATING"))
    reg.add_bundle(BundleRecord("a1-b1", RUN_A, "folderhash-a1", BUNDLE_REFERENCE, {"in": "csv"}))

    def attempt(attempt_id, cycle=RUN_A, **kwargs):
        reg.add_attempt(cycle, make_attempt(attempt_id, **kwargs), f"fp-{attempt_id}")

    attempt("a-base", what_changed="majority baseline", hypothesis="baseline")
    _ok(reg, "a-base", 0.60)
    attempt("a1", parent_id="a-base", hypothesis="trees help", rationale=f"because {SCRIPT}",
            what_changed="add trees", bundle_id="a1-b1", evidence=("ev-1", "ev-2"),
            expected_effect="+0.05", risks=("overfit",), measure="accuracy")
    reg.add_execution(crash("a1", 1))
    _ok(reg, "a1", 0.70, exec_id="a1-ok", seed=11, code_hash="code-a1", data_hash="data-a1",
        resources={"library_versions": {"numpy": "2.5.4", "pandas": "3.0.6"}}, duration_seconds=3.5,
        artifacts=("pipeline:https://example.test/p1",))
    attempt("a2", what_changed="drop feature")
    _failed(reg, "a2", "hypothesis_falsified", "the feature was useless")
    attempt("a3", parent_id="a1", what_changed="tune depth")
    _ok(reg, "a3", 0.72)
    attempt("a4", what_changed="broken feature")
    for n in range(1, 5):
        reg.add_execution(crash("a4", n))
    reg.close_attempt("a4", "gave up")
    add_round(reg, "run-a-R0", RUN_A, ("baseline",), ("a-base",))
    add_round(reg, "run-a-R1", RUN_A, ("explore trees", "check leakage"), ("a1", "a2"))
    add_round(reg, "run-a-R2", RUN_A, ("tune",), ("a3", "a4"))
    reg.add_comparison(ComparisonRecord(RUN_A, "a1", "a-base", 0.10, 0.03, True))
    reg.add_comparison(ComparisonRecord(RUN_A, "a3", "a1", 0.02, 0.03, False))
    reg.add_reviewer_report(ReviewerReport(
        "run-a-R1-rep", "run-a-R1",
        f"# Round 1\n\n- trees **help**\n\n{SCRIPT}\n\n<img src=x onerror=alert(1)>\n\n"
        "[bad](javascript:alert(1))", T0))
    reg.add_guidance(RUN_A, Guidance("run-a-R1-g", "run-a-R1", "focus on trees", T0))
    reg.add_suggestion(Suggestion("S-extra", "try a linear blend", RUN_A))
    reg.link_cycle("S-extra", RUN_B)
    call = reg.add_llm_call(LlmCallEntry(RUN_A, "modeling round 1", "ok", T0, 2.5,
                                         input_tokens=100, output_tokens=50, role="modeling",
                                         model="m-1"))
    reg.add_llm_call(LlmCallEntry(RUN_A, "reviewer round 1", "invalid_output", T0, 1.0,
                                  role="reviewer", rejection_reason="bad json"))
    reg.add_llm_call(LlmCallEntry(RUN_A, "data round 1", "ok", T0, 0.5, role="data"))
    reg.add_role_trace(make_trace(RUN_A, "data", "run-a-R1", input={"q": "look"},
                                  output_type="DataFindings",
                                  output={"findings": ["no leakage found"]}))
    reg.add_role_trace(make_trace(RUN_A, "feature_engineering", "run-a-R1", input={"q": "fe"},
                                  output_type="FePlan", output={"plan": ["scale", "encode"]}))
    reg.add_role_trace(make_trace(RUN_A, "modeling", "run-a-R1", candidate_id="a1",
                                  llm_call_id=call, input={"task": "trees"},
                                  output={"id": "a1", "note": SCRIPT}))
    reg.add_role_trace(make_trace(RUN_A, "reviewer", "run-a-R1", input={"look": "r1"},
                                  output_type="ReviewerReport", output={"text": SCRIPT},
                                  outcome="ok"))
    reg.add_role_trace(make_trace(RUN_A, "reviewer", "run-a-R2", input={"look": "r2"},
                                  output_type=None, output=None, outcome="invalid_output"))
    reg.open_holdout_session(RUN_A, T0)
    reg.add_holdout_read(HoldoutRead(RUN_A, "a3", 25, T0))
    reg.close_holdout_session(RUN_A, T0)
    reg.add_holdout_outcome(HoldoutOutcome(RUN_A, "accuracy", "succeeded", T0, attempt_id="a3",
                                           bundle_id="a1-b1", score=0.69, rows=25))
    reg.add_final_report(ReportRecord(RUN_A, T0, {
        "cycle_id": RUN_A, "created_at": T0, "stop_reason": "stagnation", "metric": "accuracy",
        "direction": "maximize", "protocol_hash": "protohash-run-a-0123456789",
        "holdout": {"status": "succeeded", "score": 0.69, "rows": 25},
        "chosen": {"attempt_id": "a3", "bundle_id": "a1-b1", "hypothesis": "tune depth",
                   "validation_mean": 0.72, "validation_dispersion": 0.02, "executions": 1},
        "baseline_score": 0.6,
        "rounds": [{"round_id": "run-a-R1", "index": 1, "tasks": ["explore trees"],
                    "reviewer": "round **one** text", "improved": True, "guidance": []}],
        "worked": [{"round_id": "run-a-R1", "id": "a1"}], "failures": {"code_error": 1},
        "suggestions": [{"id": "S-extra", "text": "try a *linear* blend"}],
        "extra_field": {"kept": True},
    }))
    reg.add_submission(SubmissionEntry(RUN_A, "toy", "sub-1", "f" * 64, "first", T0, "complete",
                                       public_score=0.68, completed_at=T0))
    # the second cycle: running, nothing read
    attempt("b-base", RUN_B, what_changed="baseline b")
    _ok(reg, "b-base", 0.5)
    add_round(reg, "run-b-R0", RUN_B, ("baseline",), ("b-base",))
    reg.close()
    if write_project_file:
        (path.parent / "project.json").write_text(json.dumps(PROJECT_JSON), encoding="utf-8")
    return path


def build_large_registry(path, attempts=500, rounds=50):
    """One cycle with ``attempts`` attempts in ``rounds`` rounds, executions and role traces."""
    reg = SqliteRegistry(path)
    reg.add_cycle(make_cycle("big", iterations=rounds, state="ITERATING"))
    per_round = attempts // rounds
    n = 0
    for r in range(1, rounds + 1):
        ids = []
        for _ in range(per_round):
            n += 1
            aid = f"big-{n}"
            reg.add_attempt("big", make_attempt(aid, hypothesis=f"hypothesis {n}"), f"fp-{aid}")
            if n % 7 == 0:
                _failed(reg, aid, "timeout", "too slow")
            else:
                _ok(reg, aid, 0.5 + (n % 50) / 200, 0.01)
            ids.append(aid)
        add_round(reg, f"big-R{r}", "big", (f"task {r}",), ids)
        reg.add_role_trace(make_trace("big", "modeling", f"big-R{r}", candidate_id=ids[0],
                                      input={"round": r}, output={"id": ids[0]}))
        reg.add_llm_call(LlmCallEntry("big", f"modeling round {r}", "ok", T0, 1.0,
                                      input_tokens=10, output_tokens=5, role="modeling"))
    reg.close()
    return Path(path)


def make_old_registry(source, destination):
    """A copy of a registry file as an older version wrote it: some tables and columns missing."""
    shutil.copy(source, destination)
    db = sqlite3.connect(destination)
    try:
        for table in ("reviewer_reports", "lead_guidance", "role_traces", "bundles",
                      "holdout_outcomes", "final_reports", "rounds"):
            db.execute(f"DROP TABLE {table}")
        for table, column in (("attempts", "round_id"), ("llm_calls", "role"),
                              ("llm_calls", "rejection_reason")):
            db.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        db.commit()
    finally:
        db.close()
    return Path(destination)


class Served:
    """A real threaded werkzeug server on 127.0.0.1 with one read-only connection per request."""

    def __init__(self, db_path, project_dir=None, factory=True):
        self.db_path = Path(db_path)
        source = (lambda: SqliteRegistry(self.db_path, read_only=True)) if factory else \
            SqliteRegistry(self.db_path, read_only=True)
        self.app = create_app(source, project_dir=project_dir)
        self.server = make_web_server(self.app, "127.0.0.1", 0, threaded=True)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(TIMEOUT)

    def get(self, path, host=None, method="GET", headers=None):
        """(status, headers, body) of one request; ``host`` replaces the Host header."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)
        try:
            all_headers = dict(headers or {})
            if host is not None:
                all_headers["Host"] = host
            connection.request(method, path, headers=all_headers)
            response = connection.getresponse()
            body = response.read().decode("utf-8", "replace")
            return response.status, dict(response.getheaders()), body
        finally:
            connection.close()


def client_for(db_path, project_dir=None):
    """A Flask test client with a registry factory (one read-only connection per request)."""
    return create_app(lambda: SqliteRegistry(db_path, read_only=True),
                      project_dir=project_dir).test_client()


def page(client, url, status=200):
    response = client.get(url)
    assert response.status_code == status, (url, response.status_code)
    return response.get_data(as_text=True)


def api(client, url, status=200):
    response = client.get(url)
    assert response.status_code == status, (url, response.status_code, response.get_data(as_text=True))
    return response.get_json()


def all_page_urls(run=RUN_A):
    return [
        "/", f"/cycles/{run}", f"/cycles/{run}/rounds", f"/cycles/{run}/rounds/run-a-R1",
        f"/cycles/{run}/attempts", f"/cycles/{run}/traces", f"/cycles/{run}/traces/3",
        f"/cycles/{run}/report", f"/cycles/{run}/guidance", "/attempts/a1", "/suggestions",
        # ticket 27: the cockpit pages
        "/live", "/run", "/decisions", "/setup", "/environment", "/audit",
        # ticket 29: the performance analytics pages
        "/attempts/a1/analytics", f"/runs/{run}/compare", f"/runs/{run}/compare?sort=score&order=desc&q=trees",
        f"/runs/{run}/compare?ids=a1,a3", f"/runs/{run}/paired", f"/runs/{run}/paired?a=a1&b=a3",
    ]


def all_api_urls(run=RUN_A):
    base = f"/api/runs/{run}"
    return [
        "/api/meta", "/api/runs", base, f"{base}/chart", f"{base}/rounds",
        f"{base}/rounds/run-a-R1", f"{base}/attempts", f"{base}/executions",
        f"{base}/comparisons", f"{base}/reviewer-reports", f"{base}/guidance",
        f"{base}/submissions", f"{base}/holdout", f"{base}/final-report", f"{base}/bundles",
        f"{base}/llm-calls", f"{base}/role-traces", f"{base}/role-traces/3", "/api/attempts/a1",
        "/api/attempts/a1/executions", "/api/suggestions",
        # ticket 29: analytics
        "/api/attempts/a1/analytics", "/api/attempts/a1/folds", "/api/attempts/a1/panels", "/api/attempts/a1/segments",
        "/api/attempts/a1/hardest", "/api/attempts/a1/flags", "/api/attempts/a1/features", f"{base}/history",
        f"{base}/compare", f"{base}/ablation", f"{base}/paired?a=a1&b=a3", f"{base}/side-by-side?ids=a1,a3",
    ]


@pytest.fixture
def cockpit_db(tmp_path):
    return build_cockpit_registry(tmp_path.resolve() / "registry.db")


@pytest.fixture
def cockpit_client(cockpit_db):
    return client_for(cockpit_db, cockpit_db.parent)


class _Collector(html_parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag


def html_tags(html: str) -> list:
    """Every real start tag of ``html`` as (name, attributes); escaped text is not a tag."""
    collector = _Collector()
    collector.feed(html)
    return collector.tags


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
