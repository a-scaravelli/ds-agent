"""Fixture registry for the web tests, written only through the registry interface."""

import socket

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
from kaggle_agent.web import create_app
from kaggle_agent.registry import (
    BundleRecord,
    LlmCallEntry,
    PartitionMetric,
    SqliteRegistry,
)

SCRIPT = "<script>alert(1)</script>"
MAIN = "main"
EMPTY = "empty"
CHILD = "child"
ROUNDS = ("R1", "R2", "R3")
PIPELINE_URL = "https://example.test/p1"


def _metrics(validation=None, cv=None):
    out = []
    if validation is not None:
        out.append(PartitionMetric("validation", validation, 0.02))
    if cv is not None:
        out.append(PartitionMetric("cv", cv, 0.03))
    return out


def _succeed(reg, attempt_id, validation, cv=None, **kwargs):
    reg.add_execution(
        make_execution(attempt_id, metrics=_metrics(validation, cv), **kwargs)
    )
    reg.close_attempt(attempt_id, f"{attempt_id} concluded")


def _fail(reg, attempt_id, kind, reason):
    reg.add_execution(
        make_execution(attempt_id, outcome="failed", failure_kind=kind, reason=reason)
    )
    reg.close_attempt(attempt_id, f"{attempt_id} failed: {kind}")


def build_registry(path):
    """Write the fixture to ``path`` and close it."""
    reg = SqliteRegistry(path)
    reg.add_cycle(make_cycle(EMPTY, state="PLANNING"))
    reg.add_cycle(make_cycle(MAIN, project_id="proj-1", iterations=12, stagnation_k=4))
    reg.add_cycle(make_cycle(CHILD, state="DONE"))
    reg.add_bundle(BundleRecord("bundle-1", MAIN, "folderhash1", "bundles/b1", {"in": "csv"}))

    def attempt(attempt_id, **kwargs):
        reg.add_attempt(MAIN, make_attempt(attempt_id, **kwargs), f"fp-{attempt_id}")

    # round 1: a fix retry that succeeded, a plain success, a failure after the cap
    attempt("A1", hypothesis="trees help", rationale="because trees", what_changed="add trees",
            evidence=("ev-1", "ev-2"))
    reg.add_execution(crash("A1", 1))
    _succeed(reg, "A1", 0.70, 0.68, exec_id="A1-ok", resources={"cpu": 2}, seed=11,
             code_hash="fixed-code")
    attempt("A2", what_changed="tune depth")
    _succeed(reg, "A2", 0.75)
    attempt("A3", what_changed="broken feature")
    for n in range(1, 5):
        reg.add_execution(crash("A3", n))
    reg.close_attempt("A3", "gave up")
    # round 2
    attempt("B1", parent_id="A2", what_changed="blend", bundle_id="bundle-1",
            artifacts=("proposal-note",))
    _succeed(reg, "B1", 0.80, 0.78, artifacts=(f"pipeline:{PIPELINE_URL}",))
    attempt("B2", what_changed="huge grid")
    _fail(reg, "B2", "timeout", "ran out of time")
    # round 3
    attempt("C1", what_changed="drop feature")
    _fail(reg, "C1", "hypothesis_falsified", "the feature was useless")
    # outside every round
    attempt("X1", rationale=f"outside round {SCRIPT}", what_changed="side experiment")
    _succeed(reg, "X1", 0.60, artifacts=("pipeline:/etc/passwd",))

    add_round(reg, "R1", MAIN, ("explore models", "check leakage"), ("A1", "A2", "A3"))
    add_round(reg, "R2", MAIN, ("combine models",), ("B1", "B2"))
    add_round(reg, "R3", MAIN, ("prune",), ("C1",))

    reg.add_reviewer_report(ReviewerReport("rep-1", "R1", f"report one {SCRIPT}", T0))
    reg.add_reviewer_report(ReviewerReport("rep-2", "R2", "report two text", T0))
    reg.add_guidance(MAIN, Guidance("g1", "R1", "guidance one: focus on trees", T0))
    reg.add_suggestion(Suggestion("S2", "second suggestion text", MAIN))
    reg.link_cycle("S2", EMPTY)
    reg.link_cycle("S2", CHILD)

    call = reg.add_llm_call(
        LlmCallEntry(MAIN, "modeling", "ok", T0, 2.5, input_tokens=10, output_tokens=5,
                     role="modeling")
    )
    reg.add_role_trace(make_trace(MAIN, "coordinator", "R1", input={"goal": "start"},
                                  input_type="Goal", output_type="Round",
                                  output={"id": "R1"}))
    reg.add_role_trace(make_trace(MAIN, "modeling", "R1", candidate_id="A1", llm_call_id=call,
                                  input={"task": "trees"}, output={"id": "A1"}))
    reg.add_role_trace(make_trace(MAIN, "modeling", "R1", candidate_id="A1",
                                  input={"task": "retry"}, output={"id": "A1", "n": 2}))
    reg.add_role_trace(make_trace(MAIN, "modeling", "R2", candidate_id="B1",
                                  input={"task": "blend"}, output={"id": "B1"}))
    reg.add_role_trace(make_trace(MAIN, "reviewer", "R2", input={"look": "all"},
                                  input_type="ReviewRequest", output_type=None, output=None,
                                  outcome="invalid_output"))
    reg.add_role_trace(make_trace(MAIN, "reviewer", "R1", input={"look": "r1"},
                                  output_type="ReviewerReport", output={"text": SCRIPT}))
    reg.close()
    return path


# -- fixtures shared by the web test modules (imported by each of them) --------


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to connect a socket fails the test."""

    def refuse(*args, **kwargs):
        raise AssertionError("the web tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture
def db_path(tmp_path):
    return build_registry(tmp_path / "registry.db")


@pytest.fixture
def registry(db_path):
    with SqliteRegistry(db_path, read_only=True) as reg:
        yield reg


@pytest.fixture
def web_client(registry):
    return create_app(registry).test_client()


def all_urls():
    """One valid URL for every route of the app, built from the fixture ids."""
    return {
        "cycles": "/",
        "cycle": f"/cycles/{MAIN}",
        "round_view": f"/cycles/{MAIN}/rounds/R1",
        "metrics": f"/cycles/{MAIN}/metrics",
        "reports": f"/cycles/{MAIN}/reports",
        "traces": f"/cycles/{MAIN}/traces",
        "guidance": f"/cycles/{MAIN}/guidance",
        "attempt": "/attempts/A1",
        "suggestions": "/suggestions",
    }


def page(client, url, status=200):
    response = client.get(url)
    assert response.status_code == status, (url, response.status_code)
    return response.get_data(as_text=True)
