"""Consumption stored in the registry through the additive llm_calls table (cases 5, 11, 24)."""

import functools

import pytest

from kaggle_agent.llm import (
    CliCrash,
    CliFailureError,
    FakeLlmClient,
    Malformed,
    RateLimited,
    Reply,
    Usage,
    record_call,
)
from kaggle_agent.proposals import ShortPlan
from kaggle_agent.registry import SqliteRegistry
from llm_helpers import PLAN_JSON, SECRET_KEY, SECRET_TOKEN, permissions
from registry_helpers import make_cycle


def _registry(path=":memory:"):
    registry = SqliteRegistry(path)
    registry.add_cycle(make_cycle("C1"))
    return registry


def test_calls_are_stored_in_order_with_outcome_duration_and_tokens(tmp_path):
    registry = _registry()
    client = FakeLlmClient(
        [RateLimited(duration=1), Malformed(duration=2), Reply(PLAN_JSON, Usage(10, 5), 3)],
        on_call=functools.partial(record_call, registry, "C1"),
    )
    client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    rows = registry.list_llm_calls("C1")
    assert [r.outcome for r in rows] == ["rate_limited", "invalid_output", "ok"]
    assert [r.duration_seconds for r in rows] == [1, 2, 3]
    assert (rows[2].input_tokens, rows[2].output_tokens) == (10, 5)
    assert [r.attempt for r in rows] == [1, 2, 3] and all(r.purpose == "plan" for r in rows)
    columns = {c[1] for c in registry._db.execute("PRAGMA table_info(llm_calls)")}
    assert not columns & {"prompt", "response", "text", "token", "access_token"}


def test_model_is_stored_per_call_and_survives_reopening(tmp_path):
    path = str(tmp_path / "r.db")
    registry = _registry(path)
    client = FakeLlmClient(
        [Reply(PLAN_JSON, model="reported"), Reply(PLAN_JSON), Reply(PLAN_JSON)],
        on_call=functools.partial(record_call, registry, "C1"),
    )
    client.ask("p", ShortPlan, permissions(tmp_path), "plan", model="asked")
    client.ask("p", ShortPlan, permissions(tmp_path), "plan", model="asked")
    client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    before = registry.list_llm_calls("C1")
    assert [r.model for r in before] == ["reported", "asked", None]
    registry.close()
    reopened = SqliteRegistry(path)
    assert reopened.list_llm_calls("C1") == before


def test_token_never_reaches_registry_rows(tmp_path):
    registry = _registry()
    client = FakeLlmClient(
        [CliCrash(), Reply(PLAN_JSON)], on_call=functools.partial(record_call, registry, "C1")
    )
    with pytest.raises(CliFailureError):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    dump = repr(registry.list_llm_calls("C1"))
    assert SECRET_TOKEN not in dump and SECRET_KEY not in dump
    assert [r.outcome for r in registry.list_llm_calls("C1")] == ["cli_error", "ok"]


def test_unknown_cycle_is_rejected(tmp_path):
    registry = _registry()
    client = FakeLlmClient([Reply(PLAN_JSON)], on_call=functools.partial(record_call, registry, "NOPE"))
    with pytest.raises(Exception):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")


def test_rejection_reason_is_stored_and_survives_reopening(tmp_path):
    path = str(tmp_path / "r.db")
    registry = _registry(path)
    client = FakeLlmClient(
        [Malformed("nope"), Reply(PLAN_JSON)],
        on_call=functools.partial(record_call, registry, "C1"),
    )
    client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    before = registry.list_llm_calls("C1")
    assert before[0].rejection_reason.startswith("plan: invalid JSON") and before[1].rejection_reason is None
    registry.close()
    assert SqliteRegistry(path).list_llm_calls("C1") == before
