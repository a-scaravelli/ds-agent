"""Shared role helpers: retries, rejection, rate limits, traces, row mapping, wiring (cases 6-15, 21)."""

import ast
import functools
from pathlib import Path

import pytest

import kaggle_agent.agents as agents
from kaggle_agent.agents import (
    PURPOSES,
    RoleConfigurationError,
    RoleRejectedError,
    call_role,
    invoke_role,
    record_trace,
    run_data,
)
from kaggle_agent.llm import (
    CallRecord,
    FakeLlmClient,
    LlmError,
    Malformed,
    QuotaExhausted,
    QuotaExhaustedError,
    RateLimited,
    RateLimitError,
    Reply,
    Usage,
    recording,
    record_call,
)
from kaggle_agent.llm.fake import valid_attempt_json
from agents_helpers import (
    CANDIDATE_ROLES,
    ROLES,
    fixed_now,
    fresh_registry,
    context,
    data_input,
    data_json,
    permissions,
    role_input_for,
    run_role,
    settle,
    valid_reply,
)

ONE_STEP = ("coordinator", "data", "reviewer")
MARKER = "zq81-marker-p0xj"


def _traces(registry, role):
    return registry.list_role_traces("C1", role=role)


@pytest.mark.parametrize("role", ROLES)
def test_malformed_output_is_retried_then_rejected_with_a_clear_error(role, tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Malformed()] * 3)
    with pytest.raises(RoleRejectedError) as info:
        run_role(role, client, registry, tmp_path)
    error = info.value
    assert (error.role, error.attempts) == (role, 3)
    assert "this is not json" not in str(error) and "this is not json" not in error.reason
    assert error.reason and str(error.attempts) in str(error)
    assert len(client.calls) == 3
    assert "Your previous reply was rejected" not in client.calls[0].prompt
    assert all("Your previous reply was rejected" in c.prompt for c in client.calls[1:])


@pytest.mark.parametrize("role", ROLES)
def test_invalid_then_valid_uses_the_client_retry_and_traces_the_last_row(role, tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Malformed(), Reply(valid_reply(role))])
    result = run_role(role, client, registry, tmp_path)
    settle(role, result, registry)
    rows = registry.list_llm_calls("C1")
    assert [r.outcome for r in rows] == ["invalid_output", "ok"]
    assert rows[0].rejection_reason and rows[1].rejection_reason is None
    (trace,) = _traces(registry, role)
    assert trace.outcome == "ok" and trace.llm_call_id == 2 == registry.llm_call_ids[1]


@pytest.mark.parametrize("role", ROLES)
def test_rate_limit_is_handled_by_the_existing_client(role, tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([RateLimited(retry_after=3), Reply(valid_reply(role))])
    result = run_role(role, client, registry, tmp_path)
    settle(role, result, registry)
    assert client.sleeps == [3.0]
    assert [r.outcome for r in registry.list_llm_calls("C1")] == ["rate_limited", "ok"]
    (trace,) = _traces(registry, role)
    assert trace.outcome == "ok" and trace.llm_call_id == 2


@pytest.mark.parametrize("role", ROLES)
def test_exhausted_rate_limit_and_quota_are_traced_and_propagated(role, tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([RateLimited(retry_after=1)] * 4)
    with pytest.raises(RateLimitError) as info:
        run_role(role, client, registry, tmp_path)
    assert not isinstance(info.value, RoleRejectedError)
    assert len(registry.list_llm_calls("C1")) == 4
    (trace,) = _traces(registry, role)
    assert (trace.outcome, trace.output, trace.output_type) == ("rate_limited", None, None)
    assert trace.llm_call_id == 4 == registry.llm_call_ids[-1]

    registry = fresh_registry()
    with pytest.raises(QuotaExhaustedError):
        run_role(role, FakeLlmClient([QuotaExhausted()]), registry, tmp_path)
    (trace,) = _traces(registry, role)
    assert (trace.outcome, trace.output, trace.llm_call_id) == ("quota", None, 1)


@pytest.mark.parametrize("role", ONE_STEP)
def test_every_call_writes_a_trace_with_typed_input_and_output(role, tmp_path):
    registry = fresh_registry()
    result = run_role(role, FakeLlmClient([Reply(valid_reply(role))]), registry, tmp_path)
    (trace,) = _traces(registry, role)
    assert (trace.round_id, trace.outcome, trace.candidate_id) == ("R1", "ok", None)
    assert trace.input_type == type(role_input_for(role)).__name__
    assert trace.input == role_input_for(role).to_dict()
    assert trace.output_type == type(result).__name__
    assert trace.output == result.to_dict()
    assert trace.llm_call_id == registry.llm_call_ids[0] == 1
    (row,) = registry.list_llm_calls("C1", role=role)
    assert (row.role, row.outcome, row.purpose) == (role, "ok", PURPOSES[role])


@pytest.mark.parametrize("role", CANDIDATE_ROLES)
def test_candidate_trace_is_written_after_the_attempt_with_candidate_id(role, tmp_path):
    registry = fresh_registry()
    result = run_role(role, FakeLlmClient([Reply(valid_reply(role))]), registry, tmp_path)
    assert _traces(registry, role) == []  # deferred and explicit
    from registry_helpers import new_attempt

    attempt = result.output
    new_attempt(registry, attempt.id, what_changed="x")
    record_trace(registry, result.trace, candidate_id=attempt.id)
    (trace,) = _traces(registry, role)
    assert trace.candidate_id == attempt.id and trace.outcome == "ok"
    assert trace.output == attempt.to_dict() and trace.output_type == "Attempt"
    assert trace.llm_call_id == registry.llm_call_ids[-1]


@pytest.mark.parametrize("role", CANDIDATE_ROLES)
def test_rejected_candidate_role_is_traced_with_no_candidate_and_no_output(role, tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Malformed()] * 3)
    with pytest.raises(RoleRejectedError) as info:
        run_role(role, client, registry, tmp_path)
    assert not hasattr(info.value, "trace") and not hasattr(info.value, "pending")
    (trace,) = _traces(registry, role)
    assert trace.candidate_id is None and trace.output is None and trace.output_type is None
    assert trace.outcome == "invalid_output" and trace.llm_call_id == 3


@pytest.mark.parametrize("role", ROLES)
def test_rejected_call_keeps_no_raw_model_text(role, tmp_path):
    registry = fresh_registry()
    text = '{"%s": "%s"}' % (MARKER, MARKER)
    client = FakeLlmClient([Malformed(text)] * 3)
    with pytest.raises(RoleRejectedError) as info:
        run_role(role, client, registry, tmp_path)
    error = info.value
    assert MARKER not in str(error) and MARKER not in error.reason
    assert error.__cause__ is None and error.__suppress_context__
    dumped = [t.to_json() for t in _traces(registry, role)]
    dumped += [repr(r) for r in registry.list_llm_calls("C1")]
    assert dumped and all(MARKER not in d for d in dumped)


def test_llm_call_rows_are_built_by_one_shared_function(tmp_path, monkeypatch):
    seen = []
    original = recording.entry_from_record

    def counting(cycle_id, record):
        seen.append(record)
        return original(cycle_id, record)

    monkeypatch.setattr(recording, "entry_from_record", counting)
    record = CallRecord("p", "ok", "2026-10-30T10:00:00+00:00", 1.0, Usage(1, 2), 1, role="data")
    via_record = fresh_registry()
    entry = record_call(via_record, "C1", record)
    assert seen == [record]
    assert via_record.list_llm_calls("C1") == [entry]

    seen.clear()
    via_role = fresh_registry()
    client = FakeLlmClient([Reply(data_json(), Usage(1, 2))])
    run_data(client, via_role, context(), data_input(), permissions(tmp_path), now=fixed_now)
    assert seen == list(client.call_log())
    stored = via_role.list_llm_calls("C1")[0]
    assert stored == original("C1", client.call_log()[0])

    source = (Path(agents.__file__).parent / "call.py").read_text(encoding="utf-8")
    names = {
        n.func.id for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "LlmCallEntry" not in names
    import inspect

    assert list(inspect.signature(original).parameters) == ["cycle_id", "call"]


def test_a_client_already_recording_in_the_same_registry_is_refused(tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient(
        [Reply(data_json())], on_call=functools.partial(record_call, registry, "C1")
    )
    with pytest.raises(RoleConfigurationError):
        run_role("data", client, registry, tmp_path)
    assert client.calls == [] and registry.list_llm_calls("C1") == []
    assert registry.list_role_traces("C1") == []
    with pytest.raises(RoleConfigurationError):
        invoke_role(
            client, registry, context(), "data", PURPOSES["data"], "p", agents.DataFindings,
            data_input(), permissions(tmp_path),
        )
    assert client.calls == []
    assert not issubclass(RoleConfigurationError, LlmError)


def test_a_lambda_wrapper_is_not_detected_and_duplicates_rows(tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Reply(data_json())], on_call=lambda c: record_call(registry, "C1", c))
    run_role("data", client, registry, tmp_path)
    assert len(registry.list_llm_calls("C1")) == 2


def test_a_hook_for_another_registry_is_accepted(tmp_path):
    registry, other = fresh_registry(), fresh_registry()
    client = FakeLlmClient(
        [Reply(data_json())], on_call=functools.partial(record_call, other, "C1")
    )
    run_role("data", client, registry, tmp_path)
    assert len(registry.list_llm_calls("C1")) == 1 and len(other.list_llm_calls("C1")) == 1


@pytest.mark.parametrize("role", ROLES)
def test_the_model_given_by_the_caller_reaches_ask_and_the_row(role, tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Reply(valid_reply(role))])
    result = run_role(role, client, registry, tmp_path, model="test-alias")
    settle(role, result, registry)
    assert client.calls[0].model == "test-alias"
    assert registry.list_llm_calls("C1")[0].model == "test-alias"

    client = FakeLlmClient([Reply(valid_reply(role))])
    run_role(role, client, fresh_registry(), tmp_path)
    assert client.calls[0].model is None and client.call_log()[0].model is None


def test_agents_sources_contain_no_model_names_or_role_to_model_map():
    text = "\n".join(
        p.read_text(encoding="utf-8").lower()
        for p in Path(agents.__file__).parent.glob("*.py")
    )
    for word in ("claude", "opus", "sonnet", "haiku", "model_for", "role_models", "phase_model"):
        assert word not in text, word


def test_call_role_is_invoke_role_plus_record_trace(tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Reply(data_json())])
    value = call_role(
        client, registry, context(), "data", PURPOSES["data"], "p", agents.DataFindings,
        data_input(), permissions(tmp_path), now=fixed_now,
    )
    (trace,) = _traces(registry, "data")
    assert trace.output == value.to_dict() and trace.created_at == fixed_now().isoformat()
