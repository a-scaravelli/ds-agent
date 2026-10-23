"""Template method of LlmClient: validation, retries, backoff, consumption (cases 10, 12-20, 23)."""

import pytest

from kaggle_agent.llm import (
    BackoffPolicy,
    CliCrash,
    CliFailureError,
    FakeLlmClient,
    InvalidResponseError,
    LlmClient,
    LlmError,
    Malformed,
    QuotaExhausted,
    QuotaExhaustedError,
    RateLimited,
    RateLimitError,
    RawResponse,
    Reply,
    Usage,
)
from kaggle_agent.proposals import Attempt, ShortPlan
from llm_helpers import ATTEMPT_JSON, PLAN_JSON, permissions


def test_valid_plan_and_attempt_are_validated_with_the_existing_types(tmp_path):
    perms = permissions(tmp_path)
    plan = FakeLlmClient([Reply(PLAN_JSON)]).ask("make a plan", ShortPlan, perms, "plan")
    assert plan.value == ShortPlan.from_json(PLAN_JSON) and plan.consumption.calls == 1
    attempt = FakeLlmClient([Reply(ATTEMPT_JSON)]).ask("go", Attempt, perms, "attempt")
    assert attempt.value == Attempt.from_json(ATTEMPT_JSON) and len(attempt.calls) == 1


def test_prompt_carries_the_json_contract(tmp_path):
    client = FakeLlmClient([Reply(PLAN_JSON)])
    client.ask("make a plan", ShortPlan, permissions(tmp_path), "plan")
    assert client.calls[0].prompt.startswith("make a plan") and "JSON object" in client.calls[0].prompt


def test_fenced_json_needs_no_retry(tmp_path):
    client = FakeLlmClient([Reply(f"```json\n{PLAN_JSON}\n```")])
    result = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert result.consumption.calls == 1 and len(client.calls) == 1


def test_malformed_reply_is_retried_with_the_reason(tmp_path):
    client = FakeLlmClient([Malformed("not json"), Reply(PLAN_JSON)])
    result = client.ask("original prompt", ShortPlan, permissions(tmp_path), "plan")
    second = client.calls[1].prompt
    assert "original prompt" in second and "rejected" in second and "invalid JSON" in second
    assert len(result.calls) == 2 and isinstance(result.value, ShortPlan)
    assert [c.outcome for c in result.calls] == ["invalid_output", "ok"]


def test_parseable_but_invalid_reply_passes_the_validation_error_back(tmp_path):
    import json

    bad = json.loads(ATTEMPT_JSON)
    del bad["what_changed"]
    client = FakeLlmClient([Reply(json.dumps(bad)), Reply(ATTEMPT_JSON)])
    result = client.ask("p", Attempt, permissions(tmp_path), "attempt")
    assert "what_changed: is required" in client.calls[1].prompt
    assert isinstance(result.value, Attempt)


def test_bounded_validation_retries_then_invalid_response_error(tmp_path):
    client = FakeLlmClient([Reply("{}")] * 5, max_validation_retries=2)
    with pytest.raises(InvalidResponseError) as exc:
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert exc.value.attempts == 3 and exc.value.last_error
    assert len(client.calls) == 3 and isinstance(exc.value, LlmError)


def test_rate_limit_backoff_uses_the_injected_sleep(tmp_path):
    client = FakeLlmClient(
        [RateLimited(), RateLimited(), Reply(PLAN_JSON)],
        backoff=BackoffPolicy(max_retries=3, base=1, factor=2, max_delay=100),
    )
    result = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert client.sleeps == [1, 2]
    assert [c.outcome for c in result.calls] == ["rate_limited", "rate_limited", "ok"]
    assert result.consumption.rate_limit_events == 2


def test_retry_after_is_respected_and_capped(tmp_path):
    policy = BackoffPolicy(max_retries=3, base=1, factor=2, max_delay=10)
    capped = FakeLlmClient([RateLimited(retry_after=30), Reply(PLAN_JSON)], backoff=policy)
    capped.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert capped.sleeps == [10]
    short = FakeLlmClient([RateLimited(retry_after=3), RateLimited(), Reply(PLAN_JSON)], backoff=policy)
    short.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert short.sleeps == [3, 2]


def test_persistent_rate_limit_raises_after_bounded_attempts(tmp_path):
    client = FakeLlmClient(
        [RateLimited()] * 10, backoff=BackoffPolicy(max_retries=3, base=1, factor=2, max_delay=100)
    )
    with pytest.raises(RateLimitError):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert len(client.calls) == 4 and len(client.sleeps) == 3


def test_quota_is_raised_at_once_without_sleeping(tmp_path):
    client = FakeLlmClient([QuotaExhausted(resets_at="2026-01-01T05:00:00+00:00"), Reply(PLAN_JSON)])
    with pytest.raises(QuotaExhaustedError) as exc:
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert exc.value.resets_at == "2026-01-01T05:00:00+00:00"
    assert client.sleeps == [] and len(client.calls) == 1
    assert [c.outcome for c in client.call_log()] == ["quota"]
    assert not isinstance(exc.value, RateLimitError)


def test_backoff_policy_rejects_invalid_values():
    for kwargs in ({"max_retries": -1}, {"base": -1}, {"factor": 0.5}, {"max_delay": -1}):
        with pytest.raises(ValueError):
            BackoffPolicy(**kwargs)


def test_unexpected_exceptions_from_run_become_cli_failures(tmp_path):
    class Broken(LlmClient):
        def _run(self, prompt, permissions, model):
            raise KeyError("secret detail")

    with pytest.raises(CliFailureError) as exc:
        Broken().ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert "secret detail" not in str(exc.value)


def test_consumption_reports_calls_tokens_and_duration_with_failures(tmp_path):
    client = FakeLlmClient(
        [
            Reply(PLAN_JSON, usage=Usage(10, 5), duration=2),
            Reply(PLAN_JSON, duration=3),
            CliCrash(duration=1),
        ]
    )
    first = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    second = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    with pytest.raises(CliFailureError):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert [c.duration_seconds for c in (first.calls[0], second.calls[0])] == [2, 3]
    assert (first.calls[0].usage.input_tokens, second.calls[0].usage.input_tokens) == (10, None)
    total = client.consumption()
    assert (total.calls, total.input_tokens, total.output_tokens) == (3, 10, 5)
    assert total.duration_seconds == 6


def test_model_recorded_is_reported_then_requested_then_none(tmp_path):
    client = FakeLlmClient(
        [Reply(PLAN_JSON, model="reported"), Reply(PLAN_JSON), Reply(PLAN_JSON)]
    )
    perms = permissions(tmp_path)
    a = client.ask("p", ShortPlan, perms, "plan", model="asked")  # reported wins
    b = client.ask("p", ShortPlan, perms, "plan", model="asked")
    c = client.ask("p", ShortPlan, perms, "plan")
    assert [r.calls[0].model for r in (a, b, c)] == ["reported", "asked", None]


def test_validation_retries_keep_the_requested_model(tmp_path):
    client = FakeLlmClient([Malformed(), Reply(PLAN_JSON)], model="client-model")
    result = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert [c.model for c in client.calls] == ["client-model", "client-model"]
    assert [c.model for c in result.calls] == ["client-model", "client-model"]
    override = FakeLlmClient([Reply(PLAN_JSON)], model="client-model")
    override.ask("p", ShortPlan, permissions(tmp_path), "plan", model="call-model")
    assert override.calls[0].model == "call-model"


def test_client_does_not_enforce_a_budget(tmp_path):
    client = FakeLlmClient([Reply(PLAN_JSON)] * 50)
    for _ in range(50):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert client.consumption().calls == 50


def test_raw_response_has_no_secret_fields():
    assert set(RawResponse.__dataclass_fields__) == {
        "text", "usage", "session_id", "model", "stripped_variables"
    }
