"""RealClaudeClient with a fake runner only: argv, env, mapping, contract (cases 5, 6, 8, 21, 28, 29)."""

import json
import logging
import subprocess

import pytest

from kaggle_agent.llm import (
    BackoffPolicy,
    CliFailureError,
    FakeLlmClient,
    LlmError,
    QuotaExhaustedError,
    RateLimitError,
    Reply,
    Malformed,
    RateLimited,
    Usage,
)
from kaggle_agent.llm.real import RealClaudeClient
from kaggle_agent.proposals import ShortPlan
from llm_helpers import (
    ATTEMPT_JSON,
    PLAN_JSON,
    SECRET_KEY,
    SECRET_TOKEN,
    Clock,
    FakeRunner,
    completed,
    envelope,
    ok_step,
    permissions,
)

ENV = {"PATH": "/bin", "CLAUDE_CODE_OAUTH_TOKEN": SECRET_TOKEN}


def _client(steps, env=None, **kwargs):
    clock = Clock()
    runner = FakeRunner(steps, clock=clock)
    client = RealClaudeClient(
        runner=runner, env=ENV if env is None else env, clock=clock.now, sleep=clock.sleep, **kwargs
    )
    return client, runner, clock


def test_argv_cwd_stdin_and_minimal_tools(tmp_path):
    client, runner, _ = _client([ok_step(PLAN_JSON)])
    client.ask("the prompt", ShortPlan, permissions(tmp_path, "Read"), "plan")
    call = runner.calls[0]
    argv = call.argv
    assert argv[0] == "claude" and "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--tools") + 1] == "Read"
    assert argv[argv.index("--allowedTools") + 1] == "Read"
    assert call.cwd == str(tmp_path) and call.input.startswith("the prompt")
    assert "the prompt" not in " ".join(argv)
    joined = " ".join(argv)
    assert "dangerously" not in joined and "bypassPermissions" not in joined
    assert "--bare" not in argv


def test_default_permissions_disable_every_tool(tmp_path):
    client, runner, _ = _client([ok_step(PLAN_JSON)])
    client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    argv = runner.calls[0].argv
    assert argv[argv.index("--tools") + 1] == "" and "--allowedTools" not in argv


def test_argv_never_holds_tools_beyond_the_given_ones(tmp_path):
    client, _, _ = _client([])
    argv = client.build_argv(permissions(tmp_path, "Read", "Grep"), None)
    allowed = argv[argv.index("--allowedTools") + 1 :]
    assert allowed == ["Read", "Grep"]
    assert "Bash" not in argv and "Edit" not in argv and "Write" not in argv


def test_model_flag_only_when_set_and_call_override_wins(tmp_path):
    plain, runner, _ = _client([ok_step(PLAN_JSON)])
    plain.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert "--model" not in runner.calls[0].argv
    configured, runner, _ = _client([ok_step(PLAN_JSON), ok_step(PLAN_JSON)], model="m-client")
    configured.ask("p", ShortPlan, permissions(tmp_path), "plan")
    configured.ask("p", ShortPlan, permissions(tmp_path), "plan", model="m-call")
    first, second = (c.argv for c in runner.calls)
    assert first[first.index("--model") + 1] == "m-client"
    assert second[second.index("--model") + 1] == "m-call" and "m-client" not in second


def test_token_is_only_in_the_cli_environment(tmp_path, caplog):
    registry_rows = []
    leak = f"auth failed with {SECRET_TOKEN} and {SECRET_KEY}"
    steps = [
        completed(stdout="", stderr=leak, returncode=1),
        ok_step(PLAN_JSON),
    ]
    client, runner, _ = _client(steps, env={**ENV, "ANTHROPIC_API_KEY": SECRET_KEY},
                                on_call=registry_rows.append)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(CliFailureError) as exc:
            client.ask("p", ShortPlan, permissions(tmp_path), "plan")
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    surfaces = [repr(exc.value), str(exc.value), caplog.text, repr(registry_rows), repr(client.call_log())]
    for call in runner.calls:
        surfaces += [str(call.argv), call.input]
    for text in surfaces:
        assert SECRET_TOKEN not in text and SECRET_KEY not in text
    assert runner.calls[0].env["CLAUDE_CODE_OAUTH_TOKEN"] == SECRET_TOKEN


@pytest.mark.parametrize(
    "step",
    [
        completed(stdout="RAW-OUT", stderr="RAW-ERR", returncode=1),
        FileNotFoundError("claude: RAW-PATH"),
        subprocess.TimeoutExpired(cmd=["claude"], timeout=5),
        completed(stdout="RAW-NOT-JSON", returncode=0),
        completed(stdout="[1]", returncode=0),
        PermissionError("RAW-PERM"),
    ],
)
def test_cli_failures_become_sanitized_cli_failure_errors(tmp_path, step):
    client, _, _ = _client([step])
    with pytest.raises(CliFailureError) as exc:
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert "RAW" not in str(exc.value)
    assert [c.outcome for c in client.call_log()] == ["cli_error"]


def test_envelope_success_maps_usage_session_and_reported_model(tmp_path):
    step = ok_step(
        PLAN_JSON,
        model_usage={"other": {"inputTokens": 1, "outputTokens": 1},
                     "model-x": {"inputTokens": 50, "outputTokens": 20}},
    )
    client, _, _ = _client([step])
    result = client.ask("p", ShortPlan, permissions(tmp_path), "plan", model="asked")
    call = result.calls[0]
    assert (call.usage.input_tokens, call.usage.output_tokens) == (12, 5)
    assert call.model == "model-x"
    raw = client._interpret(step, {}, ())
    assert raw.session_id == "11111111-2222-3333-4444-555555555555" and raw.model == "model-x"


def test_envelope_without_usage_leaves_tokens_unknown(tmp_path):
    client, _, _ = _client([completed(json.dumps({"is_error": False, "result": PLAN_JSON}))])
    result = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert result.calls[0].usage == Usage() and result.calls[0].model is None


def test_error_envelope_not_logged_in_and_generic(tmp_path):
    not_logged = envelope(result="Not logged in · Please run /login", is_error=True,
                          subtype="success", terminal_reason="api_error")
    client, _, _ = _client([completed(not_logged, returncode=1)])
    with pytest.raises(CliFailureError, match="not authenticated"):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    generic = envelope(result="Something odd RAW", is_error=True, terminal_reason="api_error")
    client, _, _ = _client([completed(generic, returncode=1)])
    with pytest.raises(CliFailureError) as exc:
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert "RAW" not in str(exc.value) and "api_error" in str(exc.value)


def test_rate_limit_and_quota_envelopes(tmp_path):
    limited = completed(
        envelope(result="API Error: 429 rate limit, retry after 7 seconds", is_error=True,
                 api_error_status=429), returncode=1)
    client, _, clock = _client([limited, ok_step(PLAN_JSON)],
                               backoff=BackoffPolicy(max_retries=1, base=1, factor=2, max_delay=60))
    result = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert clock.sleeps == [7] and result.calls[0].outcome == "rate_limited"
    client, _, _ = _client([limited], backoff=BackoffPolicy(max_retries=0))
    with pytest.raises(RateLimitError) as exc:
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert exc.value.retry_after == 7
    quota = completed(
        envelope(result="Claude AI usage limit reached|1767268800", is_error=True), returncode=1)
    client, _, clock = _client([quota])
    with pytest.raises(QuotaExhaustedError) as exc:
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert exc.value.resets_at == "2026-01-01T12:00:00+00:00" and clock.sleeps == []


# -- contract shared by the fake and the real client ------------------------

def _make_fake(steps):
    client = FakeLlmClient(steps, backoff=BackoffPolicy(max_retries=2, base=1, factor=2, max_delay=60))
    return client, client.sleeps


def _make_real(steps):
    client, _, clock = _client(
        steps, backoff=BackoffPolicy(max_retries=2, base=1, factor=2, max_delay=60))
    return client, clock.sleeps


def _scenarios():
    limited = completed(
        envelope(result="429 rate limit", is_error=True, api_error_status=429), returncode=1)
    return {
        "valid": ([Reply(PLAN_JSON)], [ok_step(PLAN_JSON)], ["ok"], []),
        "malformed": ([Malformed(), Reply(PLAN_JSON)],
                      [ok_step("nope"), ok_step(PLAN_JSON)], ["invalid_output", "ok"], []),
        "invalid": ([Reply("{}"), Reply(PLAN_JSON)],
                    [ok_step("{}"), ok_step(PLAN_JSON)], ["invalid_output", "ok"], []),
        "rate_limit": ([RateLimited(), Reply(PLAN_JSON)],
                       [limited, ok_step(PLAN_JSON)], ["rate_limited", "ok"], [1]),
    }


@pytest.mark.parametrize("name", ["valid", "malformed", "invalid", "rate_limit"])
@pytest.mark.parametrize("kind", ["fake", "real"])
def test_fake_and_real_client_share_the_same_contract(tmp_path, kind, name):
    fake_steps, real_steps, outcomes, sleeps = _scenarios()[name]
    client, slept = (_make_fake(fake_steps) if kind == "fake" else _make_real(real_steps))
    result = client.ask("p", ShortPlan, permissions(tmp_path), "plan", model="asked")
    assert result.value == ShortPlan.from_json(PLAN_JSON)
    assert [c.outcome for c in result.calls] == outcomes and slept == sleeps
    assert all(c.model == "asked" for c in result.calls)
    assert result.consumption.calls == len(outcomes)


@pytest.mark.parametrize("kind", ["fake", "real"])
def test_both_clients_raise_only_llm_errors_on_persistent_failure(tmp_path, kind):
    if kind == "fake":
        client, _ = _make_fake([Reply("bad")] * 3)
    else:
        client, _ = _make_real([ok_step("bad")] * 3)
    with pytest.raises(LlmError):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
