"""Subscription only: the paid-API variables are stripped, never logged by value (cases 1-5)."""

import logging

from kaggle_agent.llm.environment import PAID_API_VARIABLES, build_cli_environment, redact
from kaggle_agent.llm.real import RealClaudeClient
from llm_helpers import PLAN_JSON, SECRET_KEY, SECRET_TOKEN, Clock, FakeRunner, ok_step, permissions
from kaggle_agent.proposals import ShortPlan


def _client(env, runner, clock=None):
    clock = clock or Clock()
    return RealClaudeClient(runner=runner, env=env, clock=clock.now, sleep=clock.sleep)


def test_api_key_is_stripped_other_variables_kept_and_inputs_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET_KEY)
    env = {"ANTHROPIC_API_KEY": SECRET_KEY, "CLAUDE_CODE_OAUTH_TOKEN": SECRET_TOKEN, "PATH": "/bin"}
    snapshot = dict(env)
    runner = FakeRunner([ok_step(PLAN_JSON)])
    _client(env, runner).ask("p", ShortPlan, permissions(tmp_path), "plan")
    seen = runner.calls[0].env
    assert "ANTHROPIC_API_KEY" not in seen
    assert seen == {"CLAUDE_CODE_OAUTH_TOKEN": SECRET_TOKEN, "PATH": "/bin"}
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == SECRET_KEY
    assert env == snapshot


def test_key_alone_never_reaches_the_process_or_any_diagnostic(tmp_path, caplog):
    env = {"ANTHROPIC_API_KEY": SECRET_KEY}
    cleaned, removed = build_cli_environment(env)
    assert cleaned == {} and removed == ("ANTHROPIC_API_KEY",)
    runner = FakeRunner([ok_step(PLAN_JSON)])
    with caplog.at_level(logging.WARNING):
        result = _client(env, runner).ask("p", ShortPlan, permissions(tmp_path), "plan")
    call = runner.calls[0]
    assert SECRET_KEY not in str(call.env) + str(call.argv) + call.input
    assert SECRET_KEY not in repr(result) + repr(result.calls)


def test_strip_warns_naming_the_variable_and_never_the_value(tmp_path, caplog):
    env = {"ANTHROPIC_API_KEY": SECRET_KEY, "PATH": "/bin"}
    with caplog.at_level(logging.WARNING):
        build_cli_environment(env)
        _client(env, FakeRunner([ok_step(PLAN_JSON)])).ask(
            "p", ShortPlan, permissions(tmp_path), "plan"
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    text = "\n".join(r.getMessage() for r in warnings)
    assert text.count("ANTHROPIC_API_KEY") == 2
    for start in range(len(SECRET_KEY) - 3):
        assert SECRET_KEY[start : start + 4] not in text


def test_stripped_names_are_exposed_and_no_warning_without_paid_variables(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        with_key = _client({"ANTHROPIC_API_KEY": SECRET_KEY}, FakeRunner([ok_step(PLAN_JSON)])).ask(
            "p", ShortPlan, permissions(tmp_path), "plan"
        )
    assert with_key.stripped_variables == ("ANTHROPIC_API_KEY",)
    assert with_key.calls[0].stripped_variables == ("ANTHROPIC_API_KEY",)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        clean = _client({"PATH": "/bin"}, FakeRunner([ok_step(PLAN_JSON)])).ask(
            "p", ShortPlan, permissions(tmp_path), "plan"
        )
    assert clean.stripped_variables == () and not caplog.records
    for obj in (with_key, with_key.calls[0]):
        assert SECRET_KEY not in repr(obj) and SECRET_KEY not in str(obj)


def test_other_billing_variables_are_stripped_too():
    env = {name: "x" for name in PAID_API_VARIABLES}
    cleaned, removed = build_cli_environment(env)
    assert cleaned == {} and removed == tuple(sorted(PAID_API_VARIABLES))


def test_redact_removes_the_token_from_text():
    env = {"CLAUDE_CODE_OAUTH_TOKEN": SECRET_TOKEN}
    assert SECRET_TOKEN not in redact(f"boom {SECRET_TOKEN} boom", env)
    assert redact("plain", {}) == "plain"
