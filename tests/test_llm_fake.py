"""The fake is deterministic and simulates every failure (cases 22, 30)."""

import pytest

from kaggle_agent.llm import (
    CliCrash,
    CliFailureError,
    FakeLlmClient,
    Malformed,
    RateLimited,
    Reply,
    Usage,
)
from kaggle_agent.llm.fake import FakeScriptExhaustedError
from kaggle_agent.proposals import ShortPlan
from llm_helpers import PLAN_JSON, permissions


def test_fake_cli_crash_raises_cli_failure_and_is_recorded(tmp_path):
    client = FakeLlmClient([CliCrash("boom")])
    with pytest.raises(CliFailureError, match="boom"):
        client.ask("p", ShortPlan, permissions(tmp_path), "plan")
    assert [c.outcome for c in client.call_log()] == ["cli_error"]
    assert client.consumption().calls == 1


def _script():
    return [RateLimited(retry_after=2), Malformed(duration=1), Reply(PLAN_JSON, Usage(3, 4), 2, "m")]


def test_same_script_gives_identical_results_records_and_sleeps(tmp_path):
    runs = []
    for _ in range(2):
        client = FakeLlmClient(_script())
        result = client.ask("p", ShortPlan, permissions(tmp_path), "plan")
        runs.append((result, client.sleeps))
    assert runs[0] == runs[1]


def test_exhausted_script_raises_an_explicit_fake_error_and_calls_are_kept(tmp_path):
    perms = permissions(tmp_path, "Read")
    client = FakeLlmClient([Reply(PLAN_JSON)], model="m1")
    client.ask("first", ShortPlan, perms, "plan")
    with pytest.raises(FakeScriptExhaustedError, match="exhausted"):
        client.ask("second", ShortPlan, perms, "plan", model="m2")
    assert [(c.model, c.permissions) for c in client.calls] == [("m1", perms), ("m2", perms)]
    assert client.calls[0].prompt.startswith("first") and client.calls[1].prompt.startswith("second")
