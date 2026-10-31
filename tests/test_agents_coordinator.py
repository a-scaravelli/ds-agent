"""Coordinator role alone, with FakeLlmClient (case 1)."""

import json

import pytest

from kaggle_agent.agents import (
    CoordinatorOutput,
    RoleRejectedError,
    build_coordinator_prompt,
    run_coordinator,
)
from kaggle_agent.llm import FakeLlmClient, Malformed, RateLimited, Reply
from kaggle_agent.proposals import GoalCriteria, ShortPlan
from agents_helpers import (
    context,
    coordinator_input,
    coordinator_json,
    fixed_now,
    fresh_registry,
    permissions,
)


def _run(client, registry, tmp_path):
    return run_coordinator(
        client, registry, context(), coordinator_input(), permissions(tmp_path), now=fixed_now
    )


def test_coordinator_valid_output_is_typed(tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Reply(coordinator_json())])
    out = _run(client, registry, tmp_path)
    assert isinstance(out, CoordinatorOutput)
    assert isinstance(out.plan, ShortPlan) and isinstance(out.goal, GoalCriteria)
    assert out.tasks == ("try tree models",) and out.goal.metrics == ("accuracy",)
    (call,) = client.calls
    assert build_coordinator_prompt(coordinator_input()) in call.prompt
    assert "Required keys: plan, tasks." in call.prompt and "Constraints:" in call.prompt
    assert json.dumps(coordinator_input().to_dict(), sort_keys=True) in call.prompt
    assert client.call_log()[0].role == "coordinator"


def test_coordinator_goal_is_optional(tmp_path):
    data = json.loads(coordinator_json())
    del data["goal"]
    out = _run(FakeLlmClient([Reply(json.dumps(data))]), fresh_registry(), tmp_path)
    assert out.goal is None


def test_coordinator_malformed_is_rejected_and_rate_limit_is_waited(tmp_path):
    registry = fresh_registry()
    with pytest.raises(RoleRejectedError):
        _run(FakeLlmClient([Malformed()] * 3), registry, tmp_path)
    client = FakeLlmClient([RateLimited(retry_after=3), Reply(coordinator_json())])
    _run(client, registry, tmp_path)
    assert client.sleeps == [3.0]
    assert [t.outcome for t in registry.list_role_traces("C1", role="coordinator")] == [
        "invalid_output", "ok",
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"tasks": []},
        {"plan": {"metric": "m"}},
        {"extra": 1},
        {"goal": {"criteria": []}},
    ],
)
def test_coordinator_invalid_payloads_are_rejected(change, tmp_path):
    data = {**json.loads(coordinator_json()), **change}
    with pytest.raises(RoleRejectedError):
        _run(FakeLlmClient([Reply(json.dumps(data))] * 3), fresh_registry(), tmp_path)
