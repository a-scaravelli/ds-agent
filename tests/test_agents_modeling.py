"""Modeling role alone, with FakeLlmClient (case 4)."""

import pytest

from kaggle_agent.agents import RoleRejectedError, run_modeling
from kaggle_agent.llm import FakeLlmClient, Malformed, RateLimited, Reply
from kaggle_agent.llm.fake import valid_attempt_json
from kaggle_agent.proposals import Attempt
from agents_helpers import (
    CANDIDATE_ID, attempt, context, fixed_now, fresh_registry, modeling_input, permissions,
)


def _run(client, registry, tmp_path, role_input=None):
    return run_modeling(
        client, registry, context(), role_input or modeling_input(), permissions(tmp_path),
        now=fixed_now,
    )


def test_modeling_id_is_assigned_and_previous_attempt_is_in_the_prompt(tmp_path):
    previous = attempt("FE-9", what_changed="add interaction terms")
    client = FakeLlmClient([Reply(valid_attempt_json(id="MODEL-ID"))])
    result = _run(client, fresh_registry(), tmp_path, modeling_input(previous))
    assert isinstance(result.output, Attempt) and result.output.id == CANDIDATE_ID
    assert result.output.hypothesis == "scaling helps"
    assert "add interaction terms" in client.calls[0].prompt
    assert result.trace.input["previous_attempt"] == previous.to_dict()


def test_modeling_malformed_rejected_and_rate_limit_waited(tmp_path):
    registry = fresh_registry()
    with pytest.raises(RoleRejectedError):
        _run(FakeLlmClient([Malformed()] * 3), registry, tmp_path)
    client = FakeLlmClient([RateLimited(retry_after=1), Reply(valid_attempt_json())])
    _run(client, registry, tmp_path)
    assert client.sleeps == [1.0]
