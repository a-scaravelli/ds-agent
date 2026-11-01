"""Feature engineering role alone, with FakeLlmClient (case 3)."""

import pytest

from kaggle_agent.agents import (
    PendingTrace,
    RoleRejectedError,
    build_feature_engineering_prompt,
    run_feature_engineering,
)
from kaggle_agent.llm import FakeLlmClient, Malformed, RateLimited, Reply
from kaggle_agent.llm.fake import valid_attempt_json
from kaggle_agent.proposals import Attempt
from agents_helpers import (
    CANDIDATE_ID, context, feature_engineering_input, fixed_now, fresh_registry, permissions,
)


def _run(client, registry, tmp_path):
    return run_feature_engineering(
        client, registry, context(), feature_engineering_input(), permissions(tmp_path),
        now=fixed_now,
    )


def test_feature_engineering_id_is_assigned_by_the_caller(tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Reply(valid_attempt_json(id="MODEL-ID"))])
    result = _run(client, registry, tmp_path)
    assert isinstance(result.output, Attempt) and isinstance(result.trace, PendingTrace)
    expected = Attempt.from_json(valid_attempt_json(id=CANDIDATE_ID))
    assert result.output == expected
    assert result.output.bundle_id is None
    assert result.trace.output == expected.to_dict() and result.trace.role == "feature_engineering"
    assert build_feature_engineering_prompt(feature_engineering_input()) in client.calls[0].prompt
    assert registry.list_role_traces("C1") == []


def test_feature_engineering_malformed_rejected_and_rate_limit_waited(tmp_path):
    registry = fresh_registry()
    with pytest.raises(RoleRejectedError):
        _run(FakeLlmClient([Malformed()] * 3), registry, tmp_path)
    client = FakeLlmClient([RateLimited(retry_after=4), Reply(valid_attempt_json())])
    _run(client, registry, tmp_path)
    assert client.sleeps == [4.0]
