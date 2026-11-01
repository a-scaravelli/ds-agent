"""Data role alone, with FakeLlmClient (case 2)."""

import pytest

from kaggle_agent.agents import DataFindings, PURPOSES, RoleRejectedError, run_data
from kaggle_agent.llm import FakeLlmClient, Malformed, RateLimited, Reply
from agents_helpers import (
    context, data_input, data_json, fixed_now, fresh_registry, permissions,
)


def _run(client, registry, tmp_path):
    return run_data(client, registry, context(), data_input(), permissions(tmp_path), now=fixed_now)


def test_data_valid_output_has_the_four_fields(tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Reply(data_json())])
    out = _run(client, registry, tmp_path)
    assert isinstance(out, DataFindings)
    assert out.profile == "100 rows, 2 columns"
    assert out.quality_issues == ("10% missing age",)
    assert out.leakage_hints == ("none seen",) and out.proposed_cleaning == ("impute age",)
    record = client.call_log()[0]
    assert (record.role, record.purpose) == ("data", PURPOSES["data"])
    assert "profile the data" in client.calls[0].prompt


def test_data_malformed_is_rejected_and_rate_limit_waited(tmp_path):
    registry = fresh_registry()
    with pytest.raises(RoleRejectedError):
        _run(FakeLlmClient([Malformed()] * 3), registry, tmp_path)
    client = FakeLlmClient([RateLimited(retry_after=2), Reply(data_json())])
    _run(client, registry, tmp_path)
    assert client.sleeps == [2.0]
    assert len(registry.list_role_traces("C1", role="data")) == 2
