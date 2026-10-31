"""Reviewer role alone, with FakeLlmClient (case 5)."""

import json

import pytest

from kaggle_agent.agents import RoleRejectedError, run_reviewer
from kaggle_agent.llm import FakeLlmClient, Malformed, RateLimited, Reply
from kaggle_agent.proposals import ReviewerReport, ValidationError
from agents_helpers import (
    NOW, REPORT_ID, context, fixed_now, fresh_registry, permissions, reviewer_input,
    reviewer_json,
)


def _run(client, registry, tmp_path, report_id=REPORT_ID):
    return run_reviewer(
        client, registry, context(), reviewer_input(), permissions(tmp_path),
        report_id=report_id, now=fixed_now,
    )


def test_reviewer_report_has_fixed_id_round_and_date(tmp_path):
    registry = fresh_registry()
    client = FakeLlmClient([Reply(reviewer_json("looks fine"))])
    report = _run(client, registry, tmp_path)
    assert report == ReviewerReport(REPORT_ID, "R1", "looks fine", NOW.isoformat())
    (trace,) = registry.list_role_traces("C1", role="reviewer")
    assert trace.output_type == "ReviewerReport" and trace.output == report.to_dict()
    assert trace.input_type == "ReviewerInput"
    assert "does not decide" in client.calls[0].prompt


def test_reviewer_ignores_an_id_or_date_written_by_the_model(tmp_path):
    reply = json.dumps({"text": "t", "id": "EVIL", "created_at": "x"})
    with pytest.raises(RoleRejectedError):
        _run(FakeLlmClient([Reply(reply)] * 3), fresh_registry(), tmp_path)


def test_reviewer_malformed_rejected_and_rate_limit_waited(tmp_path):
    registry = fresh_registry()
    with pytest.raises(RoleRejectedError):
        _run(FakeLlmClient([Malformed()] * 3), registry, tmp_path)
    client = FakeLlmClient([RateLimited(retry_after=5), Reply(reviewer_json())])
    _run(client, registry, tmp_path)
    assert client.sleeps == [5.0]


def test_reviewer_needs_a_report_id_before_any_call(tmp_path):
    client = FakeLlmClient([Reply(reviewer_json())])
    with pytest.raises(ValidationError):
        _run(client, fresh_registry(), tmp_path, report_id="")
    assert client.calls == []
