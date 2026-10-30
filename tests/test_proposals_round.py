"""Guidance, round and reviewer report: validation and JSON round trip."""

import json

import pytest

from kaggle_agent.proposals import Guidance, ReviewerReport, Round, ValidationError

T = "2026-10-30T10:00:00+00:00"
GUIDANCE = {"id": "G-1", "round_id": "R-1", "text": "Try target encoding", "created_at": T}
ROUND = {"id": "R-1", "tasks": ["profile the data", "try a tree model"], "candidates": ["A-1", "A-2"]}
REPORT = {"id": "P-1", "round_id": "R-1", "text": "A-2 looks better on the segment", "created_at": T}


@pytest.mark.parametrize(
    "cls, data", [(Guidance, GUIDANCE), (Round, ROUND), (ReviewerReport, REPORT)]
)
def test_roundtrip_identical(cls, data):
    obj = cls.from_dict(data)
    assert obj.to_dict() == data
    assert cls.from_dict(obj.to_dict()) == obj
    text = obj.to_json()
    again = cls.from_json(text)
    assert again == obj and again.to_json() == text
    assert json.loads(text) == data


@pytest.mark.parametrize("cls, data", [(Guidance, GUIDANCE), (ReviewerReport, REPORT)])
def test_text_types_reject_missing_and_empty_fields(cls, data):
    for name in data:
        with pytest.raises(ValidationError, match=name):
            cls.from_dict({k: v for k, v in data.items() if k != name})
        with pytest.raises(ValidationError, match=name):
            cls.from_dict({**data, name: " "})
    for key in ("origin_round", "run_ids"):
        with pytest.raises(ValidationError, match=key):
            cls.from_dict({**data, key: "x"})


def test_round_validation():
    assert Round.from_dict({"id": "R-2", "tasks": []}).candidates == ()
    with pytest.raises(ValidationError, match="id"):
        Round.from_dict({"tasks": []})
    with pytest.raises(ValidationError, match="tasks"):
        Round.from_dict({"id": "R"})
    with pytest.raises(ValidationError, match=r"tasks"):
        Round.from_dict({**ROUND, "tasks": ["ok", ""]})
    with pytest.raises(ValidationError, match="candidates"):
        Round.from_dict({**ROUND, "candidates": ["A", "A"]})
    with pytest.raises(ValidationError, match="candidates"):
        Round.from_dict({**ROUND, "candidates": [""]})
    with pytest.raises(ValidationError, match="run_ids"):
        Round.from_dict({**ROUND, "run_ids": []})


@pytest.mark.parametrize("cls", [Guidance, Round, ReviewerReport])
def test_malformed_json_rejected(cls):
    for text in ("{oops", "[]", "7"):
        with pytest.raises(ValidationError):
            cls.from_json(text)
