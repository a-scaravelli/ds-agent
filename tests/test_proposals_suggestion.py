"""Suggestion: text, origin cycle and spawned cycles."""

import dataclasses

import pytest

from kaggle_agent.proposals import Suggestion, ValidationError

DATA = {"id": "S-1", "text": "Try target encoding", "origin_cycle": "C-1", "spawned_cycles": ["C-2", "C-3"]}


def test_suggestion_validation():
    s = Suggestion.from_dict(DATA)
    assert s.spawned_cycles == ("C-2", "C-3")
    bare = Suggestion.from_dict({k: v for k, v in DATA.items() if k != "spawned_cycles"})
    assert bare.spawned_cycles == ()
    for bad in (
        {"text": ""},
        {"text": "  "},
        {"origin_cycle": ""},
        {"spawned_cycles": ["C-2", "C-2"]},
        {"spawned_cycles": ["C-1"]},
        {"spawned_cycles": [""]},
    ):
        with pytest.raises(ValidationError):
            Suggestion.from_dict({**DATA, **bad})
    with pytest.raises(ValidationError, match="text"):
        Suggestion.from_dict({k: v for k, v in DATA.items() if k != "text"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.text = "changed"


def test_suggestion_roundtrip_identical():
    for data in (DATA, {**DATA, "spawned_cycles": []}):
        s = Suggestion.from_dict(data)
        text = s.to_json()
        assert Suggestion.from_json(text) == s
        assert Suggestion.from_json(text).to_json() == text
        assert Suggestion.from_dict(s.to_dict()) == s


def test_suggestion_unknown_keys_and_bad_json_rejected():
    with pytest.raises(ValidationError, match="extra"):
        Suggestion.from_dict({**DATA, "extra": 1})
    for text in ("{oops", "[]", "null"):
        with pytest.raises(ValidationError):
            Suggestion.from_json(text)
