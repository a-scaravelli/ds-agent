"""Attempt: mandatory fields, exploratory rule, JSON round trip."""

import json

import pytest

from design_blocks import block_with_key
from kaggle_agent.proposals import Attempt, ValidationError

T0014 = {
    "id": "T-0014",
    "phase": "iteration",
    "parent_id": "T-0009",
    "hypothesis": "Errors are concentrated in records with a missing value in 'column_a'; a missing-value indicator reduces the error in that segment.",
    "rationale": "In attempt T-0009 the error is much higher where 'column_a' is missing.",
    "evidence": ["T-0009/error_by_segment", "profile/missing_pattern"],
    "what_changed": "added a missing-value indicator feature; model and hyperparameters unchanged",
    "measure": "frozen metric on the validation set and on the segment; falsified if the improvement is below the significance threshold of the protocol",
}

FULL = {
    **T0014,
    "exploratory": False,
    "expected_effect": "small decrease of the error",
    "estimated_cost": "2 minutes of compute",
    "risks": ["leakage", "overfitting of the validation set"],
    "artifacts": ["T-0014/diff", "T-0014/config.json"],
}

MINIMAL = {
    "id": "T-0001",
    "phase": "baseline",
    "hypothesis": "A default model sets a reference.",
    "rationale": "Nothing to compare with yet.",
    "what_changed": "first model",
    "measure": "frozen metric on the validation set",
    "exploratory": True,
}


@pytest.mark.parametrize("variant", ["missing", "empty", "spaces"])
def test_missing_what_changed_rejected(variant):
    data = {k: v for k, v in T0014.items() if k != "what_changed"}
    if variant == "empty":
        data["what_changed"] = ""
    elif variant == "spaces":
        data["what_changed"] = "  \n "
    for build in (Attempt.from_dict, lambda d: Attempt.from_json(json.dumps(d))):
        with pytest.raises(ValidationError, match="what_changed"):
            build(data)


@pytest.mark.parametrize("field", ["id", "phase", "hypothesis", "rationale", "measure"])
def test_other_required_fields_and_exploratory_rule(field):
    data = {k: v for k, v in T0014.items() if k != field}
    with pytest.raises(ValidationError, match=field):
        Attempt.from_dict(data)
    with pytest.raises(ValidationError, match=field):
        Attempt.from_dict({**T0014, field: " "})

    no_evidence = {k: v for k, v in T0014.items() if k != "evidence"}
    with pytest.raises(ValidationError, match="evidence"):
        Attempt.from_dict(no_evidence)
    with pytest.raises(ValidationError, match="evidence"):
        Attempt.from_dict({**no_evidence, "exploratory": False})
    assert Attempt.from_dict({**no_evidence, "exploratory": True}).exploratory is True

    with pytest.raises(ValidationError, match="parent_id"):
        Attempt.from_dict({**T0014, "parent_id": "T-0014"})
    with pytest.raises(ValidationError, match="exploratory"):
        Attempt.from_dict({**T0014, "exploratory": "yes"})
    with pytest.raises(ValidationError):
        Attempt.from_dict({**T0014, "evidence": ["ok", ""]})


def test_attempt_roundtrip_identical():
    for data in (FULL, MINIMAL):
        attempt = Attempt.from_dict(data)
        assert Attempt.from_dict(attempt.to_dict()) == attempt
        text = attempt.to_json()
        assert Attempt.from_json(text) == attempt
        assert attempt.to_json() == text
        assert Attempt.from_json(text).to_json() == text
    again = Attempt.from_json(Attempt.from_dict(MINIMAL).to_json())
    assert again.parent_id is None and again.evidence == () and again.risks == ()


def test_attempt_example_from_design_is_valid():
    attempt = Attempt.from_dict(T0014)
    assert attempt.parent_id == "T-0009"
    assert attempt.exploratory is False
    assert attempt.evidence == ("T-0009/error_by_segment", "profile/missing_pattern")
    with pytest.raises(Exception):
        attempt.what_changed = "other"


def test_attempt_unknown_keys_and_bad_json_rejected():
    with pytest.raises(ValidationError, match="extra"):
        Attempt.from_dict({**T0014, "extra": 1})
    for text in ("{oops", "[]", "42"):
        with pytest.raises(ValidationError):
            Attempt.from_json(text)


def test_attempt_example_matches_design_block():
    block = block_with_key("what_changed")
    assert block == T0014
    attempt = Attempt.from_dict(block)
    assert attempt.parent_id == "T-0009"
    assert attempt.what_changed == T0014["what_changed"]
