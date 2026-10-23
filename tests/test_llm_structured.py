"""Parsing and validation of structured output (cases 12, 13, 15)."""

import pytest

from kaggle_agent.llm.structured import extract_json, json_contract, parse_structured
from kaggle_agent.proposals import Attempt, ShortPlan, ValidationError
from llm_helpers import ATTEMPT_JSON, PLAN_JSON


def test_valid_payloads_become_the_existing_types():
    assert isinstance(parse_structured(PLAN_JSON, ShortPlan), ShortPlan)
    assert parse_structured(ATTEMPT_JSON, Attempt).id == "A1"


def test_json_in_a_code_fence_or_with_text_around_is_extracted():
    fenced = f"Here it is:\n```json\n{PLAN_JSON}\n```\nDone."
    assert parse_structured(fenced, ShortPlan) == parse_structured(PLAN_JSON, ShortPlan)
    prose = f"Sure! {PLAN_JSON} Hope it helps."
    assert parse_structured(prose, ShortPlan) == parse_structured(PLAN_JSON, ShortPlan)
    assert extract_json("  {}  ") == "{}"


@pytest.mark.parametrize("text", ["not json", "", "[1, 2]", "{", '{"metric": 3}', None, 42])
def test_any_problem_is_a_validation_error_and_never_another_exception(text):
    with pytest.raises(ValidationError):
        parse_structured(text, ShortPlan)


def test_validation_error_names_the_field():
    with pytest.raises(ValidationError) as exc:
        parse_structured('{"id": "A", "phase": "p"}', Attempt)
    assert "hypothesis" in str(exc.value)


def test_json_contract_lists_the_fields_of_the_type():
    text = json_contract(Attempt)
    assert "what_changed" in text and "JSON" in text
    assert "Optional keys" in text and "parent_id" in text
    assert "Optional" not in json_contract(ShortPlan)
