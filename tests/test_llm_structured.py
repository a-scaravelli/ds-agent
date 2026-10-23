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


def test_shortplan_contract_has_types_nonempty_lists_and_word_limit():
    from kaggle_agent.proposals import MAX_PLAN_WORDS

    text = json_contract(ShortPlan)
    for name in ("metric", "validation", "baseline"):
        assert f"- {name}: string, required" in text
    for name in ("feature_engineering", "models"):
        assert f"- {name}: list of strings, required" in text
        assert f"- {name}: non-empty list of non-empty strings" in text
    assert f"within {MAX_PLAN_WORDS} whitespace-separated words" in text and "300" in text
    assert "use only these keys" in text and "optional" not in text.lower()


def test_attempt_contract_has_required_fields_types_and_evidence_rule():
    text = json_contract(Attempt)
    for name in ("id", "phase", "hypothesis", "rationale", "what_changed", "measure"):
        assert f"- {name}: string, required" in text
        assert f"- {name}: non-empty string" in text
    assert "- exploratory: boolean, optional" in text
    assert "- parent_id: string or null, optional" in text
    assert "- expected_effect: string or null, optional" in text
    for name in ("evidence", "risks", "artifacts"):
        assert f"- {name}: list of strings, optional" in text
    assert "different from id" in text
    assert "if evidence is empty, exploratory must be true" in text


def test_type_without_hints_or_with_unmapped_annotations_still_gets_a_contract():
    import dataclasses

    from kaggle_agent.proposals import Suggestion

    @dataclasses.dataclass
    class Probe:
        name: str
        count: int
        extra: dict
        note: str | None = None

    text = json_contract(Probe)
    assert "Required keys: name, count, extra." in text and "Optional keys: note." in text
    assert "- name: string, required" in text
    assert "- count: any JSON value, required" in text and "- extra: any JSON value, required" in text
    assert "- note: string or null, optional" in text
    assert "Constraints" not in text
    suggestion = json_contract(Suggestion)
    assert "Required keys" in suggestion and "Constraints" not in suggestion


def test_llm_code_does_not_hardcode_the_word_limit():
    from pathlib import Path

    import kaggle_agent.llm as llm

    for path in Path(llm.__file__).parent.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "300" not in source and "MAX_PLAN_WORDS" not in source, path.name
