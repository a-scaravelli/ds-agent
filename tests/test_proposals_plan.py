"""ShortPlan: five mandatory items, word limit, JSON round trip."""

import json

import pytest

from design_blocks import block_with_key
from kaggle_agent.proposals import MAX_PLAN_WORDS, ShortPlan, ValidationError

ITEMS = ["metric", "validation", "feature_engineering", "models", "baseline"]

VALID = {
    "metric": "official competition metric; it is the one that decides the leaderboard",
    "validation": "partitions that mimic the separation between train and test (evidence: comparison of the distributions)",
    "feature_engineering": ["missing-value indicators", "encoding of rare categories"],
    "models": ["boosted trees", "regularized linear model"],
    "baseline": "simple model with default parameters",
}


@pytest.mark.parametrize("item", ITEMS)
def test_missing_item_rejected(item):
    data = {k: v for k, v in VALID.items() if k != item}
    with pytest.raises(ValidationError) as exc:
        ShortPlan.from_dict(data)
    assert item in str(exc.value)
    with pytest.raises(ValidationError, match=item):
        ShortPlan.from_json(json.dumps(data))
    kwargs = dict(data)
    with pytest.raises((ValidationError, TypeError)):
        ShortPlan(**kwargs)


@pytest.mark.parametrize(
    "item, bad",
    [
        ("metric", ""),
        ("metric", "   "),
        ("metric", 5),
        ("validation", ""),
        ("baseline", "\n\t"),
        ("baseline", 3.5),
        ("feature_engineering", []),
        ("feature_engineering", ["ok", ""]),
        ("feature_engineering", "not a list"),
        ("models", []),
        ("models", ["  "]),
        ("models", [1, 2]),
    ],
)
def test_empty_or_wrong_type_rejected(item, bad):
    with pytest.raises(ValidationError):
        ShortPlan.from_dict({**VALID, item: bad})


def _plan_with_words(total: int) -> dict:
    other = sum(len(str(v).split()) if isinstance(v, str) else sum(len(x.split()) for x in v)
                for k, v in VALID.items() if k != "baseline")
    return {**VALID, "baseline": " ".join(["word"] * (total - other))}


def test_plan_over_word_limit_rejected():
    assert MAX_PLAN_WORDS == 300
    ok = ShortPlan.from_dict(_plan_with_words(MAX_PLAN_WORDS))
    assert ok.word_count() == MAX_PLAN_WORDS
    with pytest.raises(ValidationError, match=str(MAX_PLAN_WORDS)):
        ShortPlan.from_dict(_plan_with_words(MAX_PLAN_WORDS + 1))


def test_plan_json_roundtrip_identical():
    plan = ShortPlan.from_dict({**VALID, "baseline": "modello semplice, età e città"})
    text = plan.to_json()
    again = ShortPlan.from_json(text)
    assert again == plan
    assert again.to_json() == text
    assert "età" in text
    assert list(json.loads(text)) == ITEMS
    assert ShortPlan.from_dict(plan.to_dict()) == plan


def test_plan_example_from_design_is_valid():
    plan = ShortPlan.from_dict(VALID)
    assert plan.models == ("boosted trees", "regularized linear model")
    assert plan.word_count() <= MAX_PLAN_WORDS


def test_plan_unknown_keys_and_bad_json_rejected():
    with pytest.raises(ValidationError, match="extra"):
        ShortPlan.from_dict({**VALID, "extra": "x"})
    for text in ("{not json", "[1, 2]", '"text"', "null"):
        with pytest.raises(ValidationError):
            ShortPlan.from_json(text)
    with pytest.raises(ValidationError):
        ShortPlan.from_dict(["not", "a", "dict"])


def test_plan_example_matches_design_block():
    block = block_with_key("metric")
    assert block == VALID
    assert ShortPlan.from_dict(block).to_dict() == VALID


def test_contract_hints_match_the_validation():
    hints = ShortPlan.contract_hints()
    assert set(hints) - {"__all__"} == set(ITEMS) and "__all__" in hints
    assert str(MAX_PLAN_WORDS) in hints["__all__"]
    for name in ("feature_engineering", "models"):  # empty list is rejected, as the hint says
        with pytest.raises(ValidationError):
            ShortPlan.from_dict({**VALID, name: []})
    for name in ("metric", "validation", "baseline"):
        with pytest.raises(ValidationError):
            ShortPlan.from_dict({**VALID, name: " "})
    with pytest.raises(ValidationError, match="words"):
        ShortPlan.from_dict({**VALID, "baseline": "word " * (MAX_PLAN_WORDS + 1)})
