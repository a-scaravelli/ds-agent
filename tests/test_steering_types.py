"""Typed guidance of the DS: Steering, TryThis and CarryOver (cases 38, 39, 24, 53)."""

import dataclasses

import pytest

from kaggle_agent.proposals import ALLOWED_LIBRARIES, MODEL_FAMILIES, CarryOver, Steering, TryThis, ValidationError


# 39
def test_steering_rejects_a_contradiction_an_unknown_focus_and_an_unknown_family():
    with pytest.raises(ValidationError, match="both pinned and avoided"):
        Steering(pin_features=["age"], avoid_features=["age"])
    with pytest.raises(ValidationError, match="both to try and to avoid"):
        Steering(try_families=["linear"], avoid_families=["linear"])
    with pytest.raises(ValidationError, match="unknown focus"):
        Steering(focus="magic")
    with pytest.raises(ValidationError, match="unknown model family"):
        Steering(allowed_model_families=["quantum"])
    with pytest.raises(ValidationError):
        Steering(pin_features="age")


# 38
def test_steering_round_trips_and_an_empty_one_is_empty():
    steering = Steering(focus="feature_engineering", pin_features=["age", "age"], avoid_features=["id"],
                        try_families=["tree_ensemble"], allowed_model_families=["tree_ensemble", "linear"],
                        notes="  keep it simple ")
    assert steering.pin_features == ("age",) and steering.notes == "keep it simple"
    assert Steering.from_dict(steering.to_dict()) == steering
    assert not steering.empty and Steering().empty and Steering.from_dict(Steering().to_dict()).empty
    with pytest.raises(ValidationError):
        Steering.from_dict({"nope": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        steering.focus = "model_tuning"


# 53
def test_the_family_vocabulary_is_built_from_the_allowed_libraries():
    assert {"linear", "tree_ensemble"} <= set(MODEL_FAMILIES)
    assert {name for name in ALLOWED_LIBRARIES if name in MODEL_FAMILIES} == {"xgboost", "lightgbm", "catboost"}


# 40
def test_try_this_needs_a_hypothesis_and_keeps_its_constraints():
    request = TryThis("add a target encoding of the city", ["no new libraries"])
    assert TryThis.from_dict(request.to_dict()) == request
    with pytest.raises(ValidationError):
        TryThis("   ")
    with pytest.raises(ValidationError):
        TryThis("x", constraints="not a list")
    with pytest.raises(ValidationError):
        TryThis.from_dict({"hypothesis": "x", "code": "print(1)"})


# 24
def test_carry_over_is_capped_typed_and_never_holds_code():
    carry = CarryOver("toy-1", "toy-1-R1-c1", same_protocol=False, protocol_note="threshold_k 1 to 2",
                      attempt=[{"id": "a", "hypothesis": "h" * 5000, "validation_score": 0.7}],
                      reviewer=[f"text {i}" for i in range(20)])
    assert len(carry.attempt[0]["hypothesis"]) == 600 and len(carry.reviewer) == 5
    assert CarryOver.from_dict(carry.to_dict()) == carry
    with pytest.raises(ValidationError):
        CarryOver("toy-1", attempt=[{"id": "a", "code": ["a list is not a description"]}])
    with pytest.raises(ValidationError):
        CarryOver("toy-1", same_protocol="yes")
