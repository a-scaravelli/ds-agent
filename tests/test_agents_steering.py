"""Steering, carry-over and the request reach the role inputs and prompts only when set (cases 24, 38, 40)."""

import dataclasses
import json

from agents_helpers import coordinator_input, feature_engineering_input, modeling_input

from kaggle_agent.agents import (
    build_coordinator_prompt,
    build_feature_engineering_prompt,
    build_modeling_prompt,
)
from kaggle_agent.agents.prompting import CARRY_NOTE, REQUEST_NOTE, STEERING_NOTE
from kaggle_agent.proposals import CarryOver, Steering, TryThis

STEERING = Steering(focus="feature_engineering", pin_features=["age"], avoid_families=["svm"])
CARRY = CarryOver("toy-1", "toy-1-R1-c1", attempt=[{"id": "toy-1-R1-c1", "hypothesis": "bin the age",
                                                     "validation_score": 0.71}])
REQUEST = TryThis("target-encode the city", ["no new libraries"])

BUILDERS = {"coordinator": (coordinator_input, build_coordinator_prompt),
            "feature_engineering": (feature_engineering_input, build_feature_engineering_prompt),
            "modeling": (modeling_input, build_modeling_prompt)}


# 38
def test_an_input_without_the_new_fields_renders_exactly_as_before():
    for make, build in BUILDERS.values():
        role_input = make()
        data = role_input.to_dict()
        assert not {"steering", "carry", "request"} & set(data)
        prompt = build(role_input)
        for note in (STEERING_NOTE, CARRY_NOTE, REQUEST_NOTE):
            assert note not in prompt


# 38
def test_steering_and_carry_are_typed_fields_with_a_note_in_every_role_prompt():
    for name, (make, build) in BUILDERS.items():
        role_input = dataclasses.replace(make(), steering=STEERING, carry=CARRY)
        data = role_input.to_dict()
        assert data["steering"]["focus"] == "feature_engineering" and data["steering"]["pin_features"] == ["age"]
        assert data["carry"]["origin_attempt"] == "toy-1-R1-c1"
        prompt = build(role_input)
        assert STEERING_NOTE in prompt and CARRY_NOTE in prompt, name
        assert json.dumps(data, sort_keys=True, ensure_ascii=False) in prompt
        assert type(role_input).from_dict(data) == role_input
        cleared = dataclasses.replace(role_input, steering=None)
        assert "steering" not in cleared.to_dict() and STEERING_NOTE not in build(cleared)


# 40
def test_the_try_this_request_is_a_typed_field_of_the_two_candidate_roles_only():
    for name in ("feature_engineering", "modeling"):
        make, build = BUILDERS[name]
        role_input = dataclasses.replace(make(), request=REQUEST)
        assert role_input.to_dict()["request"] == {"hypothesis": "target-encode the city",
                                                   "constraints": ["no new libraries"]}
        assert REQUEST_NOTE in build(role_input)
        assert type(role_input).from_dict(role_input.to_dict()) == role_input
    assert "request" not in {f.name for f in dataclasses.fields(coordinator_input())}
