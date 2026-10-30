"""Goal criteria and roles: validation and JSON round trip."""

import json

import pytest

from kaggle_agent.proposals import ROLES, GoalCriteria, GoalCriterion, ValidationError, validate_role

TWO = {
    "criteria": [
        {"metric": "accuracy", "threshold": 0.8, "direction": "maximize"},
        {"metric": "log_loss", "threshold": 0.4, "direction": "minimize"},
    ]
}


def test_goal_criteria_roundtrip_identical():
    goal = GoalCriteria.from_dict(TWO)
    assert goal.metrics == ("accuracy", "log_loss")
    assert GoalCriteria.from_dict(goal.to_dict()) == goal
    text = goal.to_json()
    again = GoalCriteria.from_json(text)
    assert again == goal and again.to_json() == text
    built = GoalCriteria((GoalCriterion("accuracy", 0.8, "maximize"),
                          GoalCriterion("log_loss", 0.4, "minimize")))
    assert built == goal
    assert GoalCriterion.from_json(built.criteria[0].to_json()) == built.criteria[0]


@pytest.mark.parametrize(
    "data, field",
    [
        ({"criteria": []}, "criteria"),
        ({}, "criteria"),
        ({"criteria": "x"}, "criteria"),
        ({"criteria": [{"metric": "a", "threshold": 1, "direction": "maximize"}] * 2}, "criteria"),
        ({"criteria": [{"metric": "a", "threshold": 1, "direction": "up"}]}, "direction"),
        ({"criteria": [{"metric": "a", "threshold": float("nan"), "direction": "maximize"}]}, "threshold"),
        ({"criteria": [{"metric": "a", "threshold": float("inf"), "direction": "maximize"}]}, "threshold"),
        ({"criteria": [{"metric": "a", "threshold": True, "direction": "maximize"}]}, "threshold"),
        ({"criteria": [{"metric": "a", "threshold": "1", "direction": "maximize"}]}, "threshold"),
        ({"criteria": [{"metric": "", "threshold": 1, "direction": "maximize"}]}, "metric"),
        ({"criteria": [{"metric": "a", "threshold": 1, "direction": "maximize", "x": 1}]}, "x"),
        ({"criteria": [{"metric": "a", "threshold": 1}]}, "direction"),
        ({"criteria": [3]}, "criteria"),
        ({"criteria": [], "extra": 1}, "extra"),
    ],
)
def test_goal_criteria_validation(data, field):
    with pytest.raises(ValidationError, match=field):
        GoalCriteria.from_dict(data)
    with pytest.raises(ValidationError):
        GoalCriteria.from_json("{oops")


def test_roles_are_the_five_of_the_design():
    assert ROLES == ("coordinator", "data", "feature_engineering", "modeling", "reviewer")
    assert validate_role("modeling") == "modeling"
    for bad in ("planner", "", None, 3):
        with pytest.raises(ValidationError, match="role"):
            validate_role(bad)
    assert json.loads(GoalCriteria.from_dict(TWO).to_json()) == TWO
