"""RunLimits and the prompts of the roles that carry it (cases 1, 5, 6, 7, 8, 9, 10, 17)."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from agents_helpers import (
    context,
    coordinator_input,
    coordinator_json,
    feature_engineering_input,
    fixed_now,
    fresh_registry,
    modeling_input,
    permissions,
    reviewer_input,
)

from kaggle_agent.agents import (
    CoordinatorInput,
    FeatureEngineeringInput,
    ModelingInput,
    ReviewerInput,
    RunLimits,
    build_coordinator_prompt,
    build_feature_engineering_prompt,
    build_modeling_prompt,
    build_reviewer_prompt,
    run_coordinator,
)
from kaggle_agent.llm import FakeLlmClient, Reply
from kaggle_agent.proposals import ALLOWED_LIBRARIES, LIBRARY_APIS, ValidationError

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
SIX = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost")
NO_CATBOOST = tuple(name for name in ALLOWED_LIBRARIES if name != "catboost")
RULE = "exceeds 2 times the baseline dispersion"


def limits(**changes):
    values = dict(
        rounds=3, candidates=2, stagnation_k=2, fix_retries=3, current_round=1, threshold_k=2.0,
        validation_scheme="stratified_kfold", repetitions=3, allowed_libraries=ALLOWED_LIBRARIES,
    )
    return RunLimits(**{**values, **changes})


def with_limits(role_input, **changes):
    return dataclasses.replace(role_input, limits=limits(**changes))


# 1
def test_the_coordinator_input_and_prompt_carry_the_limits_the_rule_and_the_libraries(tmp_path):
    role_input = with_limits(coordinator_input())
    data = role_input.to_dict()["limits"]
    assert (data["rounds"], data["candidates"], data["current_round"], data["threshold_k"]) == (3, 2, 1, 2.0)
    assert data["stagnation_k"] == 2 and data["fix_retries"] == 3
    assert "mean paired difference exceeds 2 times the baseline dispersion" in data["significance_rule"]
    assert tuple(data["allowed_libraries"]) == SIX == ALLOWED_LIBRARIES
    registry = fresh_registry()
    client = FakeLlmClient([Reply(coordinator_json())])
    run_coordinator(client, registry, context(), role_input, permissions(tmp_path), now=fixed_now)
    (call,) = client.calls
    assert build_coordinator_prompt(role_input) in call.prompt
    payload = json.dumps(role_input.to_dict(), sort_keys=True, ensure_ascii=False)
    assert payload in call.prompt
    for text in ('"rounds": 3', '"candidates": 2', '"current_round": 1', '"threshold_k": 2.0',
                 "mean paired difference exceeds 2 times the baseline dispersion", *SIX):
        assert text in call.prompt, text
    (trace,) = registry.list_role_traces("C1", role="coordinator")
    assert trace.input["limits"] == data


# 5
def test_the_rule_is_in_plain_words_with_the_threshold_of_the_protocol():
    assert limits(threshold_k=1.5).significance_rule == (
        "An improvement counts only if the mean paired difference exceeds 1.5 times the baseline "
        "dispersion (threshold_k = 1.5)."
    )
    prompt = build_coordinator_prompt(with_limits(coordinator_input(), threshold_k=1.5, repetitions=3))
    assert "1.5 times the baseline dispersion" in prompt
    assert '"validation_scheme": "stratified_kfold"' in prompt and '"repetitions": 3' in prompt


# 6
def test_the_libraries_have_one_source_and_the_imports_work_in_a_fresh_interpreter():
    from kaggle_agent.writer import DEFAULT_DEPENDENCIES
    from kaggle_agent.writer.prompt import build_prompt
    from writer_helpers import make_request, task

    assert DEFAULT_DEPENDENCIES is ALLOWED_LIBRARIES and ALLOWED_LIBRARIES == SIX
    assert set(LIBRARY_APIS) <= set(ALLOWED_LIBRARIES)
    code = ("import kaggle_agent.proposals, kaggle_agent.agents, kaggle_agent.writer, "
            "kaggle_agent.loop, kaggle_agent.orchestrator")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(SRC.parent), "PATH": ""})
    assert done.returncode == 0, done.stderr
    request = make_request("x")
    text = build_prompt(request, task(), DEFAULT_DEPENDENCIES)
    assert ", ".join(SIX) in text  # the writer prompt lists the same six


# 7
def test_feature_engineering_modeling_and_reviewer_receive_the_limits():
    fe = build_feature_engineering_prompt(with_limits(feature_engineering_input()))
    modeling = build_modeling_prompt(with_limits(modeling_input()))
    for prompt in (" ".join(fe.split()), " ".join(modeling.split())):
        assert "allowed_libraries" in prompt and "torch" in prompt and "candidates per round" in prompt
        assert all(name in prompt for name in SIX)
    review = build_reviewer_prompt(with_limits(reviewer_input()))
    assert "current_round" in review and "significance rule" in review and RULE in review
    assert "does not decide" in review
    assert isinstance(with_limits(feature_engineering_input()), FeatureEngineeringInput)
    from kaggle_agent.agents import CoordinatorOutput, ReviewerNotes
    from kaggle_agent.proposals import Attempt

    assert [f.name for f in dataclasses.fields(ReviewerNotes)] == ["text"]
    assert [f.name for f in dataclasses.fields(Attempt)][:2] == ["id", "phase"]
    assert [f.name for f in dataclasses.fields(CoordinatorOutput)] == ["plan", "tasks", "goal"]


# 8
def test_the_coordinator_prompt_tells_to_plan_inside_the_limits_and_is_unchanged_without_them():
    plain = build_coordinator_prompt(coordinator_input())
    assert "Run limits" not in plain and "limits" not in json.loads(plain.split("Input (JSON):\n")[1])
    prompt = build_coordinator_prompt(with_limits(coordinator_input()))
    for text in ("plan only for the given number of", "fit tuning inside them",
                 "Never plan beyond the last round", "torch, tensorflow", "Use only the libraries"):
        assert text in prompt.replace("\n", " ").replace("  ", " "), text
    instructions = plain.split("\n\nInput (JSON):")[0]
    assert prompt.startswith(instructions)
    from kaggle_agent.agents.prompting import render_prompt
    from kaggle_agent.agents.coordinator import INSTRUCTIONS

    assert plain == render_prompt(INSTRUCTIONS, coordinator_input())


# 9
@pytest.mark.parametrize("changes", [
    {"rounds": 0}, {"candidates": 0}, {"rounds": True}, {"rounds": 2.0}, {"fix_retries": -1},
    {"threshold_k": -0.1}, {"threshold_k": True}, {"allowed_libraries": ()}, {"allowed_libraries": ("",)},
    {"current_round": -1}, {"repetitions": 0}, {"validation_scheme": " "},
])
def test_invalid_limits_are_rejected(changes):
    with pytest.raises(ValidationError):
        limits(**changes)


def test_limits_edges_and_round_trip_and_unknown_key():
    ok = limits(fix_retries=0, current_round=0)
    assert ok.fix_retries == 0 and ok.current_round == 0
    assert RunLimits.from_dict(ok.to_dict()) == ok and RunLimits.from_json(ok.to_json()) == ok
    data = ok.to_dict()
    data["significance_rule"] = "something else"  # derived: accepted and recomputed
    assert RunLimits.from_dict(data) == ok
    with pytest.raises(ValidationError):
        RunLimits.from_dict({**ok.to_dict(), "extra": 1})
    with pytest.raises(ValidationError):
        RunLimits.from_dict({"rounds": 3})
    with pytest.raises(ValueError):
        dataclasses.replace(coordinator_input(), limits={"rounds": 3})


# 10
@pytest.mark.parametrize("make", [coordinator_input, feature_engineering_input, modeling_input,
                                  reviewer_input])
def test_old_input_dicts_without_limits_still_parse(make):
    role_input = make()
    data = role_input.to_dict()
    assert "limits" not in data and role_input.limits is None
    again = type(role_input).from_dict(data)
    assert again == role_input and again.limits is None and again.to_dict() == data
    full = type(role_input).from_dict(with_limits(role_input).to_dict())
    assert full.limits == limits()


# 17
def test_the_prompts_name_the_boosting_libraries_their_apis_and_the_bundle_rules():
    from kaggle_agent.loop.writer import BundleRequest  # noqa: F401
    role_prompts = [
        build_coordinator_prompt(with_limits(coordinator_input())),
        build_feature_engineering_prompt(with_limits(feature_engineering_input())),
        build_modeling_prompt(with_limits(modeling_input())),
    ]
    for prompt in role_prompts:
        for name in ("xgboost", "lightgbm", "catboost", "XGBClassifier", "LGBMClassifier",
                     "CatBoostClassifier", "allow_writing_files=False", "BUNDLE_MODEL_DIR",
                     "thread_count", "n_jobs", "seeds"):
            assert name in prompt, name
    for build in (build_coordinator_prompt, build_feature_engineering_prompt, build_modeling_prompt):
        role_input = {build_coordinator_prompt: coordinator_input,
                      build_feature_engineering_prompt: feature_engineering_input,
                      build_modeling_prompt: modeling_input}[build]()
        prompt = build(with_limits(role_input, allowed_libraries=NO_CATBOOST))
        instructions = prompt.split("Input (JSON):")[0]
        assert "XGBClassifier" in instructions and "LGBMClassifier" in instructions
        assert "CatBoostClassifier" not in instructions and "allow_writing_files" not in instructions
        assert "catboost" not in instructions


def test_the_writer_prompt_lists_the_libraries_and_follows_the_configured_list():
    from writer_helpers import make_request, task

    from kaggle_agent.writer.prompt import build_prompt

    full = build_prompt(make_request("x"), task(), ALLOWED_LIBRARIES)
    for name in (*SIX, "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
                 "allow_writing_files=False", "BUNDLE_MODEL_DIR", "thread_count", "seeds"):
        assert name in full, name
    reduced = build_prompt(make_request("x"), task(), NO_CATBOOST)
    assert "allow_writing_files" not in reduced and "CatBoostClassifier" not in reduced
    assert "XGBClassifier" in reduced
    plain = build_prompt(make_request("x"), task(), ("numpy", "pandas", "scikit-learn"))
    assert "Libraries:" not in plain and "XGBClassifier" not in plain
