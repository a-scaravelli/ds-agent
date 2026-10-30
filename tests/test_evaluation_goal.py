"""Goal criteria: evaluate_goal, several metrics in the frozen protocol, candidate interface."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from evaluation_helpers import (
    MajorityCandidate,
    TemporaryMetric,
    classification_frame,
    evaluate,
    frozen_engine,
    goal_dict,
    make_protocol,
)
from sklearn import metrics as skm

import kaggle_agent.evaluation as evaluation
from kaggle_agent.evaluation import METRICS, EvaluationEngine, evaluate_goal
from kaggle_agent.proposals import GoalCriteria, ValidationError

EVAL_DIR = Path(evaluation.__file__).parent
BUILTIN = {"rmse", "mae", "r2", "accuracy", "balanced_accuracy", "f1", "log_loss",
           "roc_auc", "average_precision"}


def _goal(*criteria):
    return GoalCriteria.from_dict(goal_dict(*criteria))


def test_goal_met_only_when_every_criterion_is_met_and_is_deterministic():
    goal = _goal(("accuracy", 0.8, "maximize"), ("f1", 0.5, "maximize"))
    for _ in range(3):
        both = evaluate_goal(goal, {"accuracy": 0.9, "f1": 0.6})
        assert both.met and [r.met for r in both.results] == [True, True]
        one = evaluate_goal(goal, {"accuracy": 0.9, "f1": 0.4})
        assert not one.met and [r.met for r in one.results] == [True, False]
        equal = evaluate_goal(goal, {"accuracy": 0.8, "f1": 0.5})
        assert equal.met
    assert both == evaluate_goal(goal, {"accuracy": 0.9, "f1": 0.6})
    result = one.results[1]
    assert (result.metric, result.value, result.threshold, result.direction) == (
        "f1", 0.4, 0.5, "maximize")
    minimize = _goal(("log_loss", 0.4, "minimize"))
    assert evaluate_goal(minimize, {"log_loss": 0.3}).met
    assert evaluate_goal(minimize, {"log_loss": 0.4}).met
    assert not evaluate_goal(minimize, {"log_loss": 0.41}).met
    with pytest.raises(ValidationError, match="f1"):
        evaluate_goal(goal, {"accuracy": 0.9})


def test_goal_with_registered_metrics_is_scored_on_the_validation_predictions():
    protocol = make_protocol(goal=goal_dict(("accuracy", 0.5, "maximize"),
                                            ("balanced_accuracy", 0.4, "maximize")))
    engine, reg = frozen_engine(protocol)
    frame = classification_frame()
    result = evaluate(engine, MajorityCandidate(), frame)
    names = {m.partition for m in result.execution.metrics}
    assert {"goal/accuracy", "goal/balanced_accuracy", "validation"} <= names
    assert set(result.goal_scores) == {"accuracy", "balanced_accuracy"}
    assert result.goal_scores["accuracy"] == result.scores  # accuracy is also the main metric
    values = EvaluationEngine.goal_values_from_execution(result.execution)
    for name, by_partition in result.goal_scores.items():
        assert values[name] == pytest.approx(np.mean(list(by_partition.values())))
    assert values["balanced_accuracy"] == pytest.approx(0.5)  # a majority vote
    by_name = {m.partition: m for m in result.execution.metrics}
    assert by_name["goal/accuracy"].mean == pytest.approx(result.mean)
    assert by_name["goal/balanced_accuracy"].dispersion == pytest.approx(0.0, abs=1e-9)
    assert result.goal.met is True and len(result.goal.results) == 2
    assert result.goal == evaluate_goal(protocol.goal, values)
    again = evaluate(engine, MajorityCandidate(), frame, "A2")
    assert again.goal == result.goal and again.goal_scores == result.goal_scores
    assert skm.balanced_accuracy_score([0, 1], [0, 0]) == 0.5
    assert EvaluationEngine.scores_from_execution(result.execution) == result.scores


def test_a_goal_metric_registered_by_a_test_works_and_is_removed_afterwards():
    before = set(METRICS)
    with TemporaryMetric() as name:
        protocol = make_protocol(goal=goal_dict((name, 0.0, "maximize")))
        engine, _ = frozen_engine(protocol)
        result = evaluate(engine, MajorityCandidate(), classification_frame())
        assert f"goal/{name}" in {m.partition for m in result.execution.metrics}
        assert result.goal.met
    assert set(METRICS) == before
    with pytest.raises(ValidationError, match="registered") as exc:
        make_protocol(goal=goal_dict(("never_registered", 0.5, "maximize")))
    assert "accuracy" in str(exc.value)
    assert "precision" not in METRICS and "recall" not in METRICS


def test_goal_direction_and_input_kind_are_validated():
    with pytest.raises(ValidationError, match="goal"):
        make_protocol(goal=goal_dict(("accuracy", 0.8, "minimize")))
    with pytest.raises(ValidationError, match="goal") as exc:
        make_protocol(goal=goal_dict(("log_loss", 0.4, "minimize")))
    assert "labels" in str(exc.value) and "probabilities" in str(exc.value)
    with TemporaryMetric("test_other", input_kind="scores"):
        with pytest.raises(ValidationError, match="goal"):
            make_protocol(goal=goal_dict(("test_other", 0.5, "maximize")))
    with pytest.raises(ValidationError):
        make_protocol(goal={"criteria": []})


def test_candidate_may_be_any_object_with_fit_predict():
    class Minimal:
        def fit_predict(self, x_train, y_train, x_valid, seed):
            return np.zeros(len(x_valid), dtype=int)

    class NoMethod:
        pass

    engine, _ = frozen_engine()
    frame = classification_frame()
    assert evaluate(engine, Minimal(), frame).mean > 0
    with pytest.raises(AttributeError, match="fit_predict"):
        evaluate(engine, NoMethod(), frame, "A2")
    with engine.holdout_session(frame) as holdout:
        assert holdout.evaluate_holdout(Minimal(), "A1").rows > 0
    from kaggle_agent.evaluation import Candidate
    from kaggle_agent.evaluation.candidate import Candidate as FromModule
    from kaggle_agent.evaluation.engine import Candidate as FromEngine

    assert Candidate is FromModule is FromEngine
    assert list(inspect.signature(Candidate.fit_predict).parameters) == [
        "self", "x_train", "y_train", "x_valid", "seed"]


def test_evaluation_imports_no_tests_llm_or_fake_names():
    for path in EVAL_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                root = module.split(".")
                assert root[0] not in {"tests", "evaluation_helpers", "registry_helpers"}, path
                assert root[:2] != ["kaggle_agent", "llm"], path
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                assert "fake" not in node.name.lower(), f"{path.name}: {node.name}"


def test_no_builtin_metric_was_added_for_goals():
    assert set(METRICS) == BUILTIN
