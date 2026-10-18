"""Builders and fake candidates shared by the evaluation tests (all offline)."""

import numpy as np
import pandas as pd
from registry_helpers import make_cycle, new_attempt

from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.registry import SqliteRegistry


def classification_frame(n=240, seed=0):
    """Binary target (about 20% positives), 40 groups and a shuffled time column."""
    rng = np.random.default_rng(seed)
    t = rng.permutation(n)
    return pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "group": rng.integers(0, 40, size=n),
        "t": t,
        "y": (rng.random(n) < 0.2).astype(int),
    })


def regression_frame(n=120, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x1": x, "t": rng.permutation(n), "y": 2 * x + rng.normal(size=n)})


def protocol_dict(**overrides):
    data = {
        "metric": "accuracy",
        "metric_params": {},
        "direction": "maximize",
        "target": "y",
        "validation_scheme": "random",
        "validation_params": {"n_folds": 3, "stratify_by": "y"},
        "holdout_scheme": "random",
        "holdout_params": {"stratify_by": "y"},
        "holdout_share": 0.2,
        "seed": 11,
        "repetitions": 1,
        "threshold_k": 2.0,
    }
    data.update(overrides)
    return data


def make_protocol(**overrides):
    return Protocol.from_dict(protocol_dict(**overrides))


def chrono_protocol(**overrides):
    base = {
        "validation_scheme": "chronological",
        "validation_params": {"time_column": "t", "n_folds": 3},
        "holdout_scheme": "chronological",
        "holdout_params": {"time_column": "t"},
    }
    return make_protocol(**{**base, **overrides})


class MajorityCandidate:
    """Predicts the most frequent training label."""

    def __init__(self):
        self.calls = []

    def fit_predict(self, x_train, y_train, x_valid, seed):
        self.calls.append((x_train, y_train, x_valid, seed))
        return np.full(len(x_valid), y_train.mode().iloc[0])


class MeanCandidate:
    def fit_predict(self, x_train, y_train, x_valid, seed):
        return np.full(len(x_valid), y_train.mean())


class SeedNoiseCandidate:
    """Training mean plus noise drawn from the seed, so the score depends on the seed."""

    def fit_predict(self, x_train, y_train, x_valid, seed):
        noise = np.random.default_rng(seed).normal(scale=0.5, size=len(x_valid))
        return y_train.mean() + noise


class FailingCandidate:
    def fit_predict(self, x_train, y_train, x_valid, seed):
        raise RuntimeError("boom")


class OffsetCandidate:
    """Training mean plus a fixed offset: a controllable 'better or worse' model."""

    def __init__(self, offset):
        self.offset = offset

    def fit_predict(self, x_train, y_train, x_valid, seed):
        return np.full(len(x_valid), y_train.mean() + self.offset)


def new_registry(attempts=("A1", "A2", "A3")):
    reg = SqliteRegistry()
    reg.add_cycle(make_cycle("C1"))
    for attempt in attempts:
        new_attempt(reg, attempt)
    return reg


def frozen_engine(protocol=None, reg=None, **kwargs):
    reg = reg or new_registry()
    engine = EvaluationEngine(reg, "C1", protocol or make_protocol(), **kwargs)
    engine.freeze()
    return engine, reg


def evaluate(engine, candidate, frame, attempt_id="A1", n=1):
    return engine.evaluate_validation(
        candidate, frame, attempt_id=attempt_id, execution_id=f"{attempt_id}-E{n}",
        config={"model": "fake"}, code_hash="code", data_hash="data",
    )
