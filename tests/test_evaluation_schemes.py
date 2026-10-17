"""Partition schemes: random, group, chronological, extensible registry (no network)."""

import numpy as np
import pytest
from evaluation_helpers import chrono_protocol, classification_frame, make_protocol

from kaggle_agent.evaluation import SCHEMES, FrozenSplit, SplitScheme, schemes
from kaggle_agent.proposals import ValidationError

FRAME = classification_frame()


def _rows(frame, positions):
    return frame.iloc[np.asarray(positions)]


def test_random_scheme_with_and_without_stratification():
    scheme = SCHEMES["random"]
    overall = FRAME["y"].mean()
    folds = scheme.folds(FRAME, {"n_folds": 4, "stratify_by": "y"}, 3)
    for _, valid in folds:
        assert abs(_rows(FRAME, valid)["y"].mean() - overall) < 0.03
    plain = scheme.folds(FRAME, {"n_folds": 4}, 3)
    for results in (folds, plain):
        valids = np.concatenate([v for _, v in results])
        assert sorted(valids) == list(range(len(FRAME)))
        for train, valid in results:
            assert not set(train) & set(valid)
            assert len(train) + len(valid) == len(FRAME)
    dev, hold = scheme.holdout_split(FRAME, 0.25, {"stratify_by": "y"}, 5)
    assert abs(_rows(FRAME, hold)["y"].mean() - overall) < 0.03
    assert sorted(np.concatenate([dev, hold])) == list(range(len(FRAME)))


def test_group_scheme_never_shares_a_group():
    scheme = SCHEMES["group"]
    dev, hold = scheme.holdout_split(FRAME, 0.25, {"group_column": "group"}, 2)
    assert not set(FRAME.iloc[dev]["group"]) & set(FRAME.iloc[hold]["group"])
    dev_frame = FRAME.iloc[dev].reset_index(drop=True)
    for train, valid in scheme.folds(dev_frame, {"group_column": "group", "n_folds": 4}, 2):
        assert not set(dev_frame.iloc[train]["group"]) & set(dev_frame.iloc[valid]["group"])
    split = FrozenSplit.create(FRAME, make_protocol(
        validation_scheme="group", validation_params={"group_column": "group", "n_folds": 3},
        holdout_scheme="group", holdout_params={"group_column": "group"},
    ))
    held = set(FRAME["group"]) - set(FRAME.iloc[split.development_rows()]["group"])
    assert held


def test_chronological_scheme_validation_and_holdout_are_later():
    scheme = SCHEMES["chronological"]
    dev, hold = scheme.holdout_split(FRAME, 0.2, {"time_column": "t"}, 0)
    assert FRAME.iloc[dev]["t"].max() < FRAME.iloc[hold]["t"].min()
    dev_frame = FRAME.iloc[dev].reset_index(drop=True)
    folds = scheme.folds(dev_frame, {"time_column": "t", "n_folds": 3}, 0)
    assert len(folds) == 3
    for train, valid in folds:
        assert dev_frame.iloc[train]["t"].max() < dev_frame.iloc[valid]["t"].min()
    assert [len(t) for t, _ in folds] == sorted(len(t) for t, _ in folds)


def test_chronological_ties_stay_on_one_side():
    frame = FRAME.assign(t=np.repeat(np.arange(24), 10))
    dev, hold = SCHEMES["chronological"].holdout_split(frame, 0.2, {"time_column": "t"}, 0)
    assert frame.iloc[dev]["t"].max() < frame.iloc[hold]["t"].min()
    for train, valid in SCHEMES["chronological"].folds(
        frame.iloc[dev].reset_index(drop=True), {"time_column": "t", "n_folds": 3}, 0
    ):
        assert frame.iloc[dev].iloc[train]["t"].max() < frame.iloc[dev].iloc[valid]["t"].min()


def test_no_scheme_is_registered_as_default_and_determinism_is_declared():
    assert set(SCHEMES) >= {"random", "group", "chronological"}
    assert SCHEMES["chronological"].deterministic is True
    assert SCHEMES["random"].deterministic is False
    assert SCHEMES["group"].deterministic is False


def test_scheme_registry_is_extensible_and_checks_names_params_and_columns(monkeypatch):
    monkeypatch.setattr(schemes, "SCHEMES", dict(SCHEMES))

    class Halves(SplitScheme):
        name = "halves_test"
        deterministic = True

        def validate_params(self, params, role):
            schemes.check_params(params, role, ())

        def n_folds(self, params):
            return 2

        def holdout_split(self, frame, share, params, seed):
            n = len(frame)
            return np.arange(n // 2), np.arange(n // 2, n)

        def folds(self, frame, params, seed):
            n = len(frame)
            half = np.arange(n // 2), np.arange(n // 2, n)
            return [half, half[::-1]]

    with pytest.raises(ValidationError):
        make_protocol(validation_scheme="halves_test", validation_params={})
    schemes.register_scheme(Halves())
    protocol = make_protocol(validation_scheme="halves_test", validation_params={},
                             holdout_scheme="halves_test", holdout_params={})
    assert len(FrozenSplit.create(FRAME, protocol).partitions()) == 2
    with pytest.raises(ValidationError):
        schemes.register_scheme(Halves())
    with pytest.raises(ValidationError):
        make_protocol(validation_scheme="halves_test", validation_params={"extra": 1})

    for bad in (
        {"validation_params": {"n_folds": 3, "bogus": 1}},
        {"validation_params": {}},
        {"validation_params": {"n_folds": 1}},
        {"holdout_params": {"n_folds": 3}},
    ):
        with pytest.raises(ValidationError):
            make_protocol(**bad)
    for bad in (
        make_protocol(holdout_params={"stratify_by": "missing"}),
        chrono_protocol(validation_params={"time_column": "nope", "n_folds": 3}),
        make_protocol(
            holdout_scheme="group", holdout_params={"group_column": "missing"}),
    ):
        with pytest.raises(ValidationError):
            FrozenSplit.create(FRAME, bad)
    with pytest.raises(ValidationError):
        FrozenSplit.create(FRAME.assign(s="a"), chrono_protocol(
            validation_params={"time_column": "s", "n_folds": 3},
            holdout_params={"time_column": "s"}))
