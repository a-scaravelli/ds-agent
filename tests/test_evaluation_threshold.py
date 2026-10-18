"""Threshold: k times the baseline dispersion, paired, oriented by direction (no network)."""

import numpy as np
import pytest
from evaluation_helpers import make_protocol

from kaggle_agent.evaluation import compare, dispersion
from kaggle_agent.proposals import ValidationError

BASELINE = {"p1": 0.0, "p2": 2.0}  # sample std = sqrt(2)
REF = {"p1": 1.0, "p2": 1.0}


def _cand(diff):
    return {p: v + diff for p, v in REF.items()}


def test_threshold_applied_only_strictly_above_k_times_dispersion():
    protocol = make_protocol(threshold_k=2.0)
    d = dispersion(BASELINE)
    threshold = 2.0 * d
    below = compare(_cand(threshold - 1e-6), REF, d, protocol)
    equal = compare(_cand(threshold), REF, d, protocol)
    above = compare(_cand(threshold + 1e-6), REF, d, protocol)
    assert [below.is_real, equal.is_real, above.is_real] == [False, False, True]
    assert above.threshold == pytest.approx(threshold)


def test_k_comes_from_the_protocol():
    d = dispersion(BASELINE)
    candidate = _cand(1.0)
    outcomes = [
        compare(candidate, REF, d, make_protocol(threshold_k=k)).is_real for k in (0.5, 0.7, 1.0)
    ]
    assert outcomes == [True, True, False]
    assert compare(candidate, REF, d, make_protocol(threshold_k=0)).threshold == 0


def test_paired_on_same_partitions_with_reference_from_caller():
    protocol = make_protocol(threshold_k=1.0)
    with pytest.raises(ValidationError):
        compare({"p1": 1.0, "p3": 2.0}, REF, 0.1, protocol)
    ref = {"p1": 0.5, "p2": 0.1}  # not the baseline
    result = compare({"p1": 0.7, "p2": 0.5}, ref, 0.25, protocol)
    assert result.per_partition == pytest.approx({"p1": 0.2, "p2": 0.4})
    assert result.difference == pytest.approx(0.3)
    assert result.threshold == pytest.approx(0.25)
    assert result.is_real


def test_metric_direction_is_respected():
    low = {"p1": 0.5, "p2": 0.5}
    high = {"p1": 1.0, "p2": 1.0}
    minimize = make_protocol(metric="rmse", direction="minimize", threshold_k=0)
    maximize = make_protocol(metric="accuracy", direction="maximize", threshold_k=0)
    better = compare(low, high, 0.0, minimize)
    worse = compare(low, high, 0.0, maximize)
    assert better.is_real and better.difference == pytest.approx(0.5)
    assert not worse.is_real and worse.difference == pytest.approx(-0.5)


def test_dispersion_is_sample_standard_deviation():
    scores = [0.5, 0.7, 0.6, 0.9, 0.8, 0.4]
    assert dispersion(scores) == pytest.approx(np.std(scores, ddof=1))
    assert dispersion(dict(enumerate(scores))) == pytest.approx(np.std(scores, ddof=1))
    with pytest.raises(ValidationError):
        dispersion([0.5])
