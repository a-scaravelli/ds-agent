"""Metric registry: names, declared directions, extensibility (no network)."""

import numpy as np
import pytest
from evaluation_helpers import make_protocol

from kaggle_agent.evaluation import METRICS, MetricSpec, get_metric, metrics, register_metric
from kaggle_agent.proposals import ValidationError


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(metrics, "METRICS", dict(METRICS))


def test_each_metric_declares_natural_direction_and_protocol_must_agree():
    assert get_metric("rmse").direction == "minimize"
    assert get_metric("accuracy").direction == "maximize"
    assert all(spec.direction in ("maximize", "minimize") for spec in METRICS.values())
    with pytest.raises(ValidationError) as exc:
        make_protocol(metric="rmse", direction="maximize")
    assert "rmse" in str(exc.value) and "minimize" in str(exc.value)
    with pytest.raises(ValidationError) as exc:
        make_protocol(metric="accuracy", direction="minimize")
    assert "maximize" in str(exc.value)
    assert make_protocol(metric="rmse", direction="minimize").direction == "minimize"
    assert make_protocol(metric="accuracy", direction="maximize").metric == "accuracy"


@pytest.mark.parametrize("direction", [None, "", "up"])
def test_registering_a_metric_requires_a_valid_direction(isolated_registry, direction):
    with pytest.raises(ValidationError) as exc:
        register_metric(MetricSpec("custom_x", lambda a, b: 0.0, "values", direction))
    assert exc.value.field == "direction"
    assert "custom_x" not in metrics.METRICS
    register_metric(MetricSpec("custom_x", lambda a, b: 0.0, "values", "minimize"))
    assert make_protocol(metric="custom_x", direction="minimize").metric == "custom_x"
    with pytest.raises(ValidationError):
        make_protocol(metric="custom_x", direction="maximize")


def test_metrics_by_name_from_extensible_registry_not_callables(isolated_registry):
    with pytest.raises(ValidationError) as exc:
        make_protocol(metric="not_registered")
    assert exc.value.field == "metric"
    with pytest.raises(ValidationError):
        make_protocol(metric=lambda y, p: 0.0)
    register_metric(MetricSpec("not_registered", lambda a, b: 1.0, "values", "maximize"))
    assert make_protocol(metric="not_registered").metric == "not_registered"
    with pytest.raises(ValidationError):
        register_metric(MetricSpec("not_registered", lambda a, b: 1.0, "values", "maximize"))


def test_metric_params_checked_against_the_spec():
    assert make_protocol(metric="f1", metric_params={"average": "macro"}).metric == "f1"
    with pytest.raises(ValidationError):
        make_protocol(metric="f1", metric_params={"bogus": 1})
    with pytest.raises(ValidationError):
        make_protocol(metric="accuracy", metric_params={"average": "macro"})


def test_building_block_metrics_compute_known_values():
    y = np.array([0, 1, 1, 0])
    p = np.array([0, 1, 0, 0])
    assert get_metric("accuracy").score(y, p, {}) == 0.75
    assert get_metric("mae").score(y, p, {}) == 0.25
    assert get_metric("rmse").score(y, p, {}) == pytest.approx(0.5)
    assert get_metric("f1").score(y, p, {"average": "binary"}) == pytest.approx(2 / 3)
    assert get_metric("roc_auc").score(y, np.array([0.1, 0.9, 0.8, 0.2]), {}) == 1.0
