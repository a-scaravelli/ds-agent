"""Per-fold views, stability, paired comparison, correlation, history helpers (cases 14-17, 25, 35-37)."""

from types import SimpleNamespace

import numpy as np
import pytest
from analytics_helpers import oof_frame

from kaggle_agent.analytics import history
from kaggle_agent.analytics.compare import T95, paired_comparison
from kaggle_agent.analytics.correlation import prediction_correlation
from kaggle_agent.analytics.folds import fold_view, stability
from kaggle_agent.evaluation.threshold import compare as program_compare
from kaggle_agent.evaluation.threshold import dispersion
from kaggle_agent.proposals import ValidationError

SCORES = {"r0/f0": 0.80, "r0/f1": 0.82, "r0/f2": 0.70, "r0/f3": 0.81, "r0/f4": 0.79}


# 14
def test_fold_view_gives_every_point_the_sample_spread_and_the_worst_fold():
    view = fold_view(SCORES, "maximize")
    assert [p["id"] for p in view["points"]] == list(SCORES) and view["count"] == 5
    assert view["spread"] == pytest.approx(np.std(list(SCORES.values()), ddof=1))
    assert view["worst"] == {"id": "r0/f2", "score": 0.70} and view["best"]["id"] == "r0/f1"
    low = fold_view(SCORES, "minimize")
    assert low["worst"]["id"] == "r0/f1" and low["best"]["id"] == "r0/f2"
    assert fold_view({}, "maximize")["points"] == []


# 15
def test_stability_warns_when_a_gain_is_within_the_noise():
    scores = {"r0/f0": 0.800, "r0/f1": 0.800, "r1/f0": 0.804, "r1/f1": 0.804}
    small = stability(scores, "maximize", gain=0.003, bar=0.01)
    assert [r["rep"] for r in small["repetitions"]] == [0, 1]
    assert small["repeat_range"] == pytest.approx(0.004)
    assert "may be noise" in small["warning"] and "bar" in small["warning"]
    assert stability(scores, "maximize", gain=0.05, bar=0.01)["warning"] is None
    only_repeat = stability(scores, "maximize", gain=0.003, bar=0.001)
    assert "repetitions" in only_repeat["warning"]


A = {"p0": 0.70, "p1": 0.72, "p2": 0.71, "p3": 0.73}
B = {"p0": 0.71, "p1": 0.732, "p2": 0.718, "p3": 0.74}
BASELINE = {"p0": 0.5, "p1": 0.7, "p2": 0.6, "p3": 0.8}


# 16
def test_the_paired_comparison_is_exact_and_agrees_with_the_program_rule():
    result = paired_comparison(A, B, BASELINE, "maximize", 1.5)
    diffs = np.array([B[p] - A[p] for p in A])
    assert result["common"] == 4 and result["mean_difference"] == pytest.approx(diffs.mean())
    half = T95[3] * diffs.std(ddof=1) / 2
    assert result["interval"] == pytest.approx([diffs.mean() - half, diffs.mean() + half])
    assert (result["wins"], result["ties"], result["losses"]) == (4, 0, 0)
    protocol = SimpleNamespace(direction="maximize", threshold_k=1.5)
    program = program_compare(B, A, dispersion(BASELINE), protocol)
    assert result["rule"]["bar"] == pytest.approx(1.5 * dispersion(BASELINE)) == pytest.approx(program.threshold)
    assert result["rule"]["is_real"] is program.is_real is False
    # the interval excludes zero but the program's rule says "not real": both are shown with the reason
    assert result["interval"][0] > 0 and result["disagree"] is True
    assert "can therefore disagree" in result["why_they_may_disagree"] and "not real" in result["rule"]["text"]
    assert "above zero" in result["verdict"]


def test_the_direction_flips_the_sign_of_the_differences():
    low = paired_comparison(A, B, BASELINE, "minimize", 1.5)
    assert low["mean_difference"] < 0 and low["losses"] == 4 and "worse" in low["verdict"]


# 17
def test_comparison_edge_cases_common_partitions_ties_and_missing_sides():
    other = {"p2": 0.5, "p3": 0.6, "p9": 0.9}
    partial = paired_comparison(A, other, None, "maximize", None)
    assert partial["common"] == 2 and partial["only_common_used"] is True and partial["rule"] is None
    same = paired_comparison(A, dict(A), BASELINE, "maximize", 1.5)
    assert same["tie"] is True and "tie" in same["verdict"] and same["ties"] == 4
    missing = paired_comparison(A, {}, BASELINE, "maximize", 1.5)
    assert missing["not_comparable"] is True and "no partition in common" in missing["reason"]


# 25
def test_prediction_correlation_is_exact_and_flags_non_overlapping_rows():
    a = oof_frame([1, 2, 3, 4, 5], prediction=[1, 2, 3, 4, 5])
    b = oof_frame([1, 2, 3, 4, 5], prediction=[2, 4, 6, 8, 10])
    out = prediction_correlation(a, b, "regression", (), "values")
    assert out["pearson"] == pytest.approx(1.0) and out["spearman"] == pytest.approx(1.0)
    same = prediction_correlation(a, a.copy(), "regression", (), "values")
    assert same["pearson"] == pytest.approx(1.0)
    ca = oof_frame([0, 1, 1, 1], prediction=[0, 1, 1, 0])
    cb = oof_frame([0, 1, 1, 1], prediction=[0, 1, 0, 0])
    lab = prediction_correlation(ca, cb, "classification", (0, 1), "labels")
    assert lab["label_agreement"] == 0.75 and lab["joint_error_rate"] == 0.25
    assert lab["both_wrong_of_any_wrong"] == 0.5
    assert lab["pearson"] == pytest.approx(np.corrcoef([0, 1, 1, 0], [0, 1, 0, 0])[0, 1])
    far = b.assign(row=b["row"] + 100)
    nope = prediction_correlation(a, far, "regression", (), "values")
    assert nope["not_recorded"] and nope["code"] == "not_comparable"


# 35, 36, 37 (pure helpers; the endpoints are tested in test_analytics_api.py)
ROWS = [
    {"attempt": "a1", "round": "R0", "score": 0.5, "spread": 0.1, "family": "linear", "features": 3, "time": 4.0, "text": "alpha"},
    {"attempt": "a2", "round": "R1", "score": 0.7, "spread": 0.2, "family": "boost", "features": 5, "time": 2.0, "text": "beta"},
    {"attempt": "a3", "round": "R1", "score": None, "spread": None, "family": None, "features": None, "time": 1.0, "text": "gamma"},
]


def test_all_attempts_sort_filter_and_failed_rows_last():
    assert [r["attempt"] for r in history.all_attempts(ROWS, "score", "desc")] == ["a2", "a1", "a3"]
    assert [r["attempt"] for r in history.all_attempts(ROWS, "score", "asc")] == ["a1", "a2", "a3"]
    assert [r["attempt"] for r in history.all_attempts(ROWS, round_id="R1")] == ["a2", "a3"]
    assert [r["attempt"] for r in history.all_attempts(ROWS, family="linear")] == ["a1"]
    assert [r["attempt"] for r in history.all_attempts(ROWS, q="BET")] == ["a2"]
    with pytest.raises(ValidationError):
        history.all_attempts(ROWS, "colour")
    with pytest.raises(ValidationError):
        history.all_attempts(ROWS, "score", "sideways")


def test_best_so_far_follows_the_direction_and_side_by_side_needs_two_or_three_ids():
    points = [{"score": 0.5}, {"score": 0.7}, {"score": 0.6}, {"score": None}]
    assert [p["best"] for p in history.best_so_far(points, "maximize")] == [0.5, 0.7, 0.7, 0.7]
    assert [p["best"] for p in history.best_so_far(points, "minimize")] == [0.5, 0.5, 0.5, 0.5]
    details = {i: {"hypothesis": i} for i in "abcd"}
    assert len(history.side_by_side(["a", "b", "c"], details)) == 3
    for bad in (["a"], ["a", "b", "c", "d"], ["a", "a"]):
        with pytest.raises(ValidationError):
            history.side_by_side(bad, details)
