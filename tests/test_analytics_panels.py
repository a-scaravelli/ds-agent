"""Metric panels by task type and purity of the analytics package (cases 18, 19, 20)."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from analytics_helpers import oof_frame

from kaggle_agent.analytics.classification import classification_panels
from kaggle_agent.analytics.regression import regression_panels

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
TRUTH = [0, 0, 0, 1, 1, 1, 1, 0]
P1 = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6]


# 18
def test_binary_panels_equal_the_hand_computed_values():
    out = classification_panels(oof_frame(TRUTH, proba=P1), (0, 1), "probabilities")
    assert out["confusion"]["matrix"] == [[3, 1], [1, 3]] and out["confusion"]["accuracy"] == 0.75
    assert out["roc"]["auc"] == pytest.approx(13 / 16)
    assert out["roc"]["x"][0] == 0.0 and out["roc"]["y"][-1] == 1.0 and len(out["roc"]["x"]) <= 200
    assert out["pr"]["available"] and 0 < out["pr"]["average_precision"] <= 1
    p, y = np.array(P1), np.array(TRUTH)
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert out["log_loss"]["value"] == pytest.approx(expected)
    cal = out["calibration"]
    assert sum(b["count"] for b in cal["bins"]) == 8
    assert cal["ece"] == pytest.approx(sum(b["count"] / 8 * abs(b["mean_predicted"] - b["fraction_positive"]) for b in cal["bins"]))
    row = min(out["sweep"]["rows"], key=lambda r: abs(r["threshold"] - 0.5))
    assert out["sweep"]["rows"][0]["recall"] == 1.0 and 0 <= row["flagged"] <= 1
    assert out["positive_class"] == "1"
    json.dumps(out)  # JSON-ready


def test_the_sweep_at_a_known_threshold():
    out = classification_panels(oof_frame([0, 1, 1, 0], proba=[0.0, 1.0, 0.5, 0.5]), (0, 1), "probabilities")
    at_half = next(r for r in out["sweep"]["rows"] if r["threshold"] == pytest.approx(0.5))
    assert at_half["precision"] == pytest.approx(2 / 3) and at_half["recall"] == 1.0 and at_half["flagged"] == 0.75


# 18
def test_multiclass_gives_one_vs_rest_curves_and_says_what_is_not_available():
    truth = [0, 1, 2, 0, 1, 2]
    proba = [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6]]
    out = classification_panels(oof_frame(truth, proba=proba), (0, 1, 2), "probabilities")
    assert [c["class"] for c in out["roc"]["one_vs_rest"]] == ["0", "1", "2"]
    assert out["confusion"]["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert out["calibration"]["available"] is False and "binary" in out["calibration"]["reason"]
    many = classification_panels(oof_frame(list(range(11)) * 2, prediction=list(range(11)) * 2), tuple(range(11)), "labels")
    assert many["available"] is False and "up to 10 classes" in many["reason"]


# 18
def test_labels_only_gives_the_confusion_matrix_and_the_reason_for_the_rest():
    out = classification_panels(oof_frame(TRUTH, prediction=[0, 0, 0, 1, 1, 0, 1, 1]), (0, 1), "labels")
    assert out["confusion"]["matrix"] == [[3, 1], [1, 3]]
    for key in ("roc", "pr", "calibration", "log_loss", "sweep"):
        assert out[key]["available"] is False and out[key]["reason"].startswith("not available because")
        assert "labels only" in out[key]["reason"]


def test_a_panel_uses_one_repetition_only():
    frame = oof_frame(TRUTH, proba=P1, reps=2)
    out = classification_panels(frame, (0, 1), "probabilities", rep=1)
    assert out["rows"] == 8 and out["repetition"] == 1


# 19
def test_regression_panels_equal_the_known_residuals():
    truth = np.arange(1, 11, dtype=float)
    pred = truth + np.array([1, -1] * 5)
    out = regression_panels(oof_frame(truth, prediction=pred))
    assert out["bias"] == pytest.approx(0.0) and out["rmse"] == pytest.approx(1.0) and out["mae"] == pytest.approx(1.0)
    assert sum(out["histogram"]["counts"]) == 10 and len(out["by_quantile"]) == 5
    assert all(q["mean_abs_error"] == pytest.approx(1.0) for q in out["by_quantile"])
    assert out["scatter"]["downsampled"] is False
    biased = regression_panels(oof_frame(truth, prediction=truth + 2))
    assert biased["bias"] == pytest.approx(2.0)


# 19
def test_the_scatter_is_deterministically_downsampled_above_the_cap():
    truth = np.arange(1000, dtype=float)
    frame = oof_frame(truth, prediction=truth * 1.1)
    first, second = regression_panels(frame), regression_panels(frame)
    assert first["scatter"]["downsampled"] and len(first["scatter"]["x"]) <= 400
    assert first == second


# 20
def test_the_analytics_package_is_pure_by_scan_and_by_repeated_calls():
    forbidden = {"os", "socket", "urllib", "http", "subprocess", "sqlite3", "pathlib", "shutil", "tempfile",
                 "flask", "requests", "threading", "asyncio"}
    banned_modules = ("registry", "web", "orchestrator", "sandbox", "llm", "kaggle", "loop", "pipeline", "bundle")
    for path in sorted((SRC / "analytics").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [("." * node.level) + (node.module or "")]
            else:
                names = []
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("open", "exec", "eval"):
                    raise AssertionError((path.name, node.lineno))
            for name in names:
                assert name.split(".")[0] not in forbidden, (path.name, name)
                parts = name.split(".")
                if parts[0] == "kaggle_agent":
                    assert parts[1] not in banned_modules, (path.name, name)
                    assert parts[1] != "evaluation" or parts[2:3] in (["threshold"], ["metrics"], ["protocol"]), (path.name, name)
    frame = oof_frame(TRUTH, proba=P1)
    assert classification_panels(frame, (0, 1), "probabilities") == classification_panels(frame.copy(), (0, 1), "probabilities")
