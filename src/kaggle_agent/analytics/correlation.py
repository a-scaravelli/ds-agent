"""Prediction correlation of two attempts (F12), from their OOF tables aligned by repetition and row."""

import numpy as np
import pandas as pd

from .errors import predicted_labels, row_errors
from .words import not_recorded


def _numeric(frame: pd.DataFrame, classes: tuple) -> np.ndarray | None:
    if "prediction" in frame.columns:
        try:
            return frame["prediction"].to_numpy(dtype=float)
        except (TypeError, ValueError):
            return None
    return frame["p1"].to_numpy(dtype=float) if "p1" in frame.columns and len(classes) == 2 else None


def prediction_correlation(a: pd.DataFrame, b: pd.DataFrame, task: str, classes: tuple, input_kind: str) -> dict:
    """Pearson and Spearman of the predictions, label agreement and joint-error rate on the common rows."""
    merged = a.merge(b, on=["rep", "row"], suffixes=("_a", "_b"))
    if merged.empty:
        return not_recorded("not_comparable")
    parts = {}
    for side in ("a", "b"):
        cols = [c for c in merged.columns if c.endswith("_" + side)]
        part = merged[["rep", "row", *cols]].rename(columns={c: c[:-2] for c in cols})
        parts[side] = part
    pa, pb = _numeric(parts["a"], classes), _numeric(parts["b"], classes)
    out = {"rows": int(len(merged)), "pearson": None, "spearman": None, "label_agreement": None,
           "joint_error_rate": None, "both_wrong_of_any_wrong": None}
    if pa is not None and pb is not None and len(pa) > 1:
        if np.std(pa) == 0 or np.std(pb) == 0:
            out["pearson"] = 1.0 if np.array_equal(pa, pb) else None
            out["spearman"] = out["pearson"]
        else:
            out["pearson"] = float(np.corrcoef(pa, pb)[0, 1])
            ra, rb = pd.Series(pa).rank().to_numpy(), pd.Series(pb).rank().to_numpy()
            out["spearman"] = float(np.corrcoef(ra, rb)[0, 1])
    if task == "classification":
        la = predicted_labels(parts["a"], classes, input_kind)
        lb = predicted_labels(parts["b"], classes, input_kind)
        if la is not None and lb is not None:
            truth = parts["a"]["truth"].to_numpy()
            out["label_agreement"] = float(np.mean(la == lb))
            wrong_a, wrong_b = la != truth, lb != truth
            out["joint_error_rate"] = float(np.mean(wrong_a & wrong_b))
            either = wrong_a | wrong_b
            out["both_wrong_of_any_wrong"] = float(np.mean((wrong_a & wrong_b)[either])) if either.any() else None
    else:
        err_a = np.abs(parts["a"]["prediction"].to_numpy(dtype=float) - parts["a"]["truth"].to_numpy(dtype=float))
        err_b = np.abs(parts["b"]["prediction"].to_numpy(dtype=float) - parts["b"]["truth"].to_numpy(dtype=float))
        if len(err_a) > 1 and np.std(err_a) > 0 and np.std(err_b) > 0:
            out["error_correlation"] = float(np.corrcoef(err_a, err_b)[0, 1])
    out["text"] = ("Two attempts whose predictions are highly correlated make the same mistakes: averaging them "
                   "adds little. Low correlation with similar scores suggests they are worth combining.")
    return out
