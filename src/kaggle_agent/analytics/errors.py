"""Per-row errors from an OOF table, shared by the segment, hardest-row and flag functions.

Classification with labels or probabilities: the error is the share of repetitions in which the row was
misclassified; with probabilities the log loss of the true class is also given. Regression: absolute and
signed error. Scores-only metrics have no per-row error (``None``).
"""

import numpy as np
import pandas as pd

EPS = 1e-15


def predicted_labels(frame: pd.DataFrame, classes: tuple, input_kind: str):
    """The predicted label of each row, or None when the predictions are not labels or probabilities."""
    if "prediction" in frame.columns:
        if input_kind == "labels":
            return frame["prediction"].to_numpy()
        if input_kind == "probabilities" and len(classes) == 2:
            return np.where(frame["prediction"].to_numpy(dtype=float) >= 0.5, classes[1], classes[0])
        return None
    columns = [c for c in frame.columns if c.startswith("p") and c[1:].isdigit()]
    if input_kind != "probabilities" and input_kind != "scores":
        return None
    values = frame[columns].to_numpy(dtype=float)
    index = values.argmax(axis=1)
    classes_arr = np.asarray(classes, dtype=object)
    if len(classes_arr) != values.shape[1]:
        return None
    return classes_arr[index]


def true_class_probability(frame: pd.DataFrame, classes: tuple, input_kind: str):
    if input_kind != "probabilities":
        return None
    truth = frame["truth"].to_numpy()
    if "prediction" in frame.columns:
        if len(classes) != 2:
            return None
        p1 = frame["prediction"].to_numpy(dtype=float)
        return np.where(truth == classes[1], p1, 1.0 - p1)
    columns = [f"p{i}" for i in range(len(classes))]
    if not all(c in frame.columns for c in columns):
        return None
    values = frame[columns].to_numpy(dtype=float)
    lookup = {c: i for i, c in enumerate(classes)}
    idx = np.array([lookup.get(t, 0) for t in truth])
    return values[np.arange(len(values)), idx]


def row_errors(frame: pd.DataFrame, task: str, classes: tuple, input_kind: str) -> pd.DataFrame | None:
    """One row per training-frame row id, averaged over repetitions; None when no error can be defined.

    Columns: ``error`` (what is ranked), ``truth``, ``prediction`` (last repetition), ``signed`` (regression)
    and ``log_loss`` (probabilities).
    """
    work = frame.sort_values(["row", "rep"], kind="stable").reset_index(drop=True)
    if task == "regression":
        pred = work["prediction"].to_numpy(dtype=float)
        truth = work["truth"].to_numpy(dtype=float)
        work["signed"] = pred - truth
        work["error"] = np.abs(work["signed"])
        work["_pred"] = pred
    else:
        labels = predicted_labels(work, classes, input_kind)
        if labels is None:
            return None
        work["error"] = (labels != work["truth"].to_numpy()).astype(float)
        work["_pred"] = labels
        prob = true_class_probability(work, classes, input_kind)
        if prob is not None:
            work["log_loss"] = -np.log(np.clip(prob, EPS, 1.0))
    aggregate = {"error": "mean", "truth": "last", "_pred": "last"}
    for extra in ("signed", "log_loss"):
        if extra in work.columns:
            aggregate[extra] = "mean"
    out = work.groupby("row", sort=True).agg(aggregate).rename(columns={"_pred": "prediction"})
    return out
