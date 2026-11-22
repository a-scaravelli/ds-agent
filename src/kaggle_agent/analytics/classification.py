"""Classification panels (F4) over one repetition of an OOF table; JSON-ready, with plain-words texts."""

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from .errors import predicted_labels
from .tasks import MAX_CLASSES, panel_support

MAX_POINTS = 200
BINS = 10
THRESHOLDS = 21
TEXT = {
    "confusion": "Rows are the true class, columns the predicted one. The diagonal is what the model got right.",
    "roc": "ROC: how many true positives are found (up) against false alarms (right). The area (AUC) is 1 for a perfect ranking and 0.5 for chance.",
    "pr": "Precision-recall: of the rows flagged positive, how many are right (precision) against how many positives are found (recall). Better than ROC for rare classes.",
    "calibration": "Calibration: when the model says 70%, is it right about 70% of the time? Points near the diagonal mean trustworthy probabilities.",
    "log_loss": "Log loss punishes confident wrong answers. Lower is better.",
    "sweep": "What happens to precision, recall, F1 and accuracy if the decision threshold moves.",
}


def _cut(x, y, limit=MAX_POINTS):
    if len(x) <= limit:
        return [float(v) for v in x], [float(v) for v in y]
    idx = np.unique(np.linspace(0, len(x) - 1, limit).round().astype(int))
    return [float(x[i]) for i in idx], [float(y[i]) for i in idx]


def _na(key, reason):
    return {"available": False, "reason": "not available because " + reason, "explanation": TEXT[key]}


def _positive_score(frame, classes):
    """Score of the positive class (the larger one) for a binary table, or None."""
    if "prediction" in frame.columns:
        return frame["prediction"].to_numpy(dtype=float)
    if len(classes) == 2 and "p1" in frame.columns:
        return frame["p1"].to_numpy(dtype=float)
    return None


def _binary(frame, classes, support):
    truth = (frame["truth"].to_numpy() == classes[1]).astype(int)
    score = _positive_score(frame, classes)
    out = {}
    if support["roc"] is True and score is not None and 0 < truth.sum() < len(truth):
        fpr, tpr, _ = skm.roc_curve(truth, score)
        x, y = _cut(fpr, tpr)
        out["roc"] = {"available": True, "x": x, "y": y, "auc": float(skm.roc_auc_score(truth, score)),
                      "explanation": TEXT["roc"]}
        precision, recall, _ = skm.precision_recall_curve(truth, score)
        x, y = _cut(recall, precision)
        out["pr"] = {"available": True, "x": x, "y": y,
                     "average_precision": float(skm.average_precision_score(truth, score)), "explanation": TEXT["pr"]}
    else:
        why = "the table has one class only" if support["roc"] is True else support["roc"]
        out["roc"], out["pr"] = _na("roc", why), _na("pr", why)
    if support["calibration"] is True and score is not None:
        edges = np.linspace(0, 1, BINS + 1)
        which = np.clip(np.digitize(score, edges[1:-1]), 0, BINS - 1)
        bins, ece = [], 0.0
        for b in range(BINS):
            mask = which == b
            if mask.any():
                mean_p, frac = float(score[mask].mean()), float(truth[mask].mean())
                bins.append({"bin": b, "count": int(mask.sum()), "mean_predicted": mean_p, "fraction_positive": frac})
                ece += mask.mean() * abs(mean_p - frac)
        out["calibration"] = {"available": True, "bins": bins, "ece": float(ece), "explanation": TEXT["calibration"]}
    else:
        out["calibration"] = _na("calibration", support["calibration"] if support["calibration"] is not True else "no probabilities")
    if support["log_loss"] is True and score is not None:
        out["log_loss"] = {"available": True, "value": float(skm.log_loss(truth, np.clip(score, 1e-15, 1 - 1e-15), labels=[0, 1])),
                           "explanation": TEXT["log_loss"]}
    else:
        out["log_loss"] = _na("log_loss", support["log_loss"] if support["log_loss"] is not True else "no probabilities")
    if support["sweep"] is True and score is not None:
        lo, hi = float(score.min()), float(score.max())
        rows = []
        for t in np.linspace(lo, hi, THRESHOLDS):
            pred = (score >= t).astype(int)
            tp = int(((pred == 1) & (truth == 1)).sum())
            fp = int(((pred == 1) & (truth == 0)).sum())
            fn = int(((pred == 0) & (truth == 1)).sum())
            prec = tp / (tp + fp) if tp + fp else None
            rec = tp / (tp + fn) if tp + fn else None
            f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
            rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1,
                         "accuracy": float((pred == truth).mean()), "flagged": float(pred.mean())})
        out["sweep"] = {"available": True, "rows": rows, "explanation": TEXT["sweep"]}
    else:
        out["sweep"] = _na("sweep", support["sweep"] if support["sweep"] is not True else "no scores")
    return out


def _multiclass(frame, classes, support):
    out = {}
    cols = [f"p{i}" for i in range(len(classes))]
    if support["roc"] is not True or not all(c in frame.columns for c in cols):
        why = support["roc"] if support["roc"] is not True else "there are no per-class columns"
        return {k: _na(k, why) for k in ("roc", "pr", "calibration", "log_loss", "sweep")}
    values = frame[cols].to_numpy(dtype=float)
    truth = frame["truth"].to_numpy()
    roc, pr = [], []
    for i, c in enumerate(classes):
        y = (truth == c).astype(int)
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = skm.roc_curve(y, values[:, i])
            x, yy = _cut(fpr, tpr)
            roc.append({"class": str(c), "x": x, "y": yy, "auc": float(skm.roc_auc_score(y, values[:, i]))})
            p, r, _ = skm.precision_recall_curve(y, values[:, i])
            x, yy = _cut(r, p)
            pr.append({"class": str(c), "x": x, "y": yy, "average_precision": float(skm.average_precision_score(y, values[:, i]))})
    out["roc"] = {"available": True, "one_vs_rest": roc, "explanation": TEXT["roc"] + " One curve per class against the rest."}
    out["pr"] = {"available": True, "one_vs_rest": pr, "explanation": TEXT["pr"] + " One curve per class against the rest."}
    if support["log_loss"] is True:
        out["log_loss"] = {"available": True, "explanation": TEXT["log_loss"],
                           "value": float(skm.log_loss(truth, np.clip(values, 1e-15, 1), labels=list(classes)))}
    else:
        out["log_loss"] = _na("log_loss", support["log_loss"])
    out["calibration"] = _na("calibration", "calibration is shown for binary targets only")
    out["sweep"] = _na("sweep", "the threshold sweep is shown for binary targets only")
    return out


def classification_panels(table: pd.DataFrame, classes: tuple, input_kind: str, rep: int = 0) -> dict:
    """All classification panels from one repetition (so a row is not counted twice)."""
    support = panel_support(input_kind)
    frame = table[table["rep"] == rep] if (table["rep"] == rep).any() else table[table["rep"] == table["rep"].min()]
    if len(classes) < 2 and "prediction" not in frame.columns:
        return {"available": False, "reason": "not available because the data has one class only"}
    labels_all = sorted(frame["truth"].unique().tolist()) if len(classes) < 2 else list(classes)
    if len(labels_all) > MAX_CLASSES:
        return {"available": False, "reason": "not available because this panel is for binary or up to 10 classes only"}
    if "prediction" in frame.columns and len(classes) < 2:
        classes = tuple(labels_all)
    panels = {"repetition": int(frame["rep"].iloc[0]) if len(frame) else rep, "rows": int(len(frame)),
              "classes": [str(c) for c in classes], "positive_class": str(classes[-1]) if len(classes) else None}
    predicted = predicted_labels(frame, classes, input_kind)
    if support["confusion"] is True and predicted is not None:
        truth = frame["truth"].to_numpy()
        matrix = [[int(((truth == t) & (predicted == p)).sum()) for p in classes] for t in classes]
        panels["confusion"] = {"available": True, "labels": [str(c) for c in classes], "matrix": matrix,
                               "accuracy": float((truth == predicted).mean()), "explanation": TEXT["confusion"]}
    else:
        why = support["confusion"] if support["confusion"] is not True else "no labels"
        panels["confusion"] = _na("confusion", why)
    other = _binary(frame, classes, support) if len(classes) == 2 else _multiclass(frame, classes, support)
    panels.update(other)
    panels["available"] = True
    return panels
