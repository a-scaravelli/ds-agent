"""Regression panels (F4): residuals, error by quantile of the truth, bias, RMSE, MAE."""

import numpy as np
import pandas as pd

MAX_POINTS = 400
BINS = 20
QUANTILES = 5
TEXT = {
    "residuals": "Residual = prediction minus truth. A histogram centred on zero and narrow is good; a skew means a bias.",
    "scatter": "Residual against the predicted value: a funnel or a curve means the model errs in a pattern.",
    "quantiles": "Mean error for each fifth of the truth: models often do worst on the extremes.",
    "bias": "Bias is the mean residual: positive means the model over-predicts on average.",
}


def regression_panels(table: pd.DataFrame, rep: int = 0) -> dict:
    frame = table[table["rep"] == rep] if (table["rep"] == rep).any() else table[table["rep"] == table["rep"].min()]
    truth = frame["truth"].to_numpy(dtype=float)
    pred = frame["prediction"].to_numpy(dtype=float)
    res = pred - truth
    n = len(res)
    if n == 0:
        return {"available": False, "reason": "not available because the table is empty"}
    counts, edges = np.histogram(res, bins=BINS)
    hist = {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}
    order = np.argsort(pred, kind="stable")
    idx = order if n <= MAX_POINTS else order[np.unique(np.linspace(0, n - 1, MAX_POINTS).round().astype(int))]
    scatter = {"x": [float(pred[i]) for i in idx], "y": [float(res[i]) for i in idx], "downsampled": n > MAX_POINTS}
    ranks = pd.Series(truth).rank(method="first")
    which = pd.qcut(ranks, min(QUANTILES, n), labels=False)
    quant = []
    for q in sorted(which.unique()):
        m = (which == q).to_numpy()
        quant.append({"quantile": int(q), "count": int(m.sum()), "truth_min": float(truth[m].min()),
                      "truth_max": float(truth[m].max()), "mean_abs_error": float(np.abs(res[m]).mean()),
                      "bias": float(res[m].mean())})
    return {"available": True, "repetition": int(frame["rep"].iloc[0]), "rows": n,
            "bias": float(res.mean()), "rmse": float(np.sqrt((res ** 2).mean())), "mae": float(np.abs(res).mean()),
            "histogram": hist, "scatter": scatter, "by_quantile": quant, "explanations": TEXT}
