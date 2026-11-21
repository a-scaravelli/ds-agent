"""Paired comparison of two attempts fold by fold (F2), next to the program's own improvement rule."""

from types import SimpleNamespace

import numpy as np

from kaggle_agent.evaluation.threshold import compare as program_compare
from kaggle_agent.evaluation.threshold import dispersion

# two-sided 95% t quantiles by degrees of freedom (small samples are the usual case)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
       18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
       26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
WHY_DISAGREE = (
    "The interval uses the spread of the differences of THIS pair across the folds; the program's rule "
    "compares the mean difference with a bar made from the noise of the BASELINE (threshold_k x its "
    "dispersion). The folds also share training rows, so the interval is optimistic. The two can "
    "therefore disagree: the program's rule is the one that decides."
)


def t_quantile(df: int) -> float:
    return T95.get(df, 1.96 if df > 30 else T95[1])


def paired_comparison(a: dict[str, float], b: dict[str, float], baseline: dict[str, float] | None,
                      direction: str, threshold_k: float | None) -> dict:
    """Compare ``b`` (candidate) against ``a`` (reference) on their common partitions.

    Differences are oriented by the direction: positive means ``b`` is better.
    """
    common = sorted(set(a) & set(b))
    if not common:
        return {"not_comparable": True, "common": 0, "reason": "the two attempts have no partition in common"}
    sign = 1.0 if direction == "maximize" else -1.0
    diffs = [sign * (b[p] - a[p]) for p in common]
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else None
    interval = None
    if sd is not None:
        half = t_quantile(n - 1) * sd / np.sqrt(n)
        interval = [mean - float(half), mean + float(half)]
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    ties = n - wins - losses
    if all(d == 0 for d in diffs):
        verdict = "The two attempts score exactly the same on every common fold: a tie."
    elif interval is None:
        verdict = "There is only one common fold, so no interval can be given."
    elif interval[0] > 0:
        verdict = "B is better than A on average, and the interval stays above zero."
    elif interval[1] < 0:
        verdict = "B is worse than A on average, and the interval stays below zero."
    else:
        verdict = "The difference is within the noise of these folds: the interval includes zero."
    rule = None
    if baseline is not None and threshold_k is not None and len(baseline) >= 2:
        protocol = SimpleNamespace(direction=direction, threshold_k=threshold_k)
        shared = {p: a[p] for p in common}
        result = program_compare({p: b[p] for p in common}, shared, dispersion(baseline), protocol)
        rule = {"bar": result.threshold, "threshold_k": threshold_k, "baseline_dispersion": dispersion(baseline),
                "difference": result.difference, "is_real": result.is_real,
                "text": ("The program's rule says this improvement is "
                         + ("real" if result.is_real else "not real")
                         + f": the mean difference ({result.difference:.4g}) is "
                         + ("above" if result.is_real else "not above")
                         + f" the bar ({result.threshold:.4g} = {threshold_k:g} x baseline dispersion).")}
    disagree = (rule is not None and interval is not None
                and ((interval[0] > 0 and not rule["is_real"]) or (interval[1] < 0 and rule["is_real"])))
    return {"common": n, "partitions": common, "differences": dict(zip(common, diffs)), "mean_difference": mean,
            "interval": interval, "confidence": 0.95, "wins": int(wins), "losses": int(losses), "ties": int(ties),
            "verdict": verdict, "rule": rule, "disagree": bool(disagree), "why_they_may_disagree": WHY_DISAGREE,
            "only_common_used": len(set(a) | set(b)) != n, "tie": all(d == 0 for d in diffs)}
