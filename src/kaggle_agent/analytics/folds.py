"""Per-partition scores (F1) and stability across repetitions (F10). Pure functions over plain dicts."""

import re

import numpy as np

_PARTITION = re.compile(r"r(\d+)/f(\d+)")


def split_id(partition_id: str) -> tuple[int, int]:
    match = _PARTITION.fullmatch(partition_id)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _sorted(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: split_id(item[0]))


def _std(values) -> float | None:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else None


def fold_view(scores: dict[str, float], direction: str) -> dict:
    """Every partition, the mean, the sample spread and the worst (and best) one for the direction."""
    items = _sorted(scores)
    if not items:
        return {"points": [], "mean": None, "spread": None, "worst": None, "best": None, "count": 0}
    points = [{"id": pid, "rep": split_id(pid)[0], "fold": split_id(pid)[1], "score": float(v)}
              for pid, v in items]
    key = (lambda p: p["score"])
    worst = min(points, key=key) if direction == "maximize" else max(points, key=key)
    best = max(points, key=key) if direction == "maximize" else min(points, key=key)
    values = [p["score"] for p in points]
    return {"points": points, "mean": float(np.mean(values)), "spread": _std(values),
            "worst": {"id": worst["id"], "score": worst["score"]},
            "best": {"id": best["id"], "score": best["score"]}, "count": len(points),
            "direction": direction}


def stability(scores: dict[str, float], direction: str, gain: float | None = None,
              bar: float | None = None) -> dict:
    """Score by repetition and the repeat-to-repeat range; a warning when ``gain`` is within the noise.

    ``gain`` is the improvement over the previous best (positive = better). The noise is the larger of the
    program's bar and the repeat-to-repeat range of this attempt.
    """
    by_rep: dict[int, list[float]] = {}
    for pid, value in _sorted(scores):
        by_rep.setdefault(split_id(pid)[0], []).append(float(value))
    reps = [{"rep": r, "mean": float(np.mean(v)), "folds": len(v)} for r, v in sorted(by_rep.items())]
    means = [r["mean"] for r in reps]
    spread = float(max(means) - min(means)) if len(means) > 1 else None
    warning = None
    if gain is not None:
        reasons = []
        if bar is not None and gain < bar:
            reasons.append(f"smaller than the program's bar ({bar:.4g})")
        if spread is not None and gain < spread:
            reasons.append(f"smaller than the difference between repetitions ({spread:.4g})")
        if reasons:
            warning = (f"The gain over the previous best ({gain:.4g}) is " + " and ".join(reasons)
                       + ": it may be noise, not a real improvement.")
    return {"repetitions": reps, "repeat_range": spread, "gain": gain, "bar": bar, "warning": warning,
            "note": ("Each repetition reshuffles the folds (a new seed). If the repetitions disagree by "
                     "more than a gain, the gain is not trustworthy.") if reps else None}
