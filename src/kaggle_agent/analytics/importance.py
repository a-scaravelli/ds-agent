"""Importance grouped by feature family, and the display-only ablation table (F7, F8)."""

import numpy as np


def group_importance(importances: dict[str, float], families: dict[str, str]) -> list[dict]:
    """Sum of absolute importances per family with each feature's share; largest family first."""
    total = float(sum(abs(v) for v in importances.values())) or 0.0
    groups: dict[str, dict] = {}
    for feature, value in importances.items():
        fam = families.get(feature, feature)
        g = groups.setdefault(fam, {"family": fam, "importance": 0.0, "features": []})
        g["importance"] += abs(float(value))
        g["features"].append({"feature": feature, "importance": float(value)})
    out = list(groups.values())
    for g in out:
        g["share"] = (g["importance"] / total) if total else None
        g["features"].sort(key=lambda f: (-abs(f["importance"]), f["feature"]))
    out.sort(key=lambda g: (-g["importance"], g["family"]))
    return out


def ablation_table(attempts: list[dict], families: dict[str, str]) -> dict:
    """For each family: score and spread of attempts that use it and attempts that do not.

    ``attempts`` are ``{attempt, score, spread, features}`` where ``features`` is None when the bundle gave
    no describe.json (counted as 'not provided', never guessed).
    """
    unknown = [a["attempt"] for a in attempts if a.get("features") is None]
    known = [a for a in attempts if a.get("features") is not None and a.get("score") is not None]
    rows = []
    for fam in sorted(set(families.values())):
        with_, without = [], []
        for a in known:
            (with_ if any(families.get(f, f) == fam for f in a["features"]) else without).append(a)
        if not with_ or not without:
            continue
        rows.append({"family": fam, "with": _summary(with_), "without": _summary(without)})
    return {"rows": rows, "not_provided": unknown,
            "note": "Display only: it compares attempts that happen to differ, not a controlled experiment."}


def _summary(items: list[dict]) -> dict:
    spreads = [a["spread"] for a in items if a.get("spread") is not None]
    return {"attempts": [a["attempt"] for a in items], "count": len(items),
            "mean_score": float(np.mean([a["score"] for a in items])),
            "mean_spread": float(np.mean(spreads)) if spreads else None}
