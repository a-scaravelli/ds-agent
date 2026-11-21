"""Error breakdown by segment (F5): single features and two-feature interactions.

``build_segments`` computes the full ranked table once; ``filter_segments`` only applies ``min_size`` and
``top``, so a slider is cheap and the ranking of the rest never changes.
"""

import numpy as np
import pandas as pd

MAX_BINS = 8
MAX_CATEGORIES = 20
MAX_COLUMNS = 100
MAX_PAIR_FEATURES = 8
PAIR_BINS = 4
MISSING = "(missing)"
OTHER = "(other)"


def _is_numeric(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def bin_column(col: pd.Series, max_bins: int = MAX_BINS, max_categories: int = MAX_CATEGORIES) -> pd.Series:
    """Segment label of each row: quantile bins (numeric), categories (top N + other), missing apart."""
    missing = col.isna()
    if _is_numeric(col) and col.nunique(dropna=True) > max_bins:
        cut = pd.qcut(col, max_bins, duplicates="drop")
        labels = cut.astype(object).map(lambda iv: f"[{iv.left:.4g}, {iv.right:.4g}]" if isinstance(iv, pd.Interval) else None)
    else:
        text = col.astype(object).where(~missing, None)
        top = col.value_counts().index[:max_categories if max_bins == MAX_BINS else max_bins - 1]
        keep = set(top.tolist())
        labels = text.map(lambda v: None if v is None else (str(v) if v in keep else OTHER))
    return labels.where(~missing, MISSING)


def _segments_of(labels: pd.Series, error: pd.Series, overall: float, name: str, signed=None) -> list[dict]:
    rows = []
    frame = pd.DataFrame({"s": labels, "e": error})
    if signed is not None:
        frame["g"] = signed
    for value, group in frame.groupby("s", sort=True, dropna=False):
        mean = float(group["e"].mean())
        item = {"feature": name, "segment": str(value), "count": int(len(group)), "error": mean,
                "lift": (mean / overall) if overall > 0 else None}
        if signed is not None:
            item["mean_signed_error"] = float(group["g"].mean())
        rows.append(item)
    return rows


def _rank(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (-(r["lift"] if r["lift"] is not None else -1), -r["count"], r["feature"], r["segment"]))


def build_segments(features: pd.DataFrame, errors: pd.DataFrame, task: str, interactions: bool = True) -> dict:
    """Ranked single-feature and interaction segments over rows present in both tables."""
    common = features.index.intersection(errors.index)
    feats, err = features.loc[common], errors.loc[common]
    error = err["error"]
    overall = float(error.mean()) if len(error) else 0.0
    signed = err["signed"] if "signed" in err.columns else None
    columns = list(feats.columns)[:MAX_COLUMNS]
    binned, singles = {}, []
    for name in columns:
        if feats[name].nunique(dropna=False) <= 1:
            continue
        binned[name] = bin_column(feats[name])
        singles.extend(_segments_of(binned[name], error, overall, str(name), signed))
    pairs = []
    if interactions and len(binned) >= 2:
        best: dict[str, float] = {}
        for s in singles:
            if s["lift"] is not None and s["count"] >= 5:
                best[s["feature"]] = max(best.get(s["feature"], 0.0), s["lift"])
        top = sorted(best, key=lambda f: (-best[f], f))[:MAX_PAIR_FEATURES]
        small = {f: bin_column(feats[f], PAIR_BINS, PAIR_BINS) for f in top}
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                joined = small[a].astype(str) + " & " + small[b].astype(str)
                pairs.extend(_segments_of(joined, error, overall, f"{a} x {b}", signed))
    return {"overall_error": overall, "rows": int(len(error)), "segments": _rank(singles),
            "interactions": _rank(pairs), "task": task,
            "caps": {"columns": MAX_COLUMNS, "bins": MAX_BINS, "pair_features": MAX_PAIR_FEATURES, "pair_bins": PAIR_BINS}}


def filter_segments(table: dict, min_size: int = 1, top: int = 20) -> dict:
    """Keep segments of at least ``min_size`` rows (ranking unchanged) and the first ``top`` of each list."""
    def keep(rows):
        return [r for r in rows if r["count"] >= min_size][:top]
    return {**table, "segments": keep(table["segments"]), "interactions": keep(table["interactions"]),
            "min_size": min_size}
