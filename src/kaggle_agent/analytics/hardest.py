"""Hardest rows (F6): the worst OOF rows with their development features."""

import pandas as pd

from kaggle_agent.proposals import ValidationError

MAX_LIMIT = 200
MAX_FEATURE_COLUMNS = 30
BASE = ("row", "error", "truth", "prediction")


def hardest_rows(features: pd.DataFrame, errors: pd.DataFrame, sort: str = "error", order: str = "desc",
                 limit: int = 50) -> dict:
    """The ``limit`` worst rows (by log loss when probabilities exist, else error), then sorted by ``sort``."""
    if order not in ("asc", "desc"):
        raise ValidationError("order", "must be asc or desc")
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValidationError("limit", f"must be between 1 and {MAX_LIMIT}")
    columns = [str(c) for c in list(features.columns)[:MAX_FEATURE_COLUMNS]]
    valid = (*BASE, *columns)
    if sort not in valid:
        raise ValidationError("sort", "must be one of the returned columns")
    key = "log_loss" if "log_loss" in errors.columns else "error"
    common = features.index.intersection(errors.index)
    worst = errors.loc[common].rename_axis("row").reset_index().sort_values([key, "row"], ascending=[False, True], kind="stable").head(limit)
    rows = []
    for _, r in worst.iterrows():
        feat = features.loc[int(r["row"])]
        item = {"row": int(r["row"]), "error": float(r[key]), "truth": _plain(r["truth"]),
                "prediction": _plain(r["prediction"])}
        for c in features.columns[:MAX_FEATURE_COLUMNS]:
            item[str(c)] = _plain(feat[c])
        rows.append(item)
    present = [x for x in rows if x[sort] is not None]
    absent = [x for x in rows if x[sort] is None]
    try:
        present.sort(key=lambda x: x[sort], reverse=order == "desc")
    except TypeError:
        present.sort(key=lambda x: str(x[sort]), reverse=order == "desc")
    return {"columns": list(valid), "rows": present + absent, "ranked_by": key, "sort": sort, "order": order,
            "limit": limit,
            "explanation": "The rows the model got most wrong out of fold (development data only). Look for what they have in common."}


def _plain(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v if isinstance(v, (int, float, bool, str)) else str(v)
