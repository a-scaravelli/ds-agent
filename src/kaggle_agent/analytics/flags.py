"""Leakage and sanity flags (F11): heuristics on the development data and the OOF scores. Not proof."""

import numpy as np
import pandas as pd

from .words import HEURISTIC

ID_SHARE = 0.99
CORR = 0.98
PURITY = 0.99
JUMP_MULTIPLE = 5.0
MIN_ROWS = 20


def _flag(kind: str, subject: str, text: str) -> dict:
    return {"kind": kind, "subject": subject, "text": f"{text} {HEURISTIC}"}


def column_flags(features: pd.DataFrame, target: pd.Series) -> list[dict]:
    """ID-like columns and features almost identical to the target."""
    out, n = [], len(features)
    if n < MIN_ROWS:
        return out
    classes = target.nunique()
    numeric_target = pd.api.types.is_numeric_dtype(target) and (classes > 2 or True)
    codes = pd.Series(pd.factorize(target.astype(str), sort=True)[0], index=target.index)
    for name in features.columns:
        col = features[name]
        distinct = col.nunique(dropna=True)
        discrete = not pd.api.types.is_float_dtype(col)
        if discrete and distinct >= ID_SHARE * n:
            out.append(_flag("id_like", str(name), f"'{name}' has a different value in almost every row, like an identifier; "
                             "a model that uses it can memorise rows instead of learning."))
            continue
        hit = None
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            ref = target.astype(float) if pd.api.types.is_numeric_dtype(target) else (codes.astype(float) if classes == 2 else None)
            if ref is not None and col.std() > 0 and ref.std() > 0:
                r = abs(float(np.corrcoef(col.fillna(col.median()), ref)[0, 1]))
                if r >= CORR:
                    hit = f"its correlation with the target is {r:.3f}"
        if hit is None and distinct <= max(50, n // 20) and classes <= 50:
            frame = pd.DataFrame({"c": col.astype(object).where(col.notna(), "(missing)"), "t": target.astype(str)})
            purity = float(frame.groupby("c")["t"].agg(lambda s: s.value_counts().iloc[0]).sum() / n)
            if purity >= PURITY and distinct > 1 and classes > 1:
                hit = f"knowing its value gives the target in {purity:.1%} of the rows"
        if hit:
            out.append(_flag("target_like", str(name), f"'{name}' is almost identical to the target: {hit}; it may leak the answer."))
    return out


def score_flags(history: list[dict], attempt: str, direction: str, bar: float | None, perfect: bool = False) -> list[dict]:
    """A suspicious jump over the previous best (more than 5 x the bar) and a perfect out-of-fold result."""
    out = []
    sign = 1.0 if direction == "maximize" else -1.0
    best = None
    for point in history:
        if point["attempt"] == attempt:
            score = point.get("score")
            if score is not None and best is not None and bar and sign * (score - best) > JUMP_MULTIPLE * bar:
                out.append(_flag("jump", attempt, f"The score jumped by {sign * (score - best):.4g} over the previous best, "
                                 f"more than {JUMP_MULTIPLE:g} times the bar ({bar:.4g}); big jumps are often leaks."))
            break
        s = point.get("score")
        if s is not None and (best is None or sign * (s - best) > 0):
            best = s
    if perfect:
        out.append(_flag("perfect", attempt, "The out-of-fold predictions are perfect (no error at all); real data rarely allows it."))
    return out


def is_perfect(errors: pd.DataFrame | None) -> bool:
    return errors is not None and len(errors) > 0 and bool((errors["error"] == 0).all())
