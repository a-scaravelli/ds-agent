"""All-attempts table, score history and side-by-side view (G1-G3) from plain dicts; no code is shown."""

from kaggle_agent.proposals import ValidationError

SORT_KEYS = ("score", "spread", "round", "family", "features", "time", "attempt")
MAX_SIDE = 3


def all_attempts(rows: list[dict], sort: str = "attempt", order: str = "asc", q: str | None = None,
                 round_id: str | None = None, family: str | None = None, direction: str = "maximize") -> list[dict]:
    """Filter and sort rows of ``{attempt, score, spread, round, family, features, time, status, text}``.

    Rows without a score (failed attempts) always come last.
    """
    if sort not in SORT_KEYS:
        raise ValidationError("sort", f"must be one of {', '.join(SORT_KEYS)}")
    if order not in ("asc", "desc"):
        raise ValidationError("order", "must be asc or desc")
    out = [r for r in rows
           if (not round_id or r.get("round") == round_id) and (not family or r.get("family") == family)
           and (not q or q.lower() in str(r.get("text", "") + " " + str(r.get("attempt"))).lower())]
    key = {"attempt": "attempt"}.get(sort, sort)
    have = [r for r in out if r.get(key) is not None]
    lack = [r for r in out if r.get(key) is None]
    have.sort(key=lambda r: (r[key], str(r["attempt"])), reverse=order == "desc")
    return have + lack


def best_so_far(points: list[dict], direction: str) -> list[dict]:
    """``points`` in attempt order with a ``score``; adds ``best`` (the best score up to and including it)."""
    best, out = None, []
    for p in points:
        s = p.get("score")
        if s is not None and (best is None or (s > best if direction == "maximize" else s < best)):
            best = s
        out.append({**p, "best": best})
    return out


def side_by_side(ids: list[str], details: dict[str, dict]) -> list[dict]:
    """2 or 3 attempts: hypothesis, features, hyperparameters and per-fold scores (no code, no paths)."""
    if not 2 <= len(ids) <= MAX_SIDE or len(set(ids)) != len(ids):
        raise ValidationError("ids", f"give 2 to {MAX_SIDE} different attempt ids")
    return [{"attempt": i, **{k: details[i].get(k) for k in
             ("hypothesis", "what_changed", "features", "derived_features", "hyperparameters", "folds",
              "score", "spread", "describe")}} for i in ids]
