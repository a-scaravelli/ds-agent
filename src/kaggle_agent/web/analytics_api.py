"""JSON endpoints of the performance analytics (ticket 29), all GET; the shapes are in ``docs/api.md``.

Registered on the existing /api blueprint by ``api.py`` so the envelope, the error handlers and the GET-only rule
are the same. Parameters are validated and bounded (400 on a bad value, 404 for an unknown id). Development data
only: no endpoint touches the holdout. The one write of this ticket, ``POST /api/jobs/importance``, lives in
``writes.py`` with the CSRF, Origin and audit rules of the other jobs.
"""

import re

from flask import request

from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import NotFoundError

from . import analytics_data as ad

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def attempt_id(value: str) -> str:
    if not ID.fullmatch(value or ""):
        raise NotFoundError(f"unknown attempt {str(value)[:40]!r}")
    return value


def _int(name: str, default: int, low: int, high: int) -> int:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValidationError(name, "must be a whole number") from None
    if not low <= value <= high:
        raise ValidationError(name, f"must be between {low} and {high}")
    return value


def _flag(name: str, default: bool) -> bool:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    if raw.lower() in ("1", "true", "yes"):
        return True
    if raw.lower() in ("0", "false", "no"):
        return False
    raise ValidationError(name, "must be 0 or 1")


def _text(name: str, limit: int = 100) -> str | None:
    raw = request.args.get(name)
    if raw in (None, ""):
        return None
    if len(raw) > limit:
        raise ValidationError(name, f"must be at most {limit} characters")
    return raw


def register(get, source, ok) -> None:
    @get("/attempts/<path:aid>/analytics")
    def a_overview(aid):
        src = source()
        return ok(ad.overview(src, attempt_id(aid)), src)

    @get("/attempts/<path:aid>/folds")
    def a_folds(aid):
        src = source()
        return ok(ad.fold_scores(src, attempt_id(aid)), src)

    @get("/attempts/<path:aid>/panels")
    def a_panels(aid):
        src = source()
        return ok(ad.panels(src, attempt_id(aid), _int("rep", 0, 0, 1000)), src)

    @get("/attempts/<path:aid>/segments")
    def a_segments(aid):
        src = source()
        return ok(ad.segment_table(src, attempt_id(aid), _int("min_size", 1, 1, 10 ** 7),
                                   _flag("interactions", True), _int("top", 20, 1, 100)), src)

    @get("/attempts/<path:aid>/hardest")
    def a_hardest(aid):
        src = source()
        order = _text("order") or "desc"
        return ok(ad.hardest_rows(src, attempt_id(aid), _text("sort") or "error", order, _int("limit", 50, 1, 10 ** 6)), src)

    @get("/attempts/<path:aid>/flags")
    def a_flags(aid):
        src = source()
        return ok(ad.sanity_flags(src, attempt_id(aid)), src)

    @get("/attempts/<path:aid>/features")
    def a_features(aid):
        src = source()
        return ok(ad.features(src, attempt_id(aid)), src)

    @get("/runs/<path:rid>/paired")
    def r_paired(rid):
        src = source()
        a, b = request.args.get("a"), request.args.get("b")
        if not a or not b:
            raise ValidationError("a", "give both attempts: ?a=ID&b=ID")
        return ok(ad.paired(src, rid, attempt_id(a), attempt_id(b)), src)

    @get("/runs/<path:rid>/history")
    def r_history(rid):
        src = source()
        return ok(ad.history(src, rid), src)

    @get("/runs/<path:rid>/compare")
    def r_compare(rid):
        src = source()
        return ok(ad.compare_table(src, rid, _text("sort") or "attempt", _text("order") or "asc", _text("q"),
                                   _text("round"), _text("family")), src)

    @get("/runs/<path:rid>/side-by-side")
    def r_side(rid):
        src = source()
        ids = [i for i in (request.args.get("ids") or "").split(",") if i]
        for i in ids:
            attempt_id(i)
        return ok(ad.side_by_side(src, rid, ids), src)

    @get("/runs/<path:rid>/ablation")
    def r_ablation(rid):
        src = source()
        return ok(ad.ablation(src, rid), src)
