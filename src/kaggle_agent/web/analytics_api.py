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


def valid_attempt_id(value: str) -> str:
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
    @get("/attempts/<path:attempt_id>/analytics")
    def a_overview(attempt_id):
        src = source()
        return ok(ad.overview(src, valid_attempt_id(attempt_id)), src)

    @get("/attempts/<path:attempt_id>/folds")
    def a_folds(attempt_id):
        src = source()
        return ok(ad.fold_scores(src, valid_attempt_id(attempt_id)), src)

    @get("/attempts/<path:attempt_id>/panels")
    def a_panels(attempt_id):
        src = source()
        return ok(ad.panels(src, valid_attempt_id(attempt_id), _int("rep", 0, 0, 1000)), src)

    @get("/attempts/<path:attempt_id>/segments")
    def a_segments(attempt_id):
        src = source()
        return ok(ad.segment_table(src, valid_attempt_id(attempt_id), _int("min_size", 1, 1, 10 ** 7),
                                   _flag("interactions", True), _int("top", 20, 1, 100)), src)

    @get("/attempts/<path:attempt_id>/hardest")
    def a_hardest(attempt_id):
        src = source()
        order = _text("order") or "desc"
        return ok(ad.hardest_rows(src, valid_attempt_id(attempt_id), _text("sort") or "error", order, _int("limit", 50, 1, 10 ** 6)), src)

    @get("/attempts/<path:attempt_id>/flags")
    def a_flags(attempt_id):
        src = source()
        return ok(ad.sanity_flags(src, valid_attempt_id(attempt_id)), src)

    @get("/attempts/<path:attempt_id>/features")
    def a_features(attempt_id):
        src = source()
        return ok(ad.features(src, valid_attempt_id(attempt_id)), src)

    @get("/runs/<path:run_id>/paired")
    def r_paired(run_id):
        src = source()
        a, b = request.args.get("a"), request.args.get("b")
        if not a or not b:
            raise ValidationError("a", "give both attempts: ?a=ID&b=ID")
        return ok(ad.paired(src, run_id, valid_attempt_id(a), valid_attempt_id(b)), src)

    @get("/runs/<path:run_id>/history")
    def r_history(run_id):
        src = source()
        return ok(ad.history(src, run_id), src)

    @get("/runs/<path:run_id>/compare")
    def r_compare(run_id):
        src = source()
        return ok(ad.compare_table(src, run_id, _text("sort") or "attempt", _text("order") or "asc", _text("q"),
                                   _text("round"), _text("family")), src)

    @get("/runs/<path:run_id>/side-by-side")
    def r_side(run_id):
        src = source()
        ids = [i for i in (request.args.get("ids") or "").split(",") if i]
        for i in ids:
            valid_attempt_id(i)
        return ok(ad.side_by_side(src, run_id, ids), src)

    @get("/runs/<path:run_id>/ablation")
    def r_ablation(run_id):
        src = source()
        return ok(ad.ablation(src, run_id), src)
