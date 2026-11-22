"""HTML pages of the performance analytics (ticket 29), GET only. The shell and the registry-only data render at
once; the heavy sections are placeholders that ``analytics.js`` fills from the GET API, so a slow section never
blocks rendering. A section whose input is not recorded renders its 'Not recorded for this attempt: <reason>' state
here, without waiting for the browser."""

from flask import current_app, render_template, request

from kaggle_agent.analytics.words import is_not_recorded, not_recorded
from kaggle_agent.proposals import ValidationError

from . import analytics_api, analytics_data as ad, data


def _source() -> data.Source:
    return current_app.config["CURRENT_SOURCE"]()


def _recorded(ov: dict, kind: str):
    """None when the file of ``kind`` is recorded, else the not_recorded object."""
    item = ov["recorded"].get(kind)
    if item is None:
        return not_recorded("not_in_index")
    return item if is_not_recorded(item) else None


def _describe_state(src, ov) -> dict:
    bundle = ov.get("bundle_id")
    if not bundle:
        return ov.get("not_recorded") or not_recorded("no_bundle")
    return ad._describe(src, bundle)


def register(app) -> None:
    def get(rule: str):
        return app.route(rule, methods=["GET"], provide_automatic_options=False)

    @get("/attempts/<path:attempt_id>/analytics")
    def analytics_attempt(attempt_id):
        src = _source()
        attempt_id = analytics_api.valid_attempt_id(attempt_id)
        ov = ad.overview(src, attempt_id)
        run = data.run_row(src, ov["run_id"])
        folds = ad.fold_scores(src, attempt_id)
        oof_bad = ov.get("not_recorded") or _recorded(ov, "oof")
        dev_bad = ov.get("not_recorded") or _recorded(ov, "development")
        task = ov.get("task") or {}
        label_only = task.get("input_kind") == "labels"
        describe = _describe_state(src, ov)
        return render_template(
            "analytics_attempt.html", run=run, warnings=src.warnings, a_id=attempt_id, ov=ov, folds=folds,
            oof_bad=oof_bad, dev_bad=dev_bad, label_only=label_only, describe=describe, task=task,
            can_importance=not ov.get("not_recorded") and bool(ov.get("bundle_id")),
            importance_done=not is_not_recorded(ov["recorded"].get("importance", {"not_recorded": True})))

    @get("/runs/<path:run_id>/compare")
    def analytics_compare(run_id):
        src = _source()
        run = data.run_row(src, run_id)
        args = request.args
        sort, order = args.get("sort") or "attempt", args.get("order") or "asc"
        table = ad.compare_table(src, run_id, sort, order, args.get("q") or None, args.get("round") or None,
                                 args.get("family") or None)
        ids = [i for i in (args.get("ids") or "").split(",") if i]
        side, side_error = None, None
        if ids:
            try:
                for i in ids:
                    analytics_api.valid_attempt_id(i)
                side = ad.side_by_side(src, run_id, ids)
            except ValidationError as exc:
                side_error = str(exc)
        return render_template(
            "analytics_compare.html", run=run, warnings=src.warnings, run_id=run_id, table=table, sort=sort,
            order=order, q=args.get("q") or "", round_id=args.get("round") or "", family=args.get("family") or "",
            ablation=ad.ablation(src, run_id), side=side, side_error=side_error, ids=",".join(ids))

    @get("/runs/<path:run_id>/paired")
    def analytics_paired(run_id):
        src = _source()
        run = data.run_row(src, run_id)
        a, b = request.args.get("a") or "", request.args.get("b") or ""
        attempts = [r["attempt"] for r in ad._attempt_rows(src, run_id) if r["status"] == "succeeded"]
        result = None
        if a and b:
            result = ad.paired(src, run_id, analytics_api.valid_attempt_id(a), analytics_api.valid_attempt_id(b),
                               with_correlation=False)
        return render_template("analytics_paired.html", run=run, warnings=src.warnings, run_id=run_id, a=a, b=b,
                               result=result, attempts=attempts)
