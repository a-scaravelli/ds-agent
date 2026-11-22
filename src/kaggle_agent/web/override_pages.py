"""HTML pages of the final steps, the memory across runs and the overrides (ticket 30); GET only.

They call the same functions as the JSON API. The pages that write (unseal, choose model, bar override, steering,
'Try this', re-run, reproduce, predict, send, fork) only post to the guarded endpoints of ``override_writes``.
"""

from flask import current_app, render_template, request

from kaggle_agent.proposals import FOCUSES, MODEL_FAMILIES

from . import (
    bundle_data,
    cycles_data,
    data,
    history_data,
    labels_data,
    report_export,
    submission_data,
)


def _source() -> data.Source:
    return current_app.config["CURRENT_SOURCE"]()


def _page(template: str, src, run=None, **context):
    return render_template(template, run=run, warnings=src.warnings, **context)


def register(app) -> None:
    def get(rule: str):
        return app.route(rule, methods=["GET"], provide_automatic_options=False)

    @get("/cycles/<path:cycle_id>/report/print")
    def report_print(cycle_id):
        src = _source()
        ctx = report_export.context(src, cycle_id)
        return render_template("report_print.html", print_view=True, warnings=[], **ctx)

    @get("/cycles/<path:cycle_id>/bundle")
    def bundle_page(cycle_id):
        src = _source()
        return _page("bundle.html", src, data.run_row(src, cycle_id), bundle=bundle_data.bundle_view(src, cycle_id))

    @get("/cycles/<path:cycle_id>/predict")
    def predict_page(cycle_id):
        src = _source()
        return _page("predict.html", src, data.run_row(src, cycle_id), bundle=bundle_data.bundle_view(src, cycle_id))

    @get("/cycles/<path:cycle_id>/submission")
    def submission_page(cycle_id):
        src = _source()
        return _page("submission.html", src, data.run_row(src, cycle_id), sub=submission_data.preview_view(src, cycle_id))

    @get("/cycles/<path:cycle_id>/gap")
    def gap_page(cycle_id):
        src = _source()
        return _page("gap.html", src, data.run_row(src, cycle_id), gap=submission_data.gap(src, cycle_id))

    @get("/history")
    def history_page():
        src = _source()
        return _page("history.html", src, runs=history_data.history(src))

    @get("/learnings")
    def learnings_page():
        src = _source()
        q = request.args.get("q") or ""
        project = current_app.config["PROJECT_DIR"]
        info = data.read_project_file(project) or {}
        return _page("learnings.html", src, found=history_data.learnings(src, q), q=q,
                     competition=info.get("competition"))

    @get("/fork")
    def fork_page():
        src = _source()
        project = current_app.config["PROJECT_DIR"]
        runs = data.runs(src)
        active = cycles_data.active_cycle_id(project) if project else None
        attempts = {r["id"]: [{"id": a["id"], "outcome": a["outcome"], "mean": a["mean"],
                               "hypothesis": a["hypothesis"]} for a in data.attempts(src, r["id"])
                              if a["outcome"] == "succeeded"] for r in runs}
        state = next((r["state"] for r in runs if r["id"] == active), None)
        return _page("fork.html", src, runs=runs, attempts=attempts, active=active, active_state=state,
                     entries=cycles_data.entries(project) if project else [])


def override_context(src, run_id: str, round_id: str | None = None) -> dict:
    """What the override panels of the round and attempt pages need."""
    return {"labels": labels_data.labels(src, run_id), "choice": labels_data.choice(src, run_id),
            "bar": labels_data.bar_panel(src, run_id), "steering_now": labels_data.steering_now(src, run_id),
            "focuses": FOCUSES, "families": MODEL_FAMILIES,
            "overrides": [o for o in labels_data.override_statuses(src) if o["cycle"] == run_id],
            "is_active": cycles_data.active_cycle_id(src.project_dir) == run_id}
