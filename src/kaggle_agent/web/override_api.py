"""Read API of the final steps, the memory across runs and the overrides (ticket 30); GET only.

Shapes are in ``docs/api.md``. The holdout is sealed until the DS unsealed it; the code of a bundle is never served.
"""

import csv
import io

from flask import Response, abort, request

from kaggle_agent.registry import NotFoundError

from . import (
    bundle_data,
    cycles_data,
    data,
    history_data,
    labels_data,
    report_export,
    store,
    submission_data,
)

FORMULA = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value) -> str:
    """A cell for a CSV a spreadsheet may open: text that looks like a formula gets a leading quote."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(FORMULA) else text


def register(get, source, ok) -> None:
    @get("/runs/<path:run_id>/labels")
    def labels(run_id):
        src = source()
        return ok({"labels": labels_data.labels(src, run_id), "choice": labels_data.choice(src, run_id)}, src)

    @get("/runs/<path:run_id>/bar-override")
    def bar_override(run_id):
        src = source()
        return ok(labels_data.bar_panel(src, run_id), src)

    @get("/runs/<path:run_id>/origin")
    def origin(run_id):
        src = source()
        src.registry.get_cycle(run_id)
        entry = cycles_data.entry_of(src.project_dir, run_id)
        carry = store.read_json(store.project_json_file(src.project_dir, "carry", run_id)) if entry else None
        return ok({"run_id": run_id, "entry": entry, "carry": (carry or {}).get("carry") if entry else None}, src)

    @get("/runs/<path:run_id>/bundle/files/<path:name>")
    def bundle_file(run_id, name):
        src = source()
        try:
            blob, sha = bundle_data.download(src, run_id, name)
        except NotFoundError:
            abort(404)
        response = Response(blob, mimetype="application/octet-stream")
        response.headers["Content-Disposition"] = f'attachment; filename="{name.rsplit("/", 1)[-1]}"'
        response.headers["X-Content-SHA256"] = sha
        return response

    @get("/runs/<path:run_id>/bundle")
    def bundle(run_id):
        src = source()
        return ok(bundle_data.bundle_view(src, run_id), src)

    @get("/runs/<path:run_id>/report.md")
    def report_markdown(run_id):
        src = source()
        text = report_export.markdown(report_export.context(src, run_id))
        response = Response(text, mimetype="text/markdown")
        response.headers["Content-Disposition"] = f'attachment; filename="report-{run_id}.md"'
        return response

    @get("/runs/<path:run_id>/submissions/preview")
    def submissions_preview(run_id):
        src = source()
        return ok(submission_data.preview_view(src, run_id), src)

    @get("/runs/<path:run_id>/gap")
    def gap(run_id):
        src = source()
        return ok(submission_data.gap(src, run_id), src)

    @get("/history")
    def history():
        src = source()
        return ok(history_data.history(src), src)

    @get("/learnings")
    def learnings():
        src = source()
        args = request.args
        try:
            offset, limit = int(args.get("offset", 0)), int(args.get("limit", history_data.PAGE))
        except ValueError:
            abort(400)
        return ok(history_data.learnings(src, args.get("q"), offset, limit), src)

    @get("/overrides/pending")
    def overrides_pending():
        src = source()
        return ok({"overrides": labels_data.override_statuses(src)}, src)

    @get("/jobs/<job_id>/predictions.csv")
    def predictions_csv(job_id):
        src = source()
        job = store.latest_job(src.project_dir, ("predict",))
        record = None
        for path in store.list_jobs(src.project_dir):
            candidate = store.read_json(path)
            if isinstance(candidate, dict) and candidate.get("id") == job_id and candidate.get("kind") == "predict":
                record = candidate
        if record is None or job is None:
            abort(404)
        result = store.read_json(store.job_paths(src.project_dir, job_id)[1])
        if not isinstance(result, dict) or not result.get("ok"):
            abort(404)
        out = io.StringIO()
        writer = csv.writer(out)
        ids, preds = result.get("ids"), result["predictions"]
        header = ([result["id_column"]] if ids else []) + ["prediction"]
        writer.writerow(header)
        for index, pred in enumerate(preds):
            row = [safe_cell(ids[index])] if ids else []
            writer.writerow(row + [safe_cell(pred if not isinstance(pred, list) else " ".join(map(str, pred)))])
        response = Response(out.getvalue(), mimetype="text/csv")
        response.headers["Content-Disposition"] = f'attachment; filename="predictions-{job_id}.csv"'
        return response
