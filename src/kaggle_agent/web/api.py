"""Read-only JSON API under ``/api`` (shape documented in ``docs/api.md``).

Every answer is ``{"api_version": 1, "data": ..., "warnings": [...]}``; an error is
``{"api_version": 1, "error": {"status", "title", "message"}}`` (see ``app.py``). GET only. The
HTML pages call the same ``data.py`` functions. There is no endpoint for bundle files, code or
diffs: the web never reads a bundle folder.
"""

from importlib import metadata

from flask import Blueprint, current_app, jsonify, request

from . import data, live

API_VERSION = 1
blueprint = Blueprint("api", __name__, url_prefix="/api")


def _source() -> data.Source:
    return current_app.config["CURRENT_SOURCE"]()


def _ok(payload, src: data.Source):
    return jsonify({"api_version": API_VERSION, "data": payload, "warnings": src.warnings})


def get(rule: str):
    return blueprint.route(rule, methods=["GET"], provide_automatic_options=False)


def _filters() -> dict:
    return {k: request.args.get(k) for k in ("outcome", "round", "min_score", "max_score", "q")}


@get("/meta")
def meta():
    src = _source()
    journal = current_app.config["JOURNAL_MODE"]
    mode = journal() if journal else None
    try:
        version = metadata.version("kaggle-agent")
    except metadata.PackageNotFoundError:
        version = None
    return _ok({
        "api_version": API_VERSION,
        "version": version,
        "registry": {"mode": "read-only", "journal_mode": mode,
                     "wal": None if mode is None else mode == "wal"},
        "project_file": data.read_project_file(src.project_dir) is not None,
    }, src)


@get("/runs")
def runs():
    src = _source()
    return _ok(data.runs(src, request.args.get("q")), src)


@get("/runs/<path:run_id>/chart")
def chart(run_id):
    src = _source()
    return _ok(data.chart(src, run_id, request.args.get("best")), src)


@get("/runs/<path:run_id>/rounds/<path:round_id>")
def round_detail(run_id, round_id):
    src = _source()
    return _ok(data.round_detail(src, run_id, round_id), src)


@get("/runs/<path:run_id>/rounds")
def rounds(run_id):
    src = _source()
    return _ok(data.rounds(src, run_id), src)


@get("/runs/<path:run_id>/attempts")
def attempts(run_id):
    src = _source()
    return _ok(data.attempts(src, run_id, _filters()), src)


@get("/runs/<path:run_id>/executions")
def executions(run_id):
    src = _source()
    return _ok(data.executions(src, run_id), src)


@get("/runs/<path:run_id>/comparisons")
def comparisons(run_id):
    src = _source()
    return _ok(data.comparisons(src, run_id), src)


@get("/runs/<path:run_id>/reviewer-reports")
def reviewer_reports(run_id):
    src = _source()
    return _ok(data.reviewer_reports(src, run_id), src)


@get("/runs/<path:run_id>/guidance")
def guidance(run_id):
    src = _source()
    return _ok(data.guidance(src, run_id), src)


@get("/runs/<path:run_id>/submissions")
def submissions(run_id):
    src = _source()
    return _ok(data.submissions(src, run_id), src)


@get("/runs/<path:run_id>/holdout")
def holdout(run_id):
    src = _source()
    return _ok(data.holdout(src, run_id), src)


@get("/runs/<path:run_id>/final-report")
def final_report(run_id):
    src = _source()
    return _ok(data.final_report(src, run_id), src)


@get("/runs/<path:run_id>/bundles")
def bundles(run_id):
    src = _source()
    return _ok(data.bundles(src, run_id), src)


@get("/runs/<path:run_id>/llm-calls")
def llm_calls(run_id):
    src = _source()
    args = request.args
    return _ok(data.llm_calls(src, run_id, args.get("role"), args.get("outcome"), args.get("q")), src)


@get("/runs/<path:run_id>/role-traces/<int:n>")
def role_trace(run_id, n):
    src = _source()
    return _ok(data.role_trace(src, run_id, n), src)


@get("/runs/<path:run_id>/role-traces")
def role_traces(run_id):
    src = _source()
    args = request.args
    return _ok(data.role_traces(src, run_id, args.get("role"), args.get("round"),
                                args.get("candidate")), src)


@get("/runs/<path:run_id>")
def run(run_id):
    src = _source()
    return _ok(data.run_summary(src, run_id, request.args.get("best")), src)


@get("/attempts/<path:attempt_id>/executions")
def attempt_executions(attempt_id):
    src = _source()
    return _ok(data.attempt_executions(src, attempt_id), src)


@get("/attempts/<path:attempt_id>")
def attempt(attempt_id):
    src = _source()
    return _ok(data.attempt_detail(src, attempt_id), src)


@get("/suggestions")
def suggestions():
    src = _source()
    return _ok(data.suggestions(src), src)


live.register(get, _source)  # events, console, run, decisions/pending, audit, notes, environment, jobs, config
