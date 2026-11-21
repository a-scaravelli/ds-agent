"""HTML pages of the cockpit, GET only; they call the same ``data.py`` functions as the API."""

from flask import current_app, redirect, render_template, request, url_for

from . import analytics_pages, data
from . import audit, decisions, live, store
from kaggle_agent.proposals import ALLOWED_LIBRARIES


def _source() -> data.Source:
    return current_app.config["CURRENT_SOURCE"]()


def _page(template: str, src: data.Source, run: dict | None = None, **context):
    return render_template(template, run=run, warnings=src.warnings, **context)


def register(app) -> None:
    def get(rule: str):
        return app.route(rule, methods=["GET"], provide_automatic_options=False)

    @get("/")
    def cycles():
        src = _source()
        q = request.args.get("q") or None
        return _page("cycles.html", src, runs=data.runs(src, q), q=q or "")

    @get("/cycles/<path:cycle_id>/rounds/<path:round_id>")
    def round_view(cycle_id, round_id):
        src = _source()
        run = data.run_row(src, cycle_id)
        return _page("round.html", src, run, round=data.round_detail(src, cycle_id, round_id))

    @get("/cycles/<path:cycle_id>/rounds")
    def rounds(cycle_id):
        src = _source()
        run = data.run_row(src, cycle_id)
        return _page("rounds.html", src, run, rounds=data.rounds(src, cycle_id))

    @get("/cycles/<path:cycle_id>/attempts")
    def attempts(cycle_id):
        src = _source()
        run = data.run_row(src, cycle_id)
        filters = {k: request.args.get(k)
                   for k in ("outcome", "round", "min_score", "max_score", "q")}
        rows = data.attempts(src, cycle_id, filters)
        round_ids = [r["id"] for r in data.rounds(src, cycle_id)]
        return _page("attempts.html", src, run, attempts=rows, filters=filters,
                     round_ids=round_ids, outcomes=("succeeded", "failed", "inconclusive", "open"))

    @get("/cycles/<path:cycle_id>/metrics")
    def metrics(cycle_id):
        return redirect(url_for("cycle", cycle_id=cycle_id), code=301)

    @get("/cycles/<path:cycle_id>/reports")
    def reports(cycle_id):
        return redirect(url_for("report", cycle_id=cycle_id), code=301)

    @get("/cycles/<path:cycle_id>/traces/<int:n>")
    def trace_view(cycle_id, n):
        src = _source()
        trace = data.role_trace(src, cycle_id, n)
        if request.headers.get("HX-Request"):
            return render_template("trace_fragment.html", trace=trace, cycle_id=cycle_id)
        run = data.run_row(src, cycle_id)
        return _page("trace_detail.html", src, run, trace=trace, cycle_id=cycle_id)

    @get("/cycles/<path:cycle_id>/traces")
    def traces(cycle_id):
        src = _source()
        run = data.run_row(src, cycle_id)
        role, round_id = request.args.get("role") or None, request.args.get("round") or None
        outcome, q = request.args.get("outcome") or None, request.args.get("q") or None
        calls = data.llm_calls(src, cycle_id, role, outcome, q)
        calls.sort(key=lambda c: (c["role"] or "", c["index"]))  # grouped per role
        by_role: dict = {}
        for c in calls:
            g = by_role.setdefault(c["role"], {"role": c["role"], "calls": 0, "not_ok": 0,
                                               "input_tokens": 0, "output_tokens": 0, "seconds": 0.0})
            g["calls"] += 1
            g["not_ok"] += c["outcome"] != "ok"
            g["input_tokens"] += c["input_tokens"] or 0
            g["output_tokens"] += c["output_tokens"] or 0
            g["seconds"] += c["duration_seconds"]
        traces_ = data.role_traces(src, cycle_id, role, round_id)
        round_ids = [r["id"] for r in data.rounds(src, cycle_id)]
        return _page("traces.html", src, run, calls=calls, by_role=list(by_role.values()), traces=traces_, role=role,
                     round_id=round_id, outcome=outcome, q=q or "", round_ids=round_ids,
                     roles=data.ROLE_NAMES)

    @get("/cycles/<path:cycle_id>/report")
    def report(cycle_id):
        src = _source()
        summary = data.run_summary(src, cycle_id)
        return _page(
            "report.html", src, summary, report=data.final_report(src, cycle_id),
            holdout=data.holdout(src, cycle_id), submissions=data.submissions(src, cycle_id),
            suggestions=[s for s in data.suggestions(src) if s["origin_cycle"] == cycle_id],
        )

    @get("/cycles/<path:cycle_id>/guidance")
    def guidance(cycle_id):
        src = _source()
        run = data.run_row(src, cycle_id)
        return _page("guidance.html", src, run, items=data.guidance(src, cycle_id))

    @get("/cycles/<path:cycle_id>")
    def cycle(cycle_id):
        src = _source()
        summary = data.run_summary(src, cycle_id, request.args.get("best"))
        series = data.chart(src, cycle_id, request.args.get("best"))
        return _page("cycle.html", src, summary, summary=summary, chart=series,
                     bundles=data.bundles(src, cycle_id), submissions=data.submissions(src, cycle_id))

    @get("/attempts/<path:attempt_id>")
    def attempt(attempt_id):
        src = _source()
        detail = data.attempt_detail(src, attempt_id)
        run = data.run_row(src, detail["run_id"])
        return _page("attempt.html", src, run, a=detail)

    @get("/suggestions")
    def suggestions():
        src = _source()
        return _page("suggestions.html", src, items=data.suggestions(src))

    # -- cockpit pages: live progress, run control, decisions, setup, environment, audit ----------

    def project():
        return current_app.config["PROJECT_DIR"]

    def controller():
        return current_app.config.get("CONTROLLER")

    @get("/live")
    def live_page():
        src = _source()
        recent = live.events_after(project(), 0, 10 ** 9)[-60:] if project() else []
        return _page("live.html", src, recent=recent)

    @get("/run")
    def run_control():
        src = _source()
        ctl = controller()
        state = ctl.state() if ctl else {"state": "idle", "label": "idle", "text": "Nothing is running."}
        settings = (store.read_json(store.cockpit_path(project(), "settings.json"), {}) or {}) if project() else {}
        started = bool(project()) and store.exists(store.project_path(project(), "project.json"))
        history = [r for r in (audit.timeline(project()) if project() else [])
                   if str(r["what"]).startswith("run.") or r["what"] in (
                       "finished", "paused", "stopped", "suspended", "error")][-20:]
        return _page("run_control.html", src, state=state, started=started, history=history,
                     auto_resume=bool(settings.get("auto_resume")))

    @get("/decisions")
    def decisions_page():
        src = _source()
        question = decisions.current_question(src) if project() else None
        cycle_id = decisions.cycle_id_of(src) if project() else None
        limits = None
        report = None
        if cycle_id:
            try:
                limits = data.limits_view(src, src.registry.get_cycle(cycle_id))
            except Exception:
                limits = None
            if question and question["kind"] == "review":
                report = data.final_report(src, cycle_id)
        statuses = decisions.decision_statuses(project()) if project() else []
        ctl = controller()
        # a saved decision is applied when the run resumes: offer the button unless a run is going
        offer = any(s["status"] == "pending" for s in statuses) and ctl is not None and not ctl.active()
        return _page("decisions.html", src, question=question, limits=limits, cycle_id=cycle_id,
                     statuses=statuses, resume_offer=offer,
                     conflicts=decisions.conflicts(project(), question) if project() else [],
                     lead=store.lead_toml_summary(project()) if project() else None,
                     notices=store.read_notices(project()) if project() else [], report=report)

    @get("/setup")
    def setup():
        src = _source()
        ctl = controller()
        raw = store.read_toml(ctl.config_path) if ctl else None
        started = bool(project()) and store.exists(store.project_path(project(), "project.json"))
        return _page("setup.html", src, cfg=raw or {}, config_exists=raw is not None, started=started,
                     libraries=list(ALLOWED_LIBRARIES))

    @get("/environment")
    def environment_page():
        src = _source()
        return _page("environment.html", src)

    analytics_pages.register(app)  # ticket 29: attempt analytics, compare and paired pages

    @get("/audit")
    def audit_page():
        src = _source()
        return _page("audit.html", src, entries=audit.timeline(project()) if project() else [])
