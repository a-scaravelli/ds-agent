"""Flask application of the read-only web interface.

The app reads the registry through the ``Registry`` interface and registers
GET routes only. It has no form, takes no input from the request body and
never calls a method that changes the registry. Guidance and approvals stay
outside the web at level 1.
"""

import json

from flask import Flask, render_template, request

from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import NotFoundError, Registry

from . import views
from .chart import render_chart


def _pretty(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)


def create_app(registry: Registry) -> Flask:
    """Build the app over ``registry``; only read methods of it are ever called.

    The registry is passed as an instance (not a path) so a test can pass a
    spy and the command line can pass one opened read-only.
    """
    app = Flask(__name__)
    app.jinja_env.filters["pretty"] = _pretty

    def get(rule: str):
        return app.route(rule, methods=["GET"], provide_automatic_options=False)

    @get("/")
    def cycles():
        return render_template("cycles.html", cycles=views.list_cycles(registry))

    @get("/cycles/<path:cycle_id>/rounds/<path:round_id>")
    def round_view(cycle_id, round_id):
        return render_template("round.html", **views.round_page(registry, cycle_id, round_id))

    @get("/cycles/<path:cycle_id>/metrics")
    def metrics(cycle_id):
        data = views.metrics_page(
            registry, cycle_id, request.args.get("partition"), request.args.get("best")
        )
        svg = render_chart(data["by_round"], data["best"])
        return render_template("metrics.html", svg=svg, **data)

    @get("/cycles/<path:cycle_id>/reports")
    def reports(cycle_id):
        return render_template("reports.html", **views.reports_page(registry, cycle_id))

    @get("/cycles/<path:cycle_id>/traces")
    def traces(cycle_id):
        data = views.traces_page(
            registry, cycle_id, request.args.get("role"), request.args.get("round")
        )
        return render_template("traces.html", **data)

    @get("/cycles/<path:cycle_id>/guidance")
    def guidance(cycle_id):
        return render_template("guidance.html", **views.guidance_page(registry, cycle_id))

    @get("/cycles/<path:cycle_id>")
    def cycle(cycle_id):
        return render_template("cycle.html", **views.cycle_page(registry, cycle_id))

    @get("/attempts/<path:attempt_id>")
    def attempt(attempt_id):
        return render_template("attempt.html", **views.attempt_page(registry, attempt_id))

    @get("/suggestions")
    def suggestions():
        return render_template("suggestions.html", items=views.suggestions_page(registry))

    def _error(status: int, title: str, message: str):
        return render_template("error.html", status=status, title=title, message=message), status

    @app.errorhandler(NotFoundError)
    def _not_found(exc):
        return _error(404, "Not found", str(exc))

    @app.errorhandler(404)
    def _no_page(exc):
        return _error(404, "Not found", "There is no such page.")

    @app.errorhandler(ValidationError)
    def _invalid(exc):
        return _error(400, "Invalid request", str(exc))

    @app.errorhandler(405)
    def _not_allowed(exc):
        return _error(405, "Method not allowed", "This interface is read-only: GET only.")

    return app
