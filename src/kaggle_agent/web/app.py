"""Flask application of the read-only cockpit (design section 15).

The app reads the registry through the ``Registry`` interface and registers GET routes only: the
HTML pages (``pages.py``) and the JSON API (``api.py``) both call ``data.py``. It has no form,
takes no input from the request body and never calls a method that changes the registry.

``create_app(source)`` takes a ``Registry`` instance (tests, spies) or a zero-argument factory.
With a factory every request opens its own registry (read-only in production), keeps it in
``flask.g`` and closes it when the request ends, so the server can handle requests in parallel
(the single-thread choice of ticket 24 is reversed, see ``__main__``).
"""

import json
import sqlite3
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import NotFoundError, Registry, RegistryError

from . import api, data, pages, security

API_VERSION = 1


def _pretty(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)


def _num(value, digits: int = 6) -> str:
    """A number for display with ``digits`` significant digits; ``-`` for a missing value."""
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.{digits}g}"


def create_app(source, project_dir=None, journal_mode=None) -> Flask:
    """Build the app over a registry or a registry factory.

    ``project_dir`` is the project folder; the only file read from it is ``project.json``.
    ``journal_mode`` is an optional zero-argument callable that tells the journal mode of the
    registry file (shown by ``/api/meta``).
    """
    app = Flask(__name__)
    app.jinja_env.filters["pretty"] = _pretty
    app.jinja_env.filters["num"] = _num
    app.jinja_env.trim_blocks = True
    app.jinja_env.lstrip_blocks = True
    app.config["PROJECT_DIR"] = None if project_dir is None else Path(project_dir)
    app.config["JOURNAL_MODE"] = journal_mode
    app.config["REGISTRY_IS_FACTORY"] = not isinstance(source, Registry)
    security.install(app)

    def open_registry():
        if "registry" not in g:
            g.registry = source() if app.config["REGISTRY_IS_FACTORY"] else source
        return g.registry

    @app.teardown_appcontext
    def _close_registry(exc):
        registry = g.pop("registry", None)
        if registry is not None and app.config["REGISTRY_IS_FACTORY"]:
            registry.close()

    def current_source() -> data.Source:
        if "source" not in g:
            g.source = data.Source(open_registry(), app.config["PROJECT_DIR"])
        return g.source

    app.config["CURRENT_SOURCE"] = current_source

    @app.context_processor
    def _shell():
        folder = app.config["PROJECT_DIR"]
        return {"project_name": folder.name if folder else None}

    app.register_blueprint(api.blueprint)
    pages.register(app)

    def _error(status: int, title: str, message: str):
        if request.path.startswith("/api/"):
            body = {"api_version": API_VERSION,
                    "error": {"status": status, "title": title, "message": message}}
            return jsonify(body), status
        html = render_template("error.html", status=status, title=title, message=message)
        return html, status

    @app.errorhandler(NotFoundError)
    def _not_found(exc):
        return _error(404, "Not found", str(exc))

    @app.errorhandler(ValidationError)
    def _invalid(exc):
        return _error(400, "Invalid request", str(exc))

    @app.errorhandler(RegistryError)
    @app.errorhandler(sqlite3.Error)
    def _unavailable(exc):
        return _error(
            503, "Registry not available",
            f"The registry file could not be read: {exc}. A file written by an older version "
            "may need to be opened once by the orchestrator to be migrated.")

    @app.errorhandler(HTTPException)
    def _http(exc):
        messages = {
            400: ("Invalid request", "The request is not valid."),
            403: ("Forbidden", exc.description or "This request is not allowed."),
            404: ("Not found", "There is no such page."),
            405: ("Method not allowed", "This interface is read-only: GET only."),
        }
        title, message = messages.get(exc.code, (exc.name, exc.description or exc.name))
        return _error(exc.code, title, message)

    return app
