"""Flask application of the cockpit (design section 15).

The app reads the registry through the ``Registry`` interface and never calls a method that changes it
(single writer: the orchestrator). Everything is GET (HTML pages in ``pages.py``, JSON API in ``api.py`` and
``live.py``, all calling ``data.py``) EXCEPT the write endpoints of ``writes.WRITE_ENDPOINTS``: each is a
POST wrapped in ``security.write_guard`` (CSRF token and Origin check) and writes only files in
``<project>/cockpit/`` (see ``store.py``).

``create_app(source)`` takes a ``Registry`` instance (tests, spies) or a zero-argument factory.
With a factory every request opens its own registry (read-only in production), keeps it in
``flask.g`` and closes it when the request ends, so the server can handle requests in parallel
(the single-thread choice of ticket 24 is reversed, see ``__main__``).
"""

import json
import sqlite3
import threading
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import NotFoundError, Registry, RegistryError

from . import api, data, pages, security, writes
from .audit import AuditError
from .controller import Conflict, RunController
from .environment import Environment

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


def create_app(source, project_dir=None, journal_mode=None, controller=None, sse=None,
               environment=None) -> Flask:
    """Build the app over a registry or a registry factory.

    ``project_dir`` is the project folder (``project.json`` and the ``cockpit/`` files).
    ``journal_mode`` is an optional zero-argument callable that tells the journal mode of the
    registry file (shown by ``/api/meta``). ``controller`` is the ``RunController`` (default: one built
    for ``project_dir`` with the default commands); ``sse`` overrides the stream intervals (tests);
    ``environment`` replaces the environment-card builder (tests).
    """
    app = Flask(__name__)
    app.jinja_env.filters["pretty"] = _pretty
    app.jinja_env.filters["num"] = _num
    app.jinja_env.trim_blocks = True
    app.jinja_env.lstrip_blocks = True
    app.config["PROJECT_DIR"] = None if project_dir is None else Path(project_dir)
    app.config["JOURNAL_MODE"] = journal_mode
    if controller is None and project_dir is not None:
        controller = RunController(project_dir)
    app.config["CONTROLLER"] = controller
    app.config["ENVIRONMENT"] = environment or (
        Environment(controller, project_dir) if controller is not None else None)
    app.config["SHUTDOWN"] = threading.Event()
    app.config.update({"SSE_HEARTBEAT": 15.0, "SSE_MAX_SECONDS": 60.0, "SSE_POLL": 0.25, "SSE_MAX_STREAMS": 8})
    app.config.update(sse or {})
    app.config["REGISTRY_IS_FACTORY"] = callable(source) and not isinstance(source, Registry)
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
    app.register_blueprint(writes.blueprint)
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

    @app.errorhandler(Conflict)
    def _conflict(exc):
        return _error(409, "Cannot do that now", str(exc))

    @app.errorhandler(AuditError)
    def _audit_failed(exc):
        return _error(503, "Audit log not available", str(exc))

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
            405: ("Method not allowed", "This address does not accept that method."),
        }
        title, message = messages.get(exc.code, (exc.name, exc.description or exc.name))
        return _error(exc.code, title, message)

    return app
