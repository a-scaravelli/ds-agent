"""Local-only protections of the cockpit (design section 15).

- ``install`` registers the Host check (DNS rebinding) and the security headers on every
  response, static files included.
- ``check_write_request`` and ``write_guard`` are the CSRF and Origin guard that the write
  endpoints of ticket 2 will use. They are NOT applied to any route here: the app has GET
  routes only.
"""

import functools
import hmac
import re
import secrets

from flask import Flask, abort, current_app, request

ALLOWED_HOST = re.compile(r"(localhost|127\.0\.0\.1)(:[0-9]{1,5})?", re.IGNORECASE)
CSRF_HEADER = "X-CSRF-Token"
CSRF_FIELD = "csrf_token"

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def host_allowed(host: str | None) -> bool:
    """True for ``localhost`` and ``127.0.0.1``, with or without a port."""
    return bool(host) and ALLOWED_HOST.fullmatch(host) is not None


def new_token() -> str:
    return secrets.token_urlsafe(32)


def install(app: Flask) -> None:
    """Host check before every request, security headers after; creates the CSRF token."""
    app.config.setdefault("CSRF_TOKEN", new_token())

    @app.before_request
    def _check_host():
        if not host_allowed(request.headers.get("Host")):
            abort(403, description="Only localhost and 127.0.0.1 are served.")

    @app.after_request
    def _headers(response):
        for name, value in HEADERS.items():
            response.headers[name] = value
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.context_processor
    def _token():
        return {"csrf_token": app.config["CSRF_TOKEN"]}


def check_write_request(req=None) -> None:
    """Abort with 403 unless Origin matches Host and the CSRF token is right.

    The token comes from the ``X-CSRF-Token`` header or the ``csrf_token`` form field. A missing
    Origin is refused: a browser always sends it on a write request.
    """
    req = req or request
    expected = current_app.config["CSRF_TOKEN"]
    given = req.headers.get(CSRF_HEADER) or req.form.get(CSRF_FIELD) or ""
    if not hmac.compare_digest(given.encode(), expected.encode()):
        abort(403, description="Missing or wrong CSRF token.")
    origin = req.headers.get("Origin")
    host = req.headers.get("Host")
    if not origin or not host_allowed(host) or origin not in (f"http://{host}", f"https://{host}"):
        abort(403, description="The Origin does not match the Host.")


def write_guard(view):
    """Decorator for a write endpoint (ticket 2): runs ``check_write_request`` first."""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        check_write_request()
        return view(*args, **kwargs)

    return wrapper
