"""Start command: ``python -m kaggle_agent.web --db PATH [--project DIR] [--host HOST] [--port PORT]``.

Opens the registry read-only and serves the cockpit. This is the only module of the web package
that imports the SQLite registry.

Decision reversal of ticket 24: that ticket made the server single-threaded (``threaded=False``)
because the registry held one SQLite connection, which may only be used in the thread that created
it. The cockpit needs concurrent requests (live updates of ticket 2 use server-sent events), so the
server is threaded now and every request opens its own read-only connection and closes it at the
end (``create_app`` takes a factory). The orchestrator opens the file for writing and turns on WAL
(see ``SqliteRegistry``), so a reader never blocks the writer. A read-only reader of a WAL file
needs a writable folder for the ``-shm`` file.

The page is local only: it binds to 127.0.0.1 by default and rejects requests whose Host header is
not localhost or 127.0.0.1. ``--project`` names the project folder, only to read ``project.json``
(metric direction and protocol); the web never reads a bundle folder, so there is no
``--workspace`` option.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from kaggle_agent.registry import RegistryError
from kaggle_agent.registry.sqlite import SqliteRegistry

from .app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOOPBACK = ("127.0.0.1", "localhost")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kaggle_agent.web",
        description="Read-only cockpit over the experiment registry (local only).",
    )
    parser.add_argument("--db", required=True, help="path of the SQLite registry file")
    parser.add_argument(
        "--project",
        help="project folder, only to read project.json (default: the folder of the registry)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port (default 8000)")
    return parser


class BindError(Exception):
    """The server could not listen on the requested address (werkzeug exits instead of raising)."""


def make_web_server(app, host: str, port: int, threaded: bool = True):
    """A werkzeug server for ``app``; port 0 picks a free port (read ``server_port``).

    Threaded by default: every request runs in its own thread with its own registry connection.
    """
    from werkzeug.serving import make_server

    try:
        return make_server(host, port, app, threaded=threaded)
    except SystemExit:  # werkzeug prints the OS error and exits when the bind fails
        raise BindError(f"cannot listen on {host}:{port}") from None


def journal_mode_of(db_path) -> str | None:
    """The journal mode stored in the registry file (``wal`` once the orchestrator opened it)."""
    try:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        db = sqlite3.connect(uri, uri=True)
        try:
            return str(db.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        finally:
            db.close()
    except sqlite3.Error:
        return None


def main(argv: list[str] | None = None, run=None) -> int:
    """Parse ``argv``, check the registry opens read-only and start the app.

    ``run(app, host, port, debug)`` replaces the real server in tests.
    Returns 2 with a message on stderr when the registry cannot be opened.
    """
    args = _parser().parse_args(argv)
    db_path = Path(args.db)
    try:
        SqliteRegistry(db_path, read_only=True).close()  # the same check a request makes
    except RegistryError as exc:
        print(
            f"error: {exc}\nThe registry file must exist and have been created by this version "
            "(open it once in write mode to migrate an older file).",
            file=sys.stderr,
        )
        return 2
    if args.host not in LOOPBACK:
        print(
            f"warning: {args.host} is not a loopback address; the page has no login. "
            "Only localhost and 127.0.0.1 Host headers are accepted anyway.",
            file=sys.stderr,
        )
    project = Path(args.project) if args.project else db_path.resolve().parent
    app = create_app(
        lambda: SqliteRegistry(db_path, read_only=True),
        project_dir=project,
        journal_mode=lambda: journal_mode_of(db_path),
    )
    if run is None:
        def run(app, host, port, debug):
            server = make_web_server(app, host, port, threaded=True)
            print(f"Serving on http://{host}:{server.server_port}/ (Ctrl-C to stop)")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    try:
        run(app, args.host, args.port, False)
    except BindError:
        print(
            f"error: port {args.port} on {args.host} is already in use or not available; "
            "choose another with --port (for example --port 8001)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
