"""Start command: ``python -m kaggle_agent.web --db PATH [--host HOST] [--port PORT]``.

Opens the registry read-only and serves the web interface. This is the only
module of the web package that imports the SQLite registry.

The server is single-threaded on purpose (``threaded=False``): the registry holds one SQLite
connection, created in the thread that runs ``main``, and SQLite objects may only be used in the
thread that created them. A threaded server would run every request in a new thread and fail with
HTTP 500. The interface is read-only and for one local user, so serving one request at a time is
enough and needs no change to the registry.
"""

import argparse
import errno
import sys

from kaggle_agent.registry import RegistryError
from kaggle_agent.registry.sqlite import SqliteRegistry

from .app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kaggle_agent.web",
        description="Read-only web interface over the experiment registry.",
    )
    parser.add_argument("--db", required=True, help="path of the SQLite registry file")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port (default 8000)")
    return parser


def make_web_server(app, host: str, port: int, threaded: bool = False):
    """A werkzeug server for ``app``; port 0 picks a free port (read ``server_port``).

    ``threaded`` is False in production: the registry connection belongs to a single thread.
    """
    from werkzeug.serving import make_server

    return make_server(host, port, app, threaded=threaded)


def main(argv: list[str] | None = None, run=None) -> int:
    """Parse ``argv``, open the registry read-only and start the app.

    ``run(app, host, port, debug)`` replaces the real server in tests.
    Returns 2 with a message on stderr when the registry cannot be opened.
    """
    args = _parser().parse_args(argv)
    try:
        registry = SqliteRegistry(args.db, read_only=True)
    except RegistryError as exc:
        print(
            f"error: {exc}\nThe registry file must exist and have been created by this version "
            "(open it once in write mode to migrate an older file).",
            file=sys.stderr,
        )
        return 2
    app = create_app(registry)
    if run is None:
        def run(app, host, port, debug):
            server = make_web_server(app, host, port, threaded=False)
            print(f"Serving on http://{host}:{server.server_port}/ (Ctrl-C to stop)")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    try:
        run(app, args.host, args.port, False)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        print(
            f"error: port {args.port} on {args.host} is already in use; "
            "choose another with --port (for example --port 8001)",
            file=sys.stderr,
        )
        return 2
    finally:
        registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
