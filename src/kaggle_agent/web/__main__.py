"""Start command: ``python -m kaggle_agent.web --db PATH [--host HOST] [--port PORT]``.

Opens the registry read-only and serves the web interface. This is the only
module of the web package that imports the SQLite registry.
"""

import argparse
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
            app.run(host=host, port=port, debug=debug)
    try:
        run(app, args.host, args.port, False)
    finally:
        registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
