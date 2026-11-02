"""Commands: ``python -m kaggle_agent.pipeline reproduce|predict ...``.

- ``reproduce --persisted DIR --db PATH --data CSV``: re-run the frozen-protocol validation from the
  persisted bundle code and compare it with the metric in the report (exit 0 if equal within 1e-9).
- ``predict --persisted DIR --input CSV --output JSON``: predict with the stored model in a separate
  process and write JSON.

The holdout is never read by either command.
"""

import argparse
import sys
from pathlib import Path

from kaggle_agent.sandbox import SubprocessSandbox

from . import command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kaggle_agent.pipeline",
        description="Reproduce the reported metric from a persisted bundle, or predict with its stored model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("reproduce", "re-run the frozen-protocol validation and compare it with the report"),
                       ("predict", "predict with the stored model and write JSON")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--persisted", required=True, help="folder written by the pipeline (persisted.json inside)")
        p.add_argument("--timeout", type=float, default=command.DEFAULT_TIMEOUT,
                       help="seconds allowed to each child process")
        if name == "reproduce":
            p.add_argument("--db", required=True, help="path of the SQLite registry file")
            p.add_argument("--data", required=True, help="CSV of the competition data (target included)")
        else:
            p.add_argument("--input", required=True, help="CSV of the rows to predict")
            p.add_argument("--output", required=True, help="JSON file to write")
    return parser


def main(argv: list[str] | None = None, sandbox=None) -> int:
    """Run a command and return its exit code; ``sandbox`` replaces the real one in tests."""
    args = _parser().parse_args(argv)
    sandbox = sandbox or SubprocessSandbox()
    persisted = Path(args.persisted).resolve()
    if args.command == "reproduce":
        result = command.reproduce(persisted, Path(args.db).resolve(), Path(args.data).resolve(),
                                   sandbox, args.timeout)
    else:
        result = command.predict(persisted, Path(args.input).resolve(), Path(args.output).resolve(),
                                 sandbox, args.timeout)
    for line in result.lines:
        print(line, file=sys.stderr if result.code else sys.stdout)
    return result.code


if __name__ == "__main__":
    sys.exit(main())
