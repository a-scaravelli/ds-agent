"""Command line: ``python -m kaggle_agent.orchestrator start|resume|status|submit --config FILE``.

- ``start``: create the project and the cycle and run until the cycle needs the lead or stops
  (a second ``start`` on the same project changes nothing).
- ``resume``: continue a suspended, crashed or waiting cycle.
- ``status``: where the cycle stands and exactly which key of the lead file to fill next (no network).
- ``submit``: build the submission from the chosen persisted bundle after the review. WITHOUT
  ``--confirm-submit`` it is a dry run that sends nothing; with it the file is sent (once).

Exit codes: 0 finished/ok, 10 waiting for the lead, 11 suspended (quota/rate limit), 12 lead file
invalid, 13 crash (run resume), 2 usage or configuration error, 1 other error, and for ``submit``
20 not accepted yet, 21 invalid submission file, 22 daily limit, 23 score still pending,
24 a previous send may have reached Kaggle (see the README).
"""

import argparse
import sys

from . import codes
from .build import Dependencies, real_dependencies
from .codes import CommandResult
from .config import load_config
from .errors import ConfigError, OrchestratorError
from .service import resume, start, status
from .submission import submit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kaggle_agent.orchestrator",
        description="Run a tabular Kaggle competition end to end; the lead decides through a file.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("start", "create the project and a new cycle and run it"),
                       ("resume", "continue a suspended, crashed or waiting cycle"),
                       ("status", "show where the cycle stands and what the lead has to fill"),
                       ("submit", "send the submission after the review (dry run without the flag)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--config", required=True, help="path of the TOML configuration file")
        if name == "submit":
            p.add_argument("--confirm-submit", action="store_true",
                           help="really send the file to Kaggle (without it: a dry run)")
            p.add_argument("--assume-not-sent", action="store_true",
                           help="after checking Kaggle by hand: proceed although a marker says a send "
                                "may have happened")
    return parser


def run(args, deps: Dependencies | None) -> CommandResult:
    try:
        config = load_config(args.config)
    except ConfigError as error:
        return CommandResult(codes.USAGE, (f"error: {error}",))
    if args.command == "status":
        return status(config)
    try:
        deps = deps or real_dependencies()
    except OrchestratorError as error:
        return CommandResult(codes.ERROR, (f"error: {error}",))
    except Exception as error:  # no credentials, no kaggle library, ...
        return CommandResult(codes.ERROR, (f"error: cannot start the real clients: {type(error).__name__}: {error}",))
    if args.command == "start":
        return start(config, deps)
    if args.command == "resume":
        return resume(config, deps)
    try:
        return submit(config, deps, args.confirm_submit, args.assume_not_sent)
    except OrchestratorError as error:
        return CommandResult(codes.USAGE, (f"error: {error}",))
    except Exception as error:  # a failed registry write after a send leaves the marker in place
        return CommandResult(codes.ERROR, (
            f"error: {type(error).__name__}: {error}",
            "If the file may have been sent, check the Kaggle submissions page and see the README "
            "(pending marker) before running submit --confirm-submit again."))


def main(argv: list[str] | None = None, deps: Dependencies | None = None) -> int:
    """Run a command and return its exit code; ``deps`` replaces the real clients in tests."""
    args = _parser().parse_args(argv)
    result = run(args, deps)
    for line in result.lines:
        print(line, file=sys.stderr if result.code in (codes.ERROR, codes.USAGE) else sys.stdout)
    return result.code


if __name__ == "__main__":
    sys.exit(main())
