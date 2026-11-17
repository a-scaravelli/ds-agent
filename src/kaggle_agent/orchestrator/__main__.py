"""Command line: ``python -m kaggle_agent.orchestrator start|resume|status|submit --config FILE``.

- ``start``: create the project and the cycle and run until the cycle needs the lead or stops
  (a second ``start`` on the same project changes nothing).
- ``resume``: continue a suspended, crashed or waiting cycle.
- ``status``: where the cycle stands and exactly which key of the lead file to fill next (no network).
- ``submit``: build the submission from the chosen persisted bundle after the review. WITHOUT
  ``--confirm-submit`` it is a dry run that sends nothing; with it the file is sent (once).

- ``validate-config``, ``preview``, ``profile`` (``--out FILE``): small jobs of the cockpit that write one JSON
  result to ``--out`` (see ``jobs``). ``start`` and ``resume`` take ``--supervised`` (heartbeat, events and
  pause for the cockpit's run controller) and always take the run lock: a second run exits 1 at once.

Exit codes: 0 finished/ok, 10 waiting for the lead, 11 suspended (quota, rate limit or expired Claude login),
12 lead file invalid, 13 crash (run resume), 14 paused by the DS (run resume), 15 stopped by SIGTERM (run
resume), 2 usage or configuration error, 1 other error, and for ``submit``
20 not accepted yet, 21 invalid submission file, 22 daily limit, 23 score still pending,
24 a previous send may have reached Kaggle (see the README).
"""

import argparse
import json
import os
import signal
import sys

from . import codes, jobs
from .cockpit import CockpitFiles, Heartbeat, Interrupted, RunLock, utc_text
from .events import Emitter, EventLog
from .build import Dependencies, real_dependencies
from .codes import CommandResult
from .config import load_config
from .errors import ConfigError, OrchestratorError
from .service import resume, start, status
from .submission import run_submit


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
        if name in ("start", "resume"):
            p.add_argument("--supervised", action="store_true",
                           help="write the heartbeat and the event log for the cockpit")
        if name == "submit":
            p.add_argument("--confirm-submit", action="store_true",
                           help="really send the file to Kaggle (without it: a dry run)")
            p.add_argument("--assume-not-sent", action="store_true",
                           help="after checking Kaggle by hand: proceed although a marker says a send "
                                "may have happened")
    for name in ("validate-config", "preview", "profile"):
        p = sub.add_parser(name, help="cockpit job: write a JSON result to --out")
        p.add_argument("--config", required=True, help="path of the TOML configuration file")
        p.add_argument("--out", required=True, help="file that receives the JSON result")
    return parser


def _run_locked(args, config, deps) -> CommandResult:
    """start / resume under the run lock; ``--supervised`` adds the heartbeat, events and exit.json."""
    files = CockpitFiles(config.paths.folder)
    lock = RunLock(files)
    if not lock.acquire():
        return CommandResult(codes.ERROR, (
            "error: another run is already active in this project (run.lock is held); "
            "no second run is started.",))
    heartbeat = emitter = None
    previous = None
    try:
        files.exit.unlink(missing_ok=True)
        if args.supervised:
            heartbeat = Heartbeat(files)
            heartbeat.start()
            emitter = Emitter(EventLog(files.events))
            previous = signal.signal(signal.SIGTERM, _raise_interrupted)
        command = start if args.command == "start" else resume
        try:
            result = (command(config, deps, files, emitter) if emitter is not None
                      else command(config, deps))
        except Interrupted:
            if emitter is not None:
                emitter.emit("stopped")
            result = CommandResult(codes.INTERRUPTED, (
                "The run was STOPPED (SIGTERM). The registry is intact; run resume to continue.",))
        if args.supervised:
            text = result.lines[0] if result.lines else ""
            _write_exit(files, result.code, text)
        return result
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        if heartbeat is not None:
            heartbeat.stop()
        lock.release()


def _raise_interrupted(signum, frame):
    raise Interrupted()


def _write_exit(files: CockpitFiles, code: int, meaning: str) -> None:
    from .project import write_atomic
    write_atomic(files.exit, json.dumps({"code": code, "meaning": meaning[:300], "at": utc_text(),
                                         "pid": os.getpid()}))


def run(args, deps: Dependencies | None) -> CommandResult:
    if args.command == "validate-config":  # reports an invalid file as a result, not as a usage error
        return CommandResult(jobs.run_job(args.command, args.config, args.out, None), ())
    try:
        config = load_config(args.config)
    except ConfigError as error:
        return CommandResult(codes.USAGE, (f"error: {error}",))
    if args.command == "status":
        return status(config)
    if args.command in ("preview", "profile"):  # the real clients are built only if data is missing
        return CommandResult(jobs.run_job(args.command, args.config, args.out, deps), ())
    try:
        deps = deps or real_dependencies()
    except OrchestratorError as error:
        return CommandResult(codes.ERROR, (f"error: {error}",))
    except Exception as error:  # no credentials, no kaggle library, ...
        return CommandResult(codes.ERROR, (f"error: cannot start the real clients: {type(error).__name__}: {error}",))
    if args.command in ("start", "resume"):
        return _run_locked(args, config, deps)
    try:
        return run_submit(config, deps, args.confirm_submit, args.assume_not_sent)
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
