"""The commands ``start``, ``resume`` and ``status`` (``submit`` lives in ``submission``).

Each returns ``CommandResult(code, lines)``. Exit codes (see ``codes``): 0 the cycle finished, 10 it
waits for the lead, 11 it was suspended (quota or rate limit: run ``resume`` later), 12 the lead file
is invalid, 13 an unexpected crash with the registry intact (run ``resume``), 2 usage or
configuration error, 1 any other error. ``status`` needs no network and no Claude: it only reads the
project folder and the registry (opened read-only).
"""

import json
from datetime import timezone
from pathlib import Path

from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.loop import CycleView
from kaggle_agent.loop.machine import STATUS_STOPPED, STATUS_SUSPENDED, STATUS_WAITING
from kaggle_agent.loop.states import AWAITING_PLAN_APPROVAL, STOPPED
from kaggle_agent.pipeline import report_path, run_pipeline
from kaggle_agent.registry import Registry, RegistryError, SqliteRegistry

from . import codes
from .build import Built, Dependencies, build_context
from .codes import CommandResult
from .config import Config
from .errors import ConfigError, LeadFileError, OrchestratorError, ProjectError
from .lead_file import Ledger, describe_pending
from .project import EXAMPLE_FILE, LEDGER_FILE, NOTICES_FILE, read_project, write_atomic
from .submission import marker_path, read_marker

LAST_RUN = "last-run.json"
EXAMPLE_TEXT = """\
# Copy the table you need into lead.toml (the program never edits lead.toml).
# `status` tells you which one is waiting. Each decision is read when the cycle asks for it.

# [plan]                    # when the cycle waits for the plan approval
# decision = "approve"      # or "revise", with: plan = 1 and text = "what to change"

# [guidance]                # after a round
# round = 1                 # the round the answer is for
# action = "guidance"       # or "continue"
# text = "what to try next"

# [review]                  # after you have read the final report
# decision = "accept"
"""


def _cmd(config: Config, name: str) -> str:
    return f"python -m kaggle_agent.orchestrator {name} --config {config.source or '<config>'}"


def _now_text(deps: Dependencies) -> str:
    return deps.clock().astimezone(timezone.utc).isoformat()


def _write_last_run(config: Config, deps: Dependencies, outcome: str, state: str, reason: str) -> None:
    data = {"outcome": outcome, "state": state, "reason": reason, "at": _now_text(deps)}
    write_atomic(config.paths.folder / LAST_RUN, json.dumps(data, indent=2))


def _plan_lines(registry: Registry, cycle_id: str) -> list[str]:
    """The plan the coordinator wrote last, as the lead has to read it before approving."""
    traces = [t for t in registry.list_role_traces(cycle_id, role="coordinator") if t.output]
    if not traces:
        return []
    output = traces[-1].output
    plan = output.get("plan", {})
    lines = ["Plan proposed by the coordinator:",
             f"  metric: {plan.get('metric')}", f"  validation: {plan.get('validation')}",
             f"  feature engineering: {'; '.join(plan.get('feature_engineering', []))}",
             f"  models: {'; '.join(plan.get('models', []))}", f"  baseline: {plan.get('baseline')}",
             f"  tasks of the round: {'; '.join(output.get('tasks', []))}"]
    return lines


def _waiting_lines(built: Built, state: str) -> list[str]:
    ctx, config = built.ctx, built.config
    lines = []
    if state == AWAITING_PLAN_APPROVAL:
        lines += _plan_lines(ctx.registry, ctx.cycle_id) + [""]
    elif state == "AWAITING_REVIEW":
        lines += [f"Final report: {report_path(config.paths.out, ctx.cycle_id)}", ""]
    return lines + describe_pending(state, ctx.registry, ctx.cycle_id, config.lead_path,
                                    Ledger(config.paths.folder / LEDGER_FILE))


def run_cycle(config: Config, deps: Dependencies) -> CommandResult:
    """Build the context and run the cycle until it stops, waits for the lead or is suspended."""
    built: Built | None = None
    try:
        built = build_context(config, deps)
        result = run_pipeline(built.ctx, built.pctx)
        outcome = result.outcome
        notices = [f"notice: {text}" for text in built.lead.emitted]
        cycle_id = built.ctx.cycle_id
        if outcome.status == STATUS_STOPPED:
            _write_last_run(config, deps, "finished", outcome.state, outcome.reason or "")
            return CommandResult(codes.FINISHED, (
                f"The cycle {cycle_id} is FINISHED (stopped; {outcome.reason}).",
                f"report: {report_path(config.paths.out, cycle_id)}",
                f"To send the submission: {_cmd(config, 'submit')} (a dry run), then add --confirm-submit.",
                *notices))
        if outcome.status == STATUS_WAITING:
            _write_last_run(config, deps, "waiting", outcome.state, outcome.reason or "")
            return CommandResult(codes.WAITING, (
                f"The cycle {cycle_id} is WAITING FOR THE LEAD ({outcome.state}).",
                *_waiting_lines(built, outcome.state), *notices,
                f"When the file is filled, run: {_cmd(config, 'resume')}"))
        assert outcome.status == STATUS_SUSPENDED
        _write_last_run(config, deps, "suspended", outcome.state, outcome.reason or "")
        return CommandResult(codes.SUSPENDED, (
            f"The cycle {cycle_id} is SUSPENDED in {outcome.state}: {outcome.reason}",
            f"Nothing was lost. When the quota or rate limit allows it, run: {_cmd(config, 'resume')}",
            *notices))
    except LeadFileError as error:
        return CommandResult(codes.LEAD_INVALID, (
            f"The lead file is not valid: {error}",
            "The state of the cycle did not change. Fix the file, then run resume."))
    except (ConfigError, ProjectError) as error:
        return CommandResult(codes.USAGE, (f"error: {error}",))
    except OrchestratorError as error:
        return CommandResult(codes.ERROR, (f"error: {error}",))
    except KeyboardInterrupt:
        raise
    except Exception as error:  # the registry is written after every step, so nothing is lost
        return CommandResult(codes.CRASHED, (
            f"The run CRASHED: {type(error).__name__}: {error}",
            "The registry is intact. Run resume to continue from the saved state."))
    finally:
        if built is not None:
            built.close()


def _existing_cycle(config: Config, cycle_id: str) -> bool:
    if not config.paths.registry.is_file():
        return False
    try:
        registry = SqliteRegistry(str(config.paths.registry), read_only=True)
    except RegistryError:
        return False
    try:
        return cycle_id in {c.id for c in registry.list_cycles()}
    finally:
        registry.close()


def start(config: Config, deps: Dependencies) -> CommandResult:
    """Create the project and the cycle and run; a second start changes nothing and calls nothing."""
    state = read_project(config.paths) if config.paths.folder.exists() else None
    if state is not None:
        shown = status(config)
        return CommandResult(codes.FINISHED, (
            f"The project is already started (cycle {state.cycle_id}): no second cycle is created.",
            "Use resume to continue it, status to see where it stands.", "", *shown.lines))
    config.paths.folder.mkdir(parents=True, exist_ok=True)
    example = config.paths.folder / EXAMPLE_FILE
    if not example.exists():
        example.write_text(EXAMPLE_TEXT, encoding="utf-8")
    result = run_cycle(config, deps)
    new = read_project(config.paths)
    if new is None:
        return result
    head = (f"competition {new.competition}: target {new.target}, metric {new.metric} ({new.direction}), "
            f"cycle {new.cycle_id}",)
    return CommandResult(result.code, head + result.lines)


def resume(config: Config, deps: Dependencies) -> CommandResult:
    """Continue a suspended, crashed or waiting cycle from the state saved in the registry."""
    if read_project(config.paths) is None:
        return CommandResult(codes.USAGE, (f"error: no project in {config.paths.folder}: run start first",))
    return run_cycle(config, deps)


def status(config: Config) -> CommandResult:
    """State, round, best score, what the lead has to fill next and where the files are."""
    paths = config.paths
    try:
        state = read_project(paths)
    except ProjectError as error:
        return CommandResult(codes.USAGE, (f"error: {error}",))
    if state is None:
        return CommandResult(codes.USAGE, (f"error: no project in {paths.folder}: run start first",))
    lines = [f"competition: {state.competition}; target {state.target}; metric {state.metric} "
             f"({state.direction})", f"cycle: {state.cycle_id}"]
    if not paths.registry.is_file():
        return CommandResult(codes.FINISHED, (*lines, "state: no cycle yet (the first run did not reach "
                                                       "the registry); run resume"))
    try:
        registry = SqliteRegistry(str(paths.registry), read_only=True)
    except RegistryError as error:
        return CommandResult(codes.ERROR, (f"error: {error}",))
    try:
        if state.cycle_id not in {c.id for c in registry.list_cycles()}:
            return CommandResult(codes.FINISHED, (*lines, "state: no cycle yet; run resume"))
        cycle = registry.get_cycle(state.cycle_id)
        view = CycleView(registry, state.cycle_id,
                         EvaluationEngine(registry, state.cycle_id, Protocol.from_dict(state.protocol)))
        done = len(view.iteration_rounds())
        best = view.reference()
        score = "none yet" if best is None or view.score(best) is None else (
            f"{view.score(best):.5f} ({best.attempt.id})")
        lines += [f"state: {cycle.state}; round {done} of {cycle.iterations}", f"best score so far: {score}"]
        last = paths.folder / LAST_RUN
        if last.exists() and cycle.state != STOPPED:
            info = json.loads(last.read_text(encoding="utf-8"))
            if info.get("outcome") == "suspended":
                lines.append(f"last run: SUSPENDED in {info['state']}: {info['reason']} -> run resume")
        pending = describe_pending(cycle.state, registry, state.cycle_id, config.lead_path,
                                   Ledger(paths.folder / LEDGER_FILE))
        if cycle.state == AWAITING_PLAN_APPROVAL:
            lines += ["", *_plan_lines(registry, state.cycle_id)]
        if pending:
            lines += ["", *pending]
        lines += ["", f"lead file: {config.lead_path} ({'exists' if config.lead_path.exists() else 'not created yet'})",
                  f"notices: {paths.folder / NOTICES_FILE}",
                  f"registry: {paths.registry}"]
        report = report_path(paths.out, state.cycle_id)
        if report.exists():
            lines.append(f"report: {report}")
        if cycle.state == STOPPED:
            lines.append(f"The cycle is finished. To send a submission: {_cmd(config, 'submit')} (a dry run), "
                         "then add --confirm-submit.")
        for entry in registry.list_submissions(state.cycle_id):
            shown = "pending" if entry.public_score is None else f"{entry.public_score}"
            lines.append(f"submission {entry.submission_id}: {entry.status}, public score {shown}")
        try:
            marker = read_marker(marker_path(paths.out, state.cycle_id))
        except OrchestratorError as error:
            marker = None
            lines.append(f"submission marker: unreadable ({error})")
        if marker is not None:
            lines.append(f"submission marker: {marker.get('state')} (sha256 {marker.get('sha256', '?')[:12]})")
        return CommandResult(codes.FINISHED, tuple(lines))
    finally:
        registry.close()
