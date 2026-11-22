"""The commands ``start``, ``resume`` and ``status`` (``submit`` lives in ``submission``).

Each returns ``CommandResult(code, lines)``. Exit codes (see ``codes``): 0 the cycle finished, 10 it
waits for the lead, 11 it was suspended (quota, rate limit or an expired Claude login: run ``resume`` later,
after ``claude auth login`` in the last case), 12 the lead file
is invalid, 13 an unexpected crash with the registry intact (run ``resume``), 2 usage or
configuration error, 1 any other error. ``status`` needs no network and no Claude: it only reads the
project folder and the registry (opened read-only).
"""

import json
from datetime import timezone
from pathlib import Path

from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.llm import is_authentication_reason
from kaggle_agent.loop import CycleView
from kaggle_agent.loop.machine import STATUS_STOPPED, STATUS_SUSPENDED, STATUS_WAITING
from kaggle_agent.loop.states import AWAITING_PLAN_APPROVAL, STOPPED
from kaggle_agent.pipeline import report_path, run_pipeline
from kaggle_agent.registry import Registry, RegistryError, SqliteRegistry

from . import codes
from .build import Built, Dependencies, build_context
from .cockpit import CockpitFiles, PauseRequested
from .codes import CommandResult
from .config import Config
from .cycles import (
    CHANGED,
    SAME,
    active_state,
    add_entry,
    build_carry,
    entry_by_request,
    entry_for,
    fold_notes,
    new_cycle_id,
    read_cycles,
    write_carry,
)
from .errors import ConfigError, ForkRefused, LeadFileError, OrchestratorError, ProjectError
from .events import NULL, Emitter, EventLog
from .cockpit import read_jsonl, utc_text
from .lead_file import Ledger, describe_pending
from .project import EXAMPLE_FILE, LEDGER_FILE, NOTICES_FILE, ledger_name_for, read_project, write_atomic
from .submission import marker_path, read_marker

def make_emitter(cockpit: CockpitFiles) -> Emitter:
    """The emitter of a supervised run (only ``service``, ``build`` and ``wrappers`` touch the event log)."""
    return Emitter(EventLog(cockpit.events))


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
    first = read_project(config.paths).cycle_id
    return lines + describe_pending(state, ctx.registry, ctx.cycle_id, config.lead_path,
                                    Ledger(config.paths.folder / ledger_name_for(ctx.cycle_id, first)))


def run_cycle(config: Config, deps: Dependencies, cockpit: CockpitFiles | None = None,
              emitter: Emitter = NULL) -> CommandResult:
    """Build the context and run the cycle until it stops, waits for the lead or is suspended.

    With ``cockpit`` (a supervised run) events are emitted and a pause request ends the run with
    ``codes.PAUSED`` at the next safe checkpoint.
    """
    built: Built | None = None
    try:
        built = build_context(config, deps, cockpit, emitter)
        emitter.emit("run_start", command="run", state=built.registry.get_cycle(built.ctx.cycle_id).state)
        result = run_pipeline(built.ctx, built.pctx)
        outcome = result.outcome
        notices = [f"notice: {text}" for text in built.lead.emitted]
        cycle_id = built.ctx.cycle_id
        if outcome.status == STATUS_STOPPED:
            emitter.emit("finished", reason=outcome.reason, state=outcome.state)
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
        emitter.emit("suspended", kind="login" if is_authentication_reason(outcome.reason) else "quota",
                     state=outcome.state)
        if is_authentication_reason(outcome.reason):
            return CommandResult(codes.SUSPENDED, (
                f"The cycle {cycle_id} is SUSPENDED in {outcome.state}: Claude CLI is not logged in "
                "or its login expired: run `claude auth login` in a normal Terminal, then run resume",
                f"Nothing was lost. After logging in, run: {_cmd(config, 'resume')}",
                *notices))
        return CommandResult(codes.SUSPENDED, (
            f"The cycle {cycle_id} is SUSPENDED in {outcome.state}: {outcome.reason}",
            f"Nothing was lost. When the quota or rate limit allows it, run: {_cmd(config, 'resume')}",
            *notices))
    except PauseRequested:
        state = built.registry.get_cycle(built.ctx.cycle_id).state if built is not None else ""
        _write_last_run(config, deps, "paused", state, "paused by the DS")
        emitter.emit("paused", state=state)
        return CommandResult(codes.PAUSED, (
            f"The run is PAUSED in {state} at a safe checkpoint, as requested. Nothing was lost.",
            f"To continue, run: {_cmd(config, 'resume')} (or press Resume in the cockpit)."))
    except LeadFileError as error:
        emitter.emit("error", kind="lead_file_invalid", type="LeadFileError")
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
        emitter.emit("error", kind="crash", type=type(error).__name__)
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


def start(config: Config, deps: Dependencies, cockpit: CockpitFiles | None = None,
          emitter: Emitter = NULL) -> CommandResult:
    """Create the project and the cycle and run; a second start changes nothing and calls nothing."""
    base = read_project(config.paths) if config.paths.folder.exists() else None
    state = None if base is None else active_state(config.paths, base)
    if state is not None:
        shown = status(config)
        entry = entry_for(read_cycles(config.paths.folder), state.cycle_id)
        origin = "" if entry is None else (f", forked from {entry['origin']['cycle_id']}"
                                           + (f" attempt {entry['origin']['attempt_id']}"
                                              if entry["origin"].get("attempt_id") else ""))
        return CommandResult(codes.FINISHED, (
            f"The project is already started (active cycle {state.cycle_id}{origin}): no second cycle is created.",
            "Use resume to continue it, status to see where it stands.", "", *shown.lines))
    config.paths.folder.mkdir(parents=True, exist_ok=True)
    example = config.paths.folder / EXAMPLE_FILE
    if not example.exists():
        example.write_text(EXAMPLE_TEXT, encoding="utf-8")
    result = run_cycle(config, deps, cockpit, emitter)
    new = read_project(config.paths)
    if new is None:
        return result
    head = (f"competition {new.competition}: target {new.target}, metric {new.metric} ({new.direction}), "
            f"cycle {new.cycle_id}",)
    return CommandResult(result.code, head + result.lines)


def resume(config: Config, deps: Dependencies, cockpit: CockpitFiles | None = None,
           emitter: Emitter = NULL) -> CommandResult:
    """Continue a suspended, crashed or waiting cycle from the state saved in the registry."""
    if read_project(config.paths) is None:
        return CommandResult(codes.USAGE, (f"error: no project in {config.paths.folder}: run start first",))
    return run_cycle(config, deps, cockpit, emitter)


def status(config: Config) -> CommandResult:
    """State, round, best score, what the lead has to fill next and where the files are."""
    paths = config.paths
    try:
        state = read_project(paths)
    except ProjectError as error:
        return CommandResult(codes.USAGE, (f"error: {error}",))
    if state is None:
        return CommandResult(codes.USAGE, (f"error: no project in {paths.folder}: run start first",))
    first_cycle = state.cycle_id
    state = active_state(paths, state)
    entry = entry_for(read_cycles(paths.folder), state.cycle_id)
    lines = [f"competition: {state.competition}; target {state.target}; metric {state.metric} "
             f"({state.direction})", f"cycle: {state.cycle_id}"]
    if entry is not None:
        lines.append(f"origin: forked from {entry['origin']['cycle_id']}"
                     + (f" attempt {entry['origin']['attempt_id']}" if entry["origin"].get("attempt_id") else "")
                     + f"; protocol: {entry['protocol_label']}")
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
                                   Ledger(paths.folder / ledger_name_for(state.cycle_id, first_cycle)))
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


def _protocol_of(paths, base, entries: list[dict], cycle_id: str) -> tuple[dict, dict]:
    """(protocol, protocol_config) of a cycle of the project."""
    entry = entry_for(entries, cycle_id)
    return (base.protocol, base.protocol_config) if entry is None else (entry["protocol"], entry["protocol_config"])


def fork(config: Config, deps: Dependencies, from_run: str, from_attempt: str | None, request_id: str,
         threshold_k: float | None, reason: str | None, files: CockpitFiles, cockpit: CockpitFiles | None = None,
         emitter: Emitter = NULL) -> CommandResult:
    """Start a NEW cycle of the same project from an earlier run (the caller holds the run lock).

    Refused, for technical reasons only, unless the active cycle is STOPPED, the origin run and attempt exist and
    a protocol change comes with a reason. The carry snapshot and the ``cycles.json`` entry are written BEFORE
    the cycle runs and are idempotent by ``request_id``: the same launch again makes no second cycle and just
    continues the one that exists.
    """
    base = read_project(config.paths)
    if base is None:
        return CommandResult(codes.USAGE, (f"error: no project in {config.paths.folder}: run start first",))
    paths = config.paths
    entries = read_cycles(paths.folder)
    known = entry_by_request(entries, request_id)
    if known is None:
        try:
            _prepare_fork(config, base, entries, from_run, from_attempt, request_id, threshold_k, reason, files)
        except ForkRefused as refusal:
            return CommandResult(codes.ERROR, (f"refused: {refusal}", "no cycle was created."))
    return run_cycle(config, deps, cockpit, emitter)


def _prepare_fork(config, base, entries, from_run, from_attempt, request_id, threshold_k, reason, files) -> None:
    paths = config.paths
    active = active_state(paths, base)
    try:
        registry = SqliteRegistry(str(paths.registry), read_only=True)
    except RegistryError as error:
        raise ForkRefused(str(error)) from None
    try:
        known = {c.id for c in registry.list_cycles()}
        if active.cycle_id not in known or registry.get_cycle(active.cycle_id).state != STOPPED:
            raise ForkRefused(f"the current cycle {active.cycle_id} is not stopped yet: a fork starts "
                              "only after the review of the last run was accepted")
        if from_run not in known:
            raise ForkRefused(f"there is no run {from_run}")
        protocol, protocol_config = _protocol_of(paths, base, entries, from_run)
        changes = {}
        if threshold_k is not None and threshold_k != protocol["threshold_k"]:
            if not reason or not reason.strip():
                raise ForkRefused("a new protocol needs a reason: say why threshold_k changes")
            changes = {"threshold_k": [protocol["threshold_k"], threshold_k]}
            protocol = {**protocol, "threshold_k": threshold_k}
            protocol_config = {**protocol_config, "threshold_k": threshold_k}
            try:  # the evaluation package validates it again
                Protocol.from_dict(protocol)
            except Exception as error:
                raise ForkRefused(f"the new protocol is not valid: {error}") from None
        if from_attempt is not None:
            attempts = {s.attempt_id for s in registry.list_attempts(from_run)}
            if from_attempt not in attempts:
                raise ForkRefused(f"there is no attempt {from_attempt} in run {from_run}")
        same = not changes
        note = "" if same else (f"this run uses a new protocol (threshold_k {changes['threshold_k'][0]:g} to "
                                f"{changes['threshold_k'][1]:g}): the scores of the origin were measured under another "
                                "protocol and are not comparable")
        notes = fold_notes(read_jsonl(files.folder / "notes.jsonl")[0], from_run, from_attempt, base.competition)
        carry = build_carry(registry, from_run, from_attempt, base.direction, same, note, notes)
    finally:
        registry.close()
    new_id = new_cycle_id(base.competition, entries)
    write_carry(paths.folder, new_id, carry)
    add_entry(paths.folder, {
        "cycle_id": new_id, "request_id": request_id, "origin": {"cycle_id": from_run, "attempt_id": from_attempt},
        "protocol": protocol, "protocol_config": protocol_config,
        "protocol_label": SAME if same else CHANGED, "changes": changes, "reason": (reason or "").strip(),
        "who": "the DS", "created_at": utc_text()})
