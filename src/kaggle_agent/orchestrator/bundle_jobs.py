"""Bundle jobs of the cockpit as orchestrator commands: ``reproduce``, ``predict`` and ``finalize`` (ticket 30).

Each writes ONE JSON result to ``--out`` like ``validate-config`` does and exits 0 when it produced a result
(``ok: false`` is a result). They wrap the existing pipeline commands; no modeling or scoring logic lives here.

- ``reproduce`` wraps ``pipeline.command.reproduce`` (registry opened read only, sandboxed) and returns OK or the
  EXACT mismatch lines. For a bundle that is not the one of the stored report (a pick of the DS made after the
  report) the reported value is the validation score of that attempt in the registry.
- ``predict`` wraps ``pipeline.command.predict`` on an uploaded CSV: the columns are checked against the
  training header (missing columns are an error, extra ones a warning), the rows are capped, and the predictions
  come back with the id column when the sample submission has one.
- ``finalize`` persists the bundle chosen by the DS AFTER the report (the registry is written only here, under the
  run lock): persist, link the ``pipeline:`` artifact, write ``out/chosen/<cycle>.json``, refresh the report
  section. It reads no holdout.
"""

import json
import re
import types
from pathlib import Path

import pandas as pd

from kaggle_agent.bundle import BundleRunner
from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.loop import CycleView
from kaggle_agent.overrides import resolve_choice
from kaggle_agent.pipeline import (
    FinalReport,
    PersistedBundleError,
    PipelineContext,
    command,
    link_pipeline,
    load_persisted,
    persisted_root,
    runner_model_store_factory,
)
from kaggle_agent.pipeline.flow import _persist
from kaggle_agent.registry import RegistryError, SqliteRegistry
from kaggle_agent.sandbox import Sandbox, SandboxPolicy

from .applied import applied_outcomes, mark
from .build import real_sandbox
from .config import Config, load_config
from .cycles import active_state
from .errors import OrchestratorError, ProjectError
from .finalize import choice_records, chosen_path, read_chosen, usable_ids, write_chosen
from .project import read_project, write_atomic
from .report_labels import collect_labels, override_comparisons, refresh_ds_section

BUNDLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_PREDICT_ROWS = 100_000
FLOAT = r"[-+0-9.eE]+|nan|inf"


def _state(config: Config):
    base = read_project(config.paths)
    if base is None:
        raise ProjectError(f"no project in {config.paths.folder}: run start first")
    return base, active_state(config.paths, base)


def effective_model(config: Config, registry, cycle_id: str) -> dict | None:
    """The final model of the cycle: ``out/chosen/<cycle>.json`` if present, else the stored report."""
    chosen = read_chosen(config.paths.out, cycle_id)
    if chosen is not None:
        return {"attempt_id": chosen["attempt"], "bundle_id": chosen["bundle"], "source": "the DS",
                "persisted": chosen.get("persisted")}
    stored = registry.get_final_report(cycle_id)
    if stored is None:
        return None
    report = FinalReport.from_dict(stored.content)
    if report.chosen is None or report.persisted_path is None:
        return None
    return {"attempt_id": report.chosen["attempt_id"], "bundle_id": report.chosen["bundle_id"],
            "source": report.chosen.get("chosen_by", "the program"), "persisted": report.persisted_path}


def _open_ro(config: Config) -> SqliteRegistry:
    try:
        return SqliteRegistry(str(config.paths.registry), read_only=True)
    except RegistryError as error:
        raise ProjectError(str(error)) from None


def _root(config: Config, registry, cycle_id: str, bundle: str | None) -> tuple[Path, dict | None]:
    if bundle is not None:
        if not BUNDLE_ID.fullmatch(bundle):
            raise ProjectError("bad bundle id")
        return persisted_root(config.paths.out, bundle), None
    model = effective_model(config, registry, cycle_id)
    if model is None:
        raise ProjectError("the cycle has no persisted bundle yet")
    return Path(model["persisted"]), model


# -- reproduce ---------------------------------------------------------------------------------------


def _numbers(lines) -> dict:
    out = {}
    for line in lines:
        found = re.match(rf"(reproduced|reported) validation [^:]+: ({FLOAT})$", line)
        if found:
            out[found.group(1)] = float(found.group(2))
    return out


def reproduce_job(config: Config, sandbox: Sandbox, bundle: str | None = None) -> dict:
    base, state = _state(config)
    registry = _open_ro(config)
    try:
        root, _ = _root(config, registry, state.cycle_id, bundle)
        try:
            persisted = load_persisted(root)
        except PersistedBundleError as error:  # the exact text of the pipeline error
            return {"ok": False, "mismatch": f"error: {error}", "lines": [f"error: {error}"], "code": 1}
        stored = registry.get_final_report(state.cycle_id)
        reported_attempt = None if stored is None or not stored.content.get("chosen") else \
            stored.content["chosen"]["attempt_id"]
        data = config.paths.data / state.train_file
        timeout = config.runtime.bundle_timeout
        if persisted.meta["attempt_id"] == reported_attempt:
            result = command.reproduce(root, config.paths.registry, data, sandbox, timeout)
            lines = list(result.lines)
            code = result.code
        else:
            code, lines = _reproduce_other(registry, persisted, data, sandbox, timeout)
    finally:
        registry.close()
    mismatch = next((l for l in lines if l.startswith("error:")), None)
    return {"ok": code == 0, "code": code, "lines": lines, "mismatch": mismatch, **_numbers(lines),
            "bundle": persisted.bundle_id, "attempt": persisted.meta["attempt_id"]}


def _reproduce_other(registry, persisted, data: Path, sandbox: Sandbox, timeout: float) -> tuple[int, list[str]]:
    """Reproduce a persisted bundle that is not the one of the report: compare with its own registry score."""
    meta = persisted.meta
    detail = registry.get_attempt(meta["attempt_id"])
    metric = next((m for m in detail.executions[-1].metrics if m.partition == "validation"), None) \
        if detail.executions else None
    if metric is None:
        return 1, [f"error: no reported metric for attempt {meta['attempt_id']}"]
    engine = EvaluationEngine(registry, meta["cycle_id"], persisted.protocol)
    runner = BundleRunner(sandbox, SandboxPolicy(timeout_seconds=timeout), persisted.loaded(),
                          target_column=meta["target"], input_kind=meta["input_kind"])
    try:
        engine.verify()
        result = engine.evaluate_validation(
            runner, pd.read_csv(data), attempt_id=meta["attempt_id"], execution_id="reproduce",
            config={"reproduce": True}, code_hash=meta["folder_hash"], data_hash=meta["data_hash"])
    except Exception as error:
        return 1, [f"error: the metric could not be reproduced: {error}"]
    lines = [f"reproduced validation {meta['metric']}: {result.mean!r}",
             f"reported validation {meta['metric']}: {metric.mean!r}"]
    if abs(result.mean - metric.mean) <= command.TOLERANCE:
        return 0, [*lines, "OK: the stored code, protocol and data reproduce the reported metric"]
    return 1, [*lines, f"error: the metrics differ by {abs(result.mean - metric.mean):.3g}"]


# -- predict -----------------------------------------------------------------------------------------


def training_columns(config: Config, state) -> list[str]:
    """The feature columns the stored model needs: the training header without the target."""
    header = pd.read_csv(config.paths.data / state.train_file, nrows=0)
    return [str(c) for c in header.columns if str(c) != state.target]


def id_column_of(config: Config, frame_columns) -> str | None:
    for sample in sorted(config.paths.data.glob("sample_submission*.csv")):
        first = str(pd.read_csv(sample, nrows=0).columns[0])
        return first if first in frame_columns else None
    return None


def predict_job(config: Config, sandbox: Sandbox, input_csv: Path, bundle: str | None = None) -> dict:
    base, state = _state(config)
    registry = _open_ro(config)
    try:
        root, _ = _root(config, registry, state.cycle_id, bundle)
    finally:
        registry.close()
    try:
        frame = pd.read_csv(input_csv, nrows=MAX_PREDICT_ROWS + 1)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return {"ok": False, "error": f"the file is not a readable CSV: {error}"}
    if len(frame) > MAX_PREDICT_ROWS:
        return {"ok": False, "error": f"the file has more than {MAX_PREDICT_ROWS} rows"}
    needed = training_columns(config, state)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        return {"ok": False, "error": "missing columns: " + ", ".join(missing), "missing": missing}
    extra = [str(c) for c in frame.columns if str(c) not in needed]
    work = Path(input_csv).with_suffix(".predictions.json")
    try:
        result = command.predict(root, input_csv, work, sandbox, config.runtime.bundle_timeout)
        if result.code != 0:
            return {"ok": False, "error": " ".join(result.lines)}
        predictions = json.loads(work.read_text(encoding="utf-8"))
    finally:
        work.unlink(missing_ok=True)
    id_name = id_column_of(config, list(map(str, frame.columns)))
    return {"ok": True, "rows": len(frame), "predictions": predictions, "id_column": id_name,
            "ids": None if id_name is None else [_cell(v) for v in frame[id_name].tolist()],
            "warnings": [f"extra column ignored: {c}" for c in extra]}


def _cell(value):
    return value if isinstance(value, (int, float, str, bool)) or value is None else str(value)


# -- finalize ----------------------------------------------------------------------------------------


def finalize_job(config: Config, sandbox: Sandbox, files) -> dict:
    """Persist the model chosen by the DS after the report (needs the run lock; writes the registry)."""
    base, state = _state(config)
    paths = config.paths
    try:
        registry = SqliteRegistry(str(paths.registry))
    except RegistryError as error:
        raise ProjectError(str(error)) from None
    try:
        cycle = registry.get_cycle(state.cycle_id)
        stored = registry.get_final_report(state.cycle_id)
        if cycle.state not in ("AWAITING_REVIEW", "STOPPED") or stored is None:
            return {"ok": False, "error": "the cycle has not reported yet: a pick made now is applied at the "
                                          "report, no job is needed"}
        protocol = Protocol.from_dict(state.protocol)
        engine = EvaluationEngine(registry, state.cycle_id, protocol)
        view = CycleView(registry, state.cycle_id, engine)
        report = FinalReport.from_dict(stored.content)
        program = None if report.chosen is None else report.chosen["attempt_id"]
        chosen_id, record = resolve_choice(program, choice_records(files, state.cycle_id, base.cycle_id),
                                           usable_ids(view, registry, state.cycle_id),
                                           applied_outcomes(files))
        if record is None:
            return {"ok": False, "error": "no valid pick of the DS for this cycle"}
        detail = view.detail(chosen_id)
        frame = pd.read_csv(paths.data / state.train_file)
        policy = SandboxPolicy(timeout_seconds=config.runtime.bundle_timeout, hidden_paths=(paths.registry,))
        pctx = PipelineContext(paths.out, runner_model_store_factory(sandbox, policy, target_column=state.target,
                                                                     input_kind=state.input_kind))
        ctx = types.SimpleNamespace(registry=registry, cycle_id=state.cycle_id, engine=engine, frame=frame,
                                    data_hash=state.data_hash)
        persisted, _, kind, error = _persist(ctx, pctx, detail)
        if persisted is None:
            return {"ok": False, "error": f"the chosen bundle could not be persisted ({kind}): {error}"}
        link_pipeline(registry, chosen_id, persisted.root)
        write_chosen(paths.out, state.cycle_id, {
            "attempt": chosen_id, "bundle": persisted.bundle_id, "persisted": str(persisted.root),
            "reason": record.get("reason", ""), "program_choice": program, "decision": record["id"]})
        mark(files, None, record, "applied")
        labels = collect_labels(files, paths.folder, registry, state.cycle_id, base.cycle_id)
        refresh_ds_section(paths.out, state.cycle_id, labels, {"attempt": chosen_id, "program_choice": program},
                           override_comparisons(registry, state.cycle_id),
                           "the holdout figure of this report belongs to the program's own pick: the engine "
                           "allows one holdout session per cycle")
        return {"ok": True, "attempt": chosen_id, "bundle": persisted.bundle_id, "program_choice": program}
    finally:
        registry.close()


def run_bundle_job(name: str, config_path: str, out: str, sandbox: Sandbox | None, *, bundle=None,
                   input_csv=None, files=None) -> int:
    """Compute one bundle job and write its JSON result to ``out``; 0 when a result was written."""
    code = 0
    try:
        config = load_config(config_path)
        sandbox = sandbox or real_sandbox()
        if name == "reproduce":
            result = reproduce_job(config, sandbox, bundle)
        elif name == "predict":
            result = predict_job(config, sandbox, Path(input_csv), bundle)
        else:
            result = finalize_job(config, sandbox, files)
    except Exception as error:
        message = str(error) if isinstance(error, OrchestratorError) else f"{type(error).__name__}: {error}"
        result, code = {"ok": False, "error": " ".join(message.split())[:500]}, 1
    write_atomic(Path(out), json.dumps(result, ensure_ascii=False))
    return code
