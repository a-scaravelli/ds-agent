"""Permutation importance as a background job (``importance --config F --attempt ID --out F [--job ID]``).

Heavy work for the cockpit, started by the web controller like ``preview`` and ``profile``. It opens the registry
READ-ONLY, runs ``EvaluationEngine.verify``, finds the attempt's last succeeded execution and its bundle (read on
this side only), builds the splits with ``FrozenSplit`` and works on DEVELOPMENT partitions only (repetition 0).
For each partition the model is fitted once in the sandbox (``BundleRunner.fit_model``), then
``predict_with_model`` scores the unshuffled and the column-shuffled validation rows (seeded with
``derive_seed``). The result goes to ``analytics/importance/<bundle>.json`` (atomic, this job is its single
writer) and the progress to ``cockpit/jobs/<job>.progress.jsonl`` (events ``job_start``, ``job_progress``,
``job_end``). It never writes the event log of the run (a live run is its single writer) and never the registry.
"""

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_agent.bundle import BundleRunner, load_bundle
from kaggle_agent.evaluation import EvaluationEngine, FrozenSplit, Protocol, derive_seed, get_metric
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.sandbox import SandboxPolicy

from .analytics_record import AnalyticsWriter, write_content
from .cockpit import append_jsonl
from .config import load_config
from .errors import OrchestratorError
from .project import read_project, write_atomic

MAX_FEATURES = 50
REPEATS = 3
DEFAULT_MAX_SECONDS = 3000.0
ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
JOB_ID = re.compile(r"[a-f0-9]{12,32}")


def progress_path(folder, job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise ValueError("bad job id")
    return Path(folder) / "cockpit" / "jobs" / f"{job_id}.progress.jsonl"


class _Progress:
    def __init__(self, folder, job_id) -> None:
        self.path = progress_path(folder, job_id) if job_id else None

    def emit(self, type_: str, **data) -> None:
        if self.path is not None:
            append_jsonl(self.path, {"type": type_, "at": time.time(), **data})


def _bundle_of(registry, attempt_id: str) -> str:
    detail = registry.get_attempt(attempt_id)
    done = [e for e in detail.executions if e.outcome == "succeeded" and (e.config or {}).get("bundle_id")]
    if not done:
        raise OrchestratorError(f"attempt {attempt_id!r} has no succeeded execution with a recorded bundle")
    return done[-1].config["bundle_id"]


def compute_importance(config, attempt_id: str, job_id: str | None, sandbox, max_seconds: float = DEFAULT_MAX_SECONDS,
                       max_features: int = MAX_FEATURES, repeats: int = REPEATS) -> dict:
    paths = config.paths
    state = read_project(paths)
    if state is None:
        raise OrchestratorError("the project is not started: there is nothing to analyse")
    protocol = Protocol.from_dict(state.protocol)
    progress = _Progress(paths.folder, job_id)
    registry = SqliteRegistry(str(paths.registry), read_only=True)
    try:
        EvaluationEngine(registry, state.cycle_id, protocol).verify()
        bundle_id = _bundle_of(registry, attempt_id)
        record = registry.get_bundle(bundle_id)
        loaded = load_bundle(Path(record.reference))
    finally:
        registry.close()
    frame = pd.read_csv(paths.data / state.train_file)
    split = FrozenSplit.create(frame, protocol)
    parts = [p for p in split.partitions() if p.repetition == 0]
    policy = SandboxPolicy(timeout_seconds=config.runtime.bundle_timeout, hidden_paths=(paths.registry,))
    runner = BundleRunner(sandbox, policy, loaded, target_column=state.target, input_kind=state.input_kind)
    metric = get_metric(protocol.metric)
    sign = 1.0 if protocol.direction == "maximize" else -1.0
    features = [str(c) for c in split.data(parts[0].valid_rows)[0].columns][:max_features]
    total = len(parts) * len(features)
    progress.emit("job_start", kind="importance", attempt=attempt_id, bundle_id=bundle_id, total=total)
    drops: dict[str, list[float]] = {f: [] for f in features}
    done, status, started = 0, "done", time.monotonic()
    try:
        for part in parts:
            x_train, y_train = split.data(part.train_rows)
            x_valid, y_valid = split.data(part.valid_rows)
            seed = derive_seed(protocol.seed, "candidate", part.repetition)
            model_dir = Path(tempfile.mkdtemp(prefix="importance-model-")).resolve()
            try:
                runner.fit_model(x_train, y_train, seed, model_dir)
                base = metric.score(y_valid, runner.predict_with_model(model_dir, x_valid, seed), protocol.metric_params)
                for feature in features:
                    if time.monotonic() - started > max_seconds:
                        status = "time_limit"
                        raise StopIteration
                    values = []
                    for repeat in range(repeats):
                        rng = np.random.default_rng(derive_seed(protocol.seed, "importance", part.id, feature, repeat))
                        shuffled = x_valid.copy()
                        shuffled[feature] = rng.permutation(x_valid[feature].to_numpy())
                        score = metric.score(y_valid, runner.predict_with_model(model_dir, shuffled, seed),
                                             protocol.metric_params)
                        values.append(sign * (base - score))
                    drops[feature].append(float(np.mean(values)))
                    done += 1
                    progress.emit("job_progress", done=done, total=total, feature=feature, fold=part.id)
            finally:
                shutil.rmtree(model_dir, ignore_errors=True)
    except StopIteration:
        pass
    rows = [{"feature": f, "importance": float(np.mean(v)), "std": float(np.std(v)) if len(v) > 1 else 0.0,
             "partitions": len(v)} for f, v in drops.items() if v]
    rows.sort(key=lambda r: (-r["importance"], r["feature"]))
    result = {"v": 1, "bundle_id": bundle_id, "attempt": attempt_id, "metric": protocol.metric,
              "direction": protocol.direction, "partitions": [p.id for p in parts], "repeats": repeats,
              "status": status, "features": rows, "skipped_features": max(0, len(split.data(parts[0].valid_rows)[0].columns) - len(features)),
              "meaning": "Importance = how much the score gets worse when one column is shuffled (validation folds)."}
    write_content(AnalyticsWriter(paths.folder).folder, f"importance/{bundle_id}.json",
                  json.dumps(result, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    progress.emit("job_end", status=status, done=done, total=total)
    return result


def run_importance(config_path: str, attempt_id: str, out: str, job_id: str | None, sandbox=None) -> int:
    """The job entry point: 0 when a result was written (``ok: false`` is a result, not a crash)."""
    code = 0
    try:
        if not ATTEMPT_ID.fullmatch(attempt_id or ""):
            raise OrchestratorError("bad attempt id")
        if job_id is not None and not JOB_ID.fullmatch(job_id):
            raise OrchestratorError("bad job id")
        config = load_config(config_path)
        if sandbox is None:
            from .build import real_sandbox
            sandbox = real_sandbox()
        result = compute_importance(config, attempt_id, job_id, sandbox)
        payload = {"ok": True, "bundle_id": result["bundle_id"], "status": result["status"],
                   "features": len(result["features"])}
    except Exception as error:
        message = str(error) if isinstance(error, OrchestratorError) else f"{type(error).__name__}: {error}"
        payload, code = {"ok": False, "error": " ".join(message.split())[:500]}, 1
        if job_id and JOB_ID.fullmatch(job_id):
            try:
                folder = load_config(config_path).paths.folder
                _Progress(folder, job_id).emit("job_end", status="failed", error=payload["error"])
            except Exception:
                pass
    write_atomic(Path(out), json.dumps(payload, ensure_ascii=False))
    return code
