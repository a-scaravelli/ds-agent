"""Retry or re-evaluate an attempt: a NEW execution on the same bundle, under the frozen protocol (ticket 30).

``rerun`` takes the run lock (the registry is written only here) and creates a NEW attempt with ``round_id`` None,
``parent_id`` of the origin and the SAME bundle: a fresh execution through ``score_bundle`` (``engine.evaluate_validation``)
with a seed-shifting candidate wrapper. The original attempt is never touched. The fingerprint includes
``seed_offset`` and ``rerun_of``, so a repeat is a duplicate and is reported as such, naming the existing attempt.
The execution config records ``seed_offset`` and ``rerun_of``: the label 're-run with another seed' comes from it.

The seed shift is applied by ``SeedShiftCandidate`` over the seed the engine derives (no ``evaluation/`` edit): it
varies the model seed only, the folds are identical. A re-run is not part of any round, nor of N, K, M or the
improvement chain.
"""

import dataclasses
import json
from pathlib import Path

import pandas as pd

from kaggle_agent.bundle import BundleRunner, load_bundle, score_bundle
from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.loop import ids
from kaggle_agent.loop.candidate import content_hash
from kaggle_agent.registry import DuplicateAttemptError, NotFoundError, RegistryError, SqliteRegistry, compute_fingerprint
from kaggle_agent.sandbox import SandboxPolicy

from .applied import mark
from .cockpit import CockpitFiles, read_jsonl
from .config import Config
from .errors import OrchestratorError, ProjectError
from .bundle_jobs import _state


class SeedShiftCandidate:
    """A candidate whose model seed is shifted by ``offset`` (``bundle`` and the rest pass through)."""

    def __init__(self, inner, offset: int) -> None:
        self._inner, self._offset = inner, offset

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def fit_predict(self, x_train, y_train, x_valid, seed):
        return self._inner.fit_predict(x_train, y_train, x_valid, seed + self._offset)


def rerun_id(origin_id: str, offset: int) -> str:
    return f"{origin_id}-retry" if offset == 0 else f"{origin_id}-rr{offset}"


def _decision(files: CockpitFiles, decision_id: str | None) -> dict | None:
    if not decision_id:
        return None
    records, _ = read_jsonl(files.decisions)
    return next((r for r in records if r.get("id") == decision_id and r.get("kind") == "rerun"), None)


def rerun_job(config: Config, sandbox, files: CockpitFiles, attempt_id: str, offset: int,
              decision_id: str | None = None) -> dict:
    base, state = _state(config)
    paths = config.paths
    record = _decision(files, decision_id)
    try:
        registry = SqliteRegistry(str(paths.registry))
    except RegistryError as error:
        raise ProjectError(str(error)) from None
    try:
        try:
            origin = registry.get_attempt(attempt_id)
        except NotFoundError:
            return {"ok": False, "error": f"there is no attempt {attempt_id}"}
        if origin.cycle_id != state.cycle_id:
            return {"ok": False, "error": f"attempt {attempt_id} belongs to another run"}
        if not origin.closed or not origin.executions:
            return {"ok": False, "error": f"attempt {attempt_id} is not finished yet"}
        if origin.outcome == "succeeded" and offset == 0:
            return {"ok": False, "error": "a succeeded attempt is re-run with another seed: use an offset of 1 or more"}
        bundle_id = origin.executions[-1].config.get("bundle_id") or origin.attempt.bundle_id
        if not bundle_id:
            return {"ok": False, "error": f"attempt {attempt_id} has no bundle to run again"}
        new_id = rerun_id(attempt_id, offset)
        config_dict = {"bundle_id": bundle_id, "seed_offset": offset, "rerun_of": attempt_id}
        loaded = load_bundle(Path(registry.get_bundle(bundle_id).reference))
        cycle = registry.get_cycle(state.cycle_id)
        fingerprint = compute_fingerprint(config_dict, content_hash(loaded.folder), state.data_hash,
                                          cycle.protocol_hash)
        existing = new_id if _exists(registry, new_id) else None
        if existing is None:
            attempt = dataclasses.replace(origin.attempt, id=new_id, phase="iteration", parent_id=attempt_id,
                                          bundle_id=bundle_id)
            try:
                registry.add_attempt(state.cycle_id, attempt, fingerprint, round_id=None)
            except DuplicateAttemptError as error:
                existing = error.existing_attempt_id
        if existing is not None:
            if record is not None:
                mark(files, None, record, "applied", reason=f"already run as {existing}")
            return {"ok": False, "duplicate": True, "existing": existing,
                    "error": f"this re-run already exists as attempt {existing}: nothing was run again"}
        protocol = Protocol.from_dict(state.protocol)
        engine = EvaluationEngine(registry, state.cycle_id, protocol)
        policy = SandboxPolicy(timeout_seconds=config.runtime.bundle_timeout, hidden_paths=(paths.registry,))
        runner = BundleRunner(sandbox, policy, loaded, target_column=state.target, input_kind=state.input_kind)
        score = score_bundle(engine, SeedShiftCandidate(runner, offset), pd.read_csv(paths.data / state.train_file),
                             attempt_id=new_id, execution_id=ids.execution_id(new_id, 1),
                             config={**config_dict, "round_id": None}, data_hash=state.data_hash)
        registry.add_execution(score.execution)
        execution = score.execution
        if execution.outcome == "succeeded":
            metric = next(m for m in execution.metrics if m.partition == "validation")
            conclusion = (f"Re-run of {attempt_id} with seed offset {offset}: validation {protocol.metric} "
                          f"{metric.mean:.4f} (dispersion {metric.dispersion:.4f}).")
            value = metric.mean
        else:
            conclusion = f"Re-run of {attempt_id} failed ({execution.failure_kind}): {execution.reason}"
            value = None
        registry.close_attempt(new_id, conclusion[:600])
        if record is not None:
            mark(files, None, record, "applied")
        return {"ok": True, "attempt": new_id, "origin": attempt_id, "seed_offset": offset,
                "outcome": execution.outcome, "score": value}
    finally:
        registry.close()


def _exists(registry, attempt_id: str) -> bool:
    try:
        registry.get_attempt(attempt_id)
    except NotFoundError:
        return False
    return True


def run_rerun_job(config_path: str, out: str, sandbox, files: CockpitFiles, attempt: str,
                  offset: int, decision: str | None) -> int:
    from .config import load_config
    from .project import write_atomic

    code = 0
    try:
        result = rerun_job(load_config(config_path), sandbox, files, attempt, offset, decision)
    except Exception as error:
        message = str(error) if isinstance(error, OrchestratorError) else f"{type(error).__name__}: {error}"
        result, code = {"ok": False, "error": " ".join(message.split())[:500]}, 1
    write_atomic(Path(out), json.dumps(result, ensure_ascii=False))
    return code
