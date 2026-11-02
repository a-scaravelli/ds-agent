"""The two commands of the pipeline: ``reproduce`` and ``predict`` (design section 15, step 11).

``reproduce`` loads a persisted bundle (the hash of its code and of its model are checked there,
the only place where the stored ``model/`` is checked), opens the registry read-only, rebuilds the
protocol from ``persisted.json``, checks it against the hash frozen on the cycle and re-runs
``evaluate_validation`` with a real ``BundleRunner``. It compares the mean with the validation metric
stored in the report. The validation metric is retrained from the persisted bundle CODE, partition
by partition (fit and predict in separate processes); it does not use the stored ``model/``, which
was fitted on the whole development data and is the model of no validation partition. So it shows that
the stored code, protocol and data give the reported metric, not that the stored trained model does.
The holdout is never read again (one read per cycle): its score is read from the registry.

``predict`` is the command that uses the stored model: it runs only the predict stage on ``model/``
in a fresh process and writes the predictions as JSON, never pickle.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kaggle_agent.bundle import BundleError, BundleRunner
from kaggle_agent.evaluation import EvaluationEngine, EvaluationError
from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import RegistryError, SqliteRegistry
from kaggle_agent.sandbox import Sandbox, SandboxPolicy

from .errors import PersistedBundleError
from .persisted import PersistedBundle, load_persisted

TOLERANCE = 1e-9
DEFAULT_TIMEOUT = 600.0


@dataclass(frozen=True)
class CommandResult:
    """Exit code and the lines printed by a command."""

    code: int
    lines: tuple[str, ...]


def _runner(persisted: PersistedBundle, sandbox: Sandbox, timeout: float) -> BundleRunner:
    return BundleRunner(
        sandbox, SandboxPolicy(timeout_seconds=timeout), persisted.loaded(),
        target_column=persisted.meta["target"], input_kind=persisted.meta["input_kind"],
    )


def reproduce(persisted_dir: Path | str, db: Path | str, data: Path | str, sandbox: Sandbox,
              timeout: float = DEFAULT_TIMEOUT) -> CommandResult:
    """Re-run the frozen-protocol validation from the persisted bundle code and compare it with the report."""
    try:
        persisted = load_persisted(persisted_dir)
    except PersistedBundleError as error:
        return CommandResult(1, (f"error: {error}",))
    meta = persisted.meta
    try:
        registry = SqliteRegistry(str(db), read_only=True)
    except RegistryError as error:
        return CommandResult(2, (f"error: {error}",))
    try:
        try:
            stored = registry.get_final_report(meta["cycle_id"])
        except RegistryError as error:
            return CommandResult(1, (f"error: {error}",))
        chosen = None if stored is None else stored.content.get("chosen")
        if chosen is None:
            return CommandResult(1, (f"error: no reported metric for cycle {meta['cycle_id']}",))
        reported = float(chosen["validation_mean"])
        engine = EvaluationEngine(registry, meta["cycle_id"], persisted.protocol)
        try:
            engine.verify()
            frame = pd.read_csv(data)
            result = engine.evaluate_validation(
                _runner(persisted, sandbox, timeout), frame, attempt_id=meta["attempt_id"],
                execution_id="reproduce", config={"reproduce": True}, code_hash=meta["folder_hash"],
                data_hash=meta["data_hash"],
            )
        except (EvaluationError, ValidationError, BundleError, OSError, ValueError) as error:
            return CommandResult(1, (f"error: the metric could not be reproduced: {error}",))
    finally:
        registry.close()
    lines = (f"reproduced validation {meta['metric']}: {result.mean!r}",
             f"reported validation {meta['metric']}: {reported!r}",
             f"data hash stored with the bundle (not verified): {meta['data_hash']}")
    if abs(result.mean - reported) <= TOLERANCE:
        return CommandResult(0, (*lines, "OK: the stored code, protocol and data reproduce the reported metric"))
    return CommandResult(1, (*lines, f"error: the metrics differ by {abs(result.mean - reported):.3g}"))


def predict(persisted_dir: Path | str, input_csv: Path | str, output_json: Path | str, sandbox: Sandbox,
            timeout: float = DEFAULT_TIMEOUT) -> CommandResult:
    """Predict the rows of ``input_csv`` with the stored model in a fresh process; write JSON."""
    try:
        persisted = load_persisted(persisted_dir)
        x = pd.read_csv(input_csv)
        predictions = _runner(persisted, sandbox, timeout).predict_with_model(
            persisted.model_dir, x, persisted.seed
        )
    except (PersistedBundleError, BundleError, OSError, ValueError) as error:
        return CommandResult(1, (f"error: {error}",))
    Path(output_json).write_text(json.dumps(predictions.tolist()), encoding="utf-8")
    return CommandResult(0, (f"wrote {len(predictions)} predictions to {output_json}",))
