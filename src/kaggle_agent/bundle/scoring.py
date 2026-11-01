"""Scoring and registration helpers for bundles (design sections 7 and 8).

Nothing here stores executions or builds rounds: that is the round pipeline (step 9).
"""

import dataclasses
import time
from dataclasses import dataclass

import pandas as pd

from kaggle_agent.evaluation import EvaluationEngine, EvaluationResult
from kaggle_agent.proposals import Attempt
from kaggle_agent.registry import BundleRecord, ExecutionRecord, Registry

from .errors import BundleRunError
from .layout import LoadedBundle, schema_of
from .runner import BundleRunner


@dataclass(frozen=True)
class BundleScore:
    """The outcome of scoring a bundle: ``execution`` is always set, ``result`` only on success."""

    execution: ExecutionRecord
    result: EvaluationResult | None = None
    error: BundleRunError | None = None


def score_bundle(engine: EvaluationEngine, runner: BundleRunner, frame: pd.DataFrame, *,
                 attempt_id: str, execution_id: str, config: dict, data_hash: str) -> BundleScore:
    """Score a bundle with the unchanged engine; run failures become a failed ``ExecutionRecord``.

    The metric is always computed by the engine from the validated predictions. Only
    ``BundleRunError`` is caught: a protocol hash problem propagates. ``resources`` holds the
    sandbox status and whether the failure may be retried as a code fix.
    """
    folder_hash = runner.bundle.folder_hash
    started = time.perf_counter()
    try:
        result = engine.evaluate_validation(
            runner, frame, attempt_id=attempt_id, execution_id=execution_id, config=config,
            code_hash=folder_hash, data_hash=data_hash,
        )
    except BundleRunError as error:
        record = ExecutionRecord(
            id=execution_id,
            attempt_id=attempt_id,
            config=config,
            seed=engine.protocol.seed,
            code_hash=folder_hash,
            data_hash=data_hash,
            protocol_hash=engine.registry.get_cycle(engine.cycle_id).protocol_hash,
            metrics=(),
            duration_seconds=time.perf_counter() - started,
            outcome="failed",
            failure_kind=error.failure_kind,
            reason=error.reason,
            resources={"sandbox_status": error.status, "retryable": error.retryable_as_code_fix},
        )
        return BundleScore(record, error=error)
    return BundleScore(result.execution, result=result)


def register_bundle(registry: Registry, loaded: LoadedBundle, cycle_id: str,
                    reference: str) -> BundleRecord:
    """Store a ``BundleRecord`` (id, folder hash, reference, schema) in the registry."""
    record = BundleRecord(
        id=loaded.bundle_id, cycle_id=cycle_id, folder_hash=loaded.folder_hash,
        reference=reference, schema=schema_of(loaded.manifest),
    )
    registry.add_bundle(record)
    return record


def attempt_with_bundle(attempt: Attempt, bundle_id: str) -> Attempt:
    """The attempt with ``bundle_id`` set; add it to the registry after the bundle is stored."""
    return dataclasses.replace(attempt, bundle_id=bundle_id)
