"""Evaluation engine: validation scores, holdout session, comparisons, frozen-protocol check.

The engine takes metric, scheme and threshold from the ``Protocol`` and never
hardcodes them. It returns execution records but does not store them:
persisting executions belongs to the loop. It does store, in the registry, the
frozen hash, the holdout reads and the comparisons.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol as Structural

import numpy as np
import pandas as pd
import sklearn

from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import (
    ComparisonRecord,
    ExecutionRecord,
    PartitionMetric,
    Registry,
)

from .errors import ProtocolHashMismatchError, ProtocolNotFrozenError
from .hashing import evaluation_code_hash, protocol_hash
from .metrics import get_metric
from .protocol import Protocol
from .schemes import derive_seed
from .split import FrozenSplit, HoldoutAccessor
from .threshold import Comparison, compare as compare_scores, dispersion

VALIDATION = "validation"


class Candidate(Structural):
    """What the engine needs from a model: fit on train, predict on validation.

    The sandbox of a later ticket will implement it; tests use fakes.
    """

    def fit_predict(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, seed: int
    ):
        """Return predictions for ``x_valid`` (one per row)."""


@dataclass(frozen=True)
class EvaluationResult:
    """Scores per partition id and the execution record ready for ``Registry.add_execution``."""

    scores: dict[str, float]
    mean: float
    dispersion: float
    execution: ExecutionRecord


@dataclass(frozen=True)
class HoldoutResult:
    attempt_id: str
    score: float
    rows: int


def library_versions() -> dict[str, str]:
    return {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


class EvaluationEngine:
    """Evaluates candidates under the frozen protocol of one cycle."""

    def __init__(self, registry: Registry, cycle_id: str, protocol: Protocol,
                 package_dir=None) -> None:
        self.registry = registry
        self.cycle_id = cycle_id
        self.protocol = protocol
        self._package_dir = package_dir
        self._metric = get_metric(protocol.metric)

    # -- freezing ------------------------------------------------------------

    def current_hash(self) -> str:
        return protocol_hash(self.protocol, evaluation_code_hash(self._package_dir))

    def freeze(self) -> str:
        """At approval: store the hash of protocol and evaluation code on the cycle."""
        value = self.current_hash()
        self.registry.set_protocol_hash(self.cycle_id, value)
        return value

    def verify(self) -> None:
        """Block the run unless protocol and code match the hash frozen on the cycle."""
        frozen = self.registry.get_cycle(self.cycle_id).protocol_hash
        if frozen is None:
            raise ProtocolNotFrozenError(f"cycle {self.cycle_id!r} has no frozen protocol")
        current = self.current_hash()
        if current != frozen:
            raise ProtocolHashMismatchError(frozen, current)

    # -- scoring helpers -----------------------------------------------------

    def _score(self, y_true, predictions) -> float:
        predicted = np.asarray(predictions)
        if len(predicted) != len(y_true):
            raise ValidationError("predictions", "must have one value per validation row")
        return self._metric.score(y_true, predicted, self.protocol.metric_params)

    # -- validation ----------------------------------------------------------

    def evaluate_validation(self, candidate: Candidate, frame: pd.DataFrame, *, attempt_id: str,
                            execution_id: str, config: dict, code_hash: str,
                            data_hash: str) -> EvaluationResult:
        """Fit and score the candidate on every validation partition."""
        self.verify()
        started = time.perf_counter()
        split = FrozenSplit.create(frame, self.protocol)
        scores: dict[str, float] = {}
        for part in split.partitions():
            x_train, y_train = split.data(part.train_rows)
            x_valid, y_valid = split.data(part.valid_rows)
            seed = derive_seed(self.protocol.seed, "candidate", part.repetition)
            predictions = candidate.fit_predict(x_train, y_train, x_valid, seed)
            scores[part.id] = self._score(y_valid, predictions)
        mean = float(np.mean(list(scores.values())))
        spread = dispersion(scores)
        metrics = [PartitionMetric(VALIDATION, mean, spread)]
        metrics += [PartitionMetric(f"{VALIDATION}/{pid}", s, 0.0) for pid, s in scores.items()]
        record = ExecutionRecord(
            id=execution_id,
            attempt_id=attempt_id,
            config=config,
            seed=self.protocol.seed,
            code_hash=code_hash,
            data_hash=data_hash,
            protocol_hash=self.registry.get_cycle(self.cycle_id).protocol_hash,
            metrics=tuple(metrics),
            duration_seconds=time.perf_counter() - started,
            outcome="succeeded",
            resources={"library_versions": library_versions()},
        )
        return EvaluationResult(scores, mean, spread, record)

    @staticmethod
    def scores_from_execution(record: ExecutionRecord) -> dict[str, float]:
        """Per-partition scores of an execution record produced by ``evaluate_validation``."""
        prefix = f"{VALIDATION}/"
        return {m.partition[len(prefix):]: m.mean for m in record.metrics
                if m.partition.startswith(prefix)}

    # -- comparisons ---------------------------------------------------------

    def compare(self, candidate_attempt_id: str, reference_attempt_id: str,
                candidate_scores: dict[str, float], reference_scores: dict[str, float],
                baseline_scores: dict[str, float]) -> Comparison:
        """Paired comparison; the result is recorded and counted in the registry."""
        self.verify()
        result = compare_scores(
            candidate_scores, reference_scores, dispersion(baseline_scores), self.protocol
        )
        self.registry.add_comparison(ComparisonRecord(
            self.cycle_id, candidate_attempt_id, reference_attempt_id,
            result.difference, result.threshold, result.is_real,
        ))
        return result

    # -- holdout -------------------------------------------------------------

    @contextmanager
    def holdout_session(self, frame: pd.DataFrame):
        """The one holdout session of the cycle: evaluate the best attempt and requested variants.

        The session closes on exit, also when an exception leaves the block,
        and cannot be reopened in this cycle.
        """
        self.verify()
        split = FrozenSplit.create(frame, self.protocol)
        accessor = HoldoutAccessor(split, self.registry, self.cycle_id, self.verify)
        with accessor.open_session():
            yield HoldoutEvaluator(self, split, accessor)


class HoldoutEvaluator:
    """Handle valid only while its session is open."""

    def __init__(self, engine: EvaluationEngine, split: FrozenSplit,
                 accessor: HoldoutAccessor) -> None:
        self._engine = engine
        self._split = split
        self._accessor = accessor

    def evaluate_holdout(self, candidate: Candidate, attempt_id: str) -> HoldoutResult:
        """Fit on the whole development data and score the holdout (labels stay with the engine)."""
        engine = self._engine
        engine.verify()
        x_holdout, y_holdout = self._accessor.read(attempt_id)
        x_dev, y_dev = self._split.development_data()
        seed = derive_seed(engine.protocol.seed, "candidate-holdout")
        predictions = candidate.fit_predict(x_dev, y_dev, x_holdout, seed)
        return HoldoutResult(attempt_id, engine._score(y_holdout, predictions), len(y_holdout))
