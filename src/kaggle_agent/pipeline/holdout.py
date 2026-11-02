"""The one holdout evaluation of the chosen bundle, recorded as a ``HoldoutOutcome`` (design section 6).

``evaluate_chosen_holdout`` opens the holdout session of the cycle once, asks the engine to
score the persisted model of the chosen attempt, and always writes an outcome. A failed run is
a record with the failure kind and a one-line reason, never an exception. The holdout labels stay
with the engine; the bundle code only receives the features of the holdout rows in the predict
stage. If a run crashed after the session was opened but before an outcome was written, the next
call records ``failed`` (``interrupted before an outcome was recorded``) and never reads again.
"""

from collections.abc import Callable
from datetime import datetime

from kaggle_agent.bundle import BundleRunError, PersistedModelCandidate
from kaggle_agent.evaluation import EvaluationEngine, EvaluationError, HoldoutSessionError
from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import HoldoutOutcome, Registry

from .persisted import ModelStore, PersistedBundle

NO_USABLE_ATTEMPT = "no usable attempt"
INTERRUPTED = "interrupted before an outcome was recorded"
MAX_REASON = 300


def one_line(reason: object) -> str:
    """One line of at most ``MAX_REASON`` characters."""
    text = " ".join(str(reason).split()) or "unknown reason"
    return text if len(text) <= MAX_REASON else text[: MAX_REASON - 3] + "..."


def run_reason(error: BundleRunError) -> str:
    """A one-line reason for a failed bundle run, without the stderr tail of a crash."""
    if error.status == "crashed":
        return one_line(f"the bundle crashed ({error.reason.split(':')[0]})")
    return one_line(error.reason)


def _failed(engine, chosen_id, bundle_id, now, kind, reason) -> HoldoutOutcome:
    return HoldoutOutcome(
        cycle_id=engine.cycle_id, metric=engine.protocol.metric, status="failed",
        recorded_at=now().isoformat(), attempt_id=chosen_id, bundle_id=bundle_id,
        failure_kind=kind, reason=one_line(reason),
    )


def evaluate_chosen_holdout(registry: Registry, engine: EvaluationEngine, frame, *,
                            chosen_id: str | None, persisted: PersistedBundle | None,
                            store: ModelStore | None, now: Callable[[], datetime],
                            persist_error: str | None = None,
                            persist_kind: str = "persist") -> HoldoutOutcome:
    """Evaluate the chosen bundle once on the holdout and record the outcome; never raises for a run failure.

    - an outcome already recorded is returned as it is;
    - a holdout session without an outcome (a crash in between) is recorded as ``failed``;
    - no chosen attempt: ``skipped`` with reason ``no usable attempt``;
    - no persisted bundle (``persist_error`` says why): ``failed`` with ``persist_kind``;
    - otherwise the engine opens the session and scores ``PersistedModelCandidate``; a
      ``BundleRunError``, ``ValidationError``, ``HoldoutSessionError`` or evaluation error
      (a changed protocol hash) is recorded as ``failed``.
    """
    cycle_id = engine.cycle_id
    existing = registry.get_holdout_outcome(cycle_id)
    if existing is not None:
        return existing
    bundle_id = persisted.bundle_id if persisted else None
    if registry.get_holdout_session(cycle_id) is not None:
        outcome = _failed(engine, chosen_id, bundle_id, now, "interrupted", INTERRUPTED)
    elif chosen_id is None:
        outcome = HoldoutOutcome(
            cycle_id=cycle_id, metric=engine.protocol.metric, status="skipped",
            recorded_at=now().isoformat(), reason=NO_USABLE_ATTEMPT,
        )
    elif persisted is None or store is None:
        outcome = _failed(engine, chosen_id, None, now, persist_kind,
                          persist_error or "the chosen bundle could not be persisted")
    else:
        outcome = _evaluate(engine, frame, chosen_id, persisted, store, now)
    registry.add_holdout_outcome(outcome)
    return outcome


def _evaluate(engine, frame, chosen_id, persisted, store, now) -> HoldoutOutcome:
    candidate = PersistedModelCandidate(store, persisted.model_dir)
    bundle_id = persisted.bundle_id
    try:
        with engine.holdout_session(frame) as session:
            result = session.evaluate_holdout(candidate, chosen_id)
    except BundleRunError as error:
        return _failed(engine, chosen_id, bundle_id, now, error.failure_kind, run_reason(error))
    except ValidationError as error:
        return _failed(engine, chosen_id, bundle_id, now, "validation", error)
    except HoldoutSessionError as error:
        return _failed(engine, chosen_id, bundle_id, now, "holdout_session", error)
    except EvaluationError as error:  # the protocol hash changed or was never frozen
        return _failed(engine, chosen_id, bundle_id, now, "protocol", error)
    return HoldoutOutcome(
        cycle_id=engine.cycle_id, metric=engine.protocol.metric, status="succeeded",
        recorded_at=now().isoformat(), attempt_id=chosen_id, bundle_id=bundle_id,
        score=result.score, rows=result.rows,
    )
