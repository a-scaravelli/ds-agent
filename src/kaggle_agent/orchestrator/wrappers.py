"""Wrappers that inject events, pause and the effective limits without editing ``loop/`` or ``evaluation/``.

All of them are optional (``build_context`` adds them only for a supervised run) and none changes a
result: they delegate every call, emit small events AFTER the call succeeded and never read back
what they wrote.

- ``EventingRegistry``: delegates to the real registry; emits events after each successful write,
  raises ``PauseRequested`` right after a commit to a WORKING state when a pause is pending, and
  overlays the DS's limit changes on ``get_cycle`` (rounds, stagnation and the stored limit keys only).
- ``eventing_roles`` / ``EventingWriter``: stage start/end around every role and writer call, and the
  pause check before each of them.
- ``EventingCandidate``: counts ``fit_predict`` calls against the partitions of the frozen split and
  emits fold start/end with the partial score of the validation fold only (never the holdout).
"""

import dataclasses
import time

from kaggle_agent.evaluation import FrozenSplit, get_metric
from kaggle_agent.loop import RoleSet
from kaggle_agent.loop.states import (
    BASELINE,
    ITERATING,
    PLANNING,
    REPORTING,
    SPLITTING,
)

from .cockpit import CockpitFiles, PauseRequested, overlay_cycle, read_jsonl
from .events import Emitter

WORKING = (PLANNING, SPLITTING, BASELINE, ITERATING, REPORTING)


class EventingRegistry:
    """A delegating wrapper of the registry (not a subclass: every other method passes through)."""

    def __init__(self, inner, emitter: Emitter, files: CockpitFiles) -> None:
        self._inner = inner
        self._emitter = emitter
        self._files = files
        self._stage: str | None = None
        self._stage_started = 0.0
        self._outcomes: dict[str, str] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def check_pause(self) -> None:
        if self._files.pause_pending():
            raise PauseRequested()

    # -- reads -------------------------------------------------------------------

    def get_cycle(self, cycle_id: str):
        cycle = self._inner.get_cycle(cycle_id)
        if cycle.parameters is None:
            return cycle
        records, _ = read_jsonl(self._files.limit_changes)
        return overlay_cycle(cycle, records) if records else cycle

    # -- writes ------------------------------------------------------------------

    def set_cycle_state(self, cycle_id: str, state: str) -> None:
        self._inner.set_cycle_state(cycle_id, state)
        now = time.perf_counter()
        if self._stage is not None:
            self._emitter.emit("stage_end", stage=self._stage, outcome="ok",
                               seconds=now - self._stage_started)
        self._emitter.emit("cycle_state", state=state)
        if state in WORKING:
            self._stage, self._stage_started = state.lower(), now
            self._emitter.emit("stage_start", stage=self._stage)
            self.check_pause()
        else:
            self._stage = None

    def add_round(self, cycle_id: str, round):
        result = self._inner.add_round(cycle_id, round)
        self._emitter.emit("round_start", round=round.id)
        return result

    def add_attempt(self, cycle_id, attempt, fingerprint, round_id=None):
        result = self._inner.add_attempt(cycle_id, attempt, fingerprint, round_id=round_id)
        self._emitter.emit("candidate_start", attempt=attempt.id, round=round_id)
        return result

    def add_execution(self, execution):
        result = self._inner.add_execution(execution)
        self._outcomes[execution.attempt_id] = execution.outcome
        return result

    def close_attempt(self, attempt_id: str, conclusion: str) -> None:
        self._inner.close_attempt(attempt_id, conclusion)
        self._emitter.emit("candidate_end", attempt=attempt_id,
                           outcome=self._outcomes.pop(attempt_id, None))

    def add_guidance(self, cycle_id, guidance):
        self._inner.add_guidance(cycle_id, guidance)
        self._emitter.emit("guidance_recorded", round=guidance.round_id)

    def add_llm_call(self, entry):
        result = self._inner.add_llm_call(entry)
        self._emitter.emit("llm_call", role=entry.role, outcome=entry.outcome,
                           input_tokens=entry.input_tokens, output_tokens=entry.output_tokens,
                           seconds=entry.duration_seconds, model=entry.model)
        return result


def _timed(emitter: Emitter, registry: EventingRegistry, stage: str, role: str | None, call, *args):
    registry.check_pause()
    emitter.emit("stage_start", stage=stage, role=role)
    started = time.perf_counter()
    outcome = "ok"
    try:
        return call(*args)
    except BaseException as error:
        outcome = type(error).__name__
        raise
    finally:
        emitter.emit("stage_end", stage=stage, role=role, outcome=outcome,
                     seconds=time.perf_counter() - started)


def eventing_roles(roles: RoleSet, emitter: Emitter, registry: EventingRegistry) -> RoleSet:
    """Every role callable wrapped with stage events and the pause check."""
    def wrap(name, call):
        return lambda *args: _timed(emitter, registry, name, name, call, *args)

    return RoleSet(**{f.name: wrap(f.name, getattr(roles, f.name)) for f in dataclasses.fields(roles)})


class EventingWriter:
    """The bundle writer with stage events (the turns of the writer arrive as ``llm_call`` events)."""

    def __init__(self, inner, emitter: Emitter, registry: EventingRegistry) -> None:
        self._inner, self._emitter, self._registry = inner, emitter, registry

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def write(self, request):
        detail = "fix" if request.fix is not None else "new"
        self._registry.check_pause()
        self._emitter.emit("stage_start", stage="writer", role="writer", detail=detail)
        started = time.perf_counter()
        outcome = "ok"
        try:
            return self._inner.write(request)
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            self._emitter.emit("stage_end", stage="writer", role="writer", outcome=outcome,
                               seconds=time.perf_counter() - started)


class EventingCandidate:
    """A ``Candidate`` that reports each fold; ``bundle`` and the rest pass through."""

    def __init__(self, inner, emitter: Emitter, ctx_frame, protocol, label: str) -> None:
        self._inner, self._emitter, self._protocol, self._label = inner, emitter, protocol, label
        self._frame = ctx_frame
        self._partitions = None
        self._count = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _split(self):
        if self._partitions is None:
            try:
                self._partitions = FrozenSplit.create(self._frame, self._protocol).partitions()
            except Exception:
                self._partitions = ()
        return self._partitions

    def _partial_score(self, partition, x_valid, predictions):
        try:
            if list(x_valid.index) != [int(i) for i in partition.valid_rows]:
                return None
            labels = self._frame.reset_index(drop=True)[self._protocol.target].iloc[partition.valid_rows]
            return float(get_metric(self._protocol.metric).score(
                labels, predictions, self._protocol.metric_params))
        except Exception:
            return None

    def fit_predict(self, x_train, y_train, x_valid, seed):
        parts = self._split()
        total = len(parts)
        index = self._count % total if total else 0
        self._count += 1
        partition = parts[index] if total else None
        info = {"bundle": self._label, "fold": index + 1, "folds": total}
        if partition is not None:
            info["partition"] = partition.id
        self._emitter.emit("fold_start", **info)
        started = time.perf_counter()
        predictions = self._inner.fit_predict(x_train, y_train, x_valid, seed)
        score = None if partition is None else self._partial_score(partition, x_valid, predictions)
        self._emitter.emit("fold_end", seconds=time.perf_counter() - started, score=score, **info)
        return predictions
