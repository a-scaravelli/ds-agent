"""Frozen split: validation partitions for the caller, the holdout behind a logged accessor.

Guarantee of this module (design sections 6 and 8): the split object exposes
only development data (train and validation partitions); the holdout is
reachable through ``HoldoutAccessor`` alone, which logs every read in the
registry before returning data, allows one read per attempt, and works only
inside the single holdout session of a cycle.

Limit: this is a guarantee of interface and log, not of isolation. Code
running in the same Python process could still reach private attributes.
Real isolation of the holdout from generated code comes with the sandbox
(ticket 6 of the implementation plan).
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import HoldoutRead, Registry, RegistryError

from .errors import HoldoutSessionError
from .protocol import Protocol
from .schemes import derive_seed, get_scheme


def now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Partition:
    """One train/validation partition of the development data.

    Rows are positions in the frame given to ``FrozenSplit.create``. The id is
    ``r{repetition}/f{fold}``.
    """

    id: str
    repetition: int
    fold: int
    train_rows: np.ndarray
    valid_rows: np.ndarray

    def __repr__(self) -> str:
        return f"Partition({self.id}, train={len(self.train_rows)}, valid={len(self.valid_rows)})"


class FrozenSplit:
    """Development partitions fixed by the protocol; the holdout stays private."""

    def __init__(self, frame: pd.DataFrame, protocol: Protocol, holdout: np.ndarray,
                 development: np.ndarray, partitions: tuple[Partition, ...]) -> None:
        self._features = frame.drop(columns=[protocol.target])
        self._target = frame[protocol.target]
        self._holdout_rows = holdout
        self._development_rows = development
        self._partitions = partitions

    @classmethod
    def create(cls, frame: pd.DataFrame, protocol: Protocol) -> "FrozenSplit":
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError("data", "must be a pandas DataFrame")
        if protocol.target not in frame.columns:
            raise ValidationError("target", f"column {protocol.target!r} is not in the data")
        frame = frame.reset_index(drop=True)
        holdout_scheme = get_scheme(protocol.holdout_scheme, "holdout_scheme")
        development, holdout = holdout_scheme.holdout_split(
            frame, protocol.holdout_share, protocol.holdout_params,
            derive_seed(protocol.seed, "holdout"),
        )
        scheme = get_scheme(protocol.validation_scheme, "validation_scheme")
        dev_frame = frame.iloc[development].reset_index(drop=True)
        partitions = []
        shared = None
        for repetition in range(protocol.repetitions):
            if scheme.deterministic and shared is not None:
                folds = shared
            else:
                folds = scheme.folds(
                    dev_frame, protocol.validation_params,
                    derive_seed(protocol.seed, "folds", repetition),
                )
                shared = folds
            for fold, (train, valid) in enumerate(folds):
                partitions.append(Partition(
                    f"r{repetition}/f{fold}", repetition, fold,
                    development[np.asarray(train)], development[np.asarray(valid)],
                ))
        return cls(frame, protocol, np.asarray(holdout), np.asarray(development), tuple(partitions))

    def __repr__(self) -> str:
        return (
            f"FrozenSplit(development_rows={len(self._development_rows)}, "
            f"partitions={len(self._partitions)})"
        )

    def partitions(self) -> tuple[Partition, ...]:
        return self._partitions

    def development_rows(self) -> np.ndarray:
        return self._development_rows.copy()

    def data(self, rows: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
        """Features and target of development rows; any other row is refused."""
        rows = np.asarray(rows)
        if not np.isin(rows, self._development_rows).all():
            raise ValidationError("rows", "only development rows can be read here")
        return self._features.iloc[rows].copy(), self._target.iloc[rows].copy()

    def development_data(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.data(self._development_rows)

    def _holdout_data(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._features.iloc[self._holdout_rows].copy(), self._target.iloc[self._holdout_rows].copy()


class HoldoutAccessor:
    """The only reader of the holdout; it lives inside the one session of a cycle.

    ``verify`` is called before the session opens and before every read, so a
    changed protocol blocks both. Every read is written to the registry before
    the data is returned: a read whose evaluation then fails still counts.
    """

    def __init__(self, split: FrozenSplit, registry: Registry, cycle_id: str,
                 verify: Callable[[], None]) -> None:
        self._split = split
        self._registry = registry
        self._cycle_id = cycle_id
        self._verify = verify
        self._open = False
        self._read: set[str] = set()

    @contextmanager
    def open_session(self):
        self._verify()
        try:
            self._registry.open_holdout_session(self._cycle_id, now_text())
        except RegistryError as exc:
            raise HoldoutSessionError(str(exc)) from exc
        self._open = True
        try:
            yield self
        finally:
            self._open = False
            self._registry.close_holdout_session(self._cycle_id, now_text())

    def read(self, attempt_id: str) -> tuple[pd.DataFrame, pd.Series]:
        """Return holdout features and target for the engine; never for candidate code."""
        if not self._open:
            raise HoldoutSessionError("the holdout session is not open")
        self._verify()
        if attempt_id in self._read:
            raise HoldoutSessionError(f"attempt {attempt_id!r} already read the holdout")
        features, target = self._split._holdout_data()
        try:
            self._registry.add_holdout_read(
                HoldoutRead(self._cycle_id, attempt_id, len(target), now_text())
            )
        except RegistryError as exc:
            raise HoldoutSessionError(str(exc)) from exc
        self._read.add(attempt_id)
        return features, target
