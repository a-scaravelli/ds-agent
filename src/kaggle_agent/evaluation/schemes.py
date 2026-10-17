"""Extensible registry of partition schemes, selected by name from the protocol (design section 5).

The building blocks are random (optionally stratified), group-based and
chronological. No scheme is a default: the agent chooses one after analyzing
the data and the choice is frozen in the protocol.

Schemes return integer row positions of the frame they receive.
"""

import hashlib
import json
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)

from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import require_text

ROLES = ("holdout", "validation")


def derive_seed(seed: int, *labels: object) -> int:
    """Deterministic child seed of ``seed`` for a label, in the range accepted by numpy."""
    digest = hashlib.sha256(json.dumps([seed, *labels]).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def check_params(params: object, role: str, required: tuple, optional: tuple = ()) -> dict:
    """Reject non-objects, unknown keys and missing required keys."""
    if role not in ROLES:
        raise ValidationError("role", f"must be one of {', '.join(ROLES)}")
    field = f"{role}_params"
    if not isinstance(params, dict):
        raise ValidationError(field, "must be an object")
    allowed = tuple(required) + tuple(optional)
    for key in params:
        if key not in allowed:
            raise ValidationError(
                field, f"unknown parameter {key!r} (allowed: {', '.join(allowed) or 'none'})"
            )
    for key in required:
        if key not in params:
            raise ValidationError(f"{field}.{key}", "is required")
    return params


def check_fold_count(value: object, field: str = "n_folds") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValidationError(field, "must be an integer of at least 2")
    return value


def _column(frame: pd.DataFrame, name: object, purpose: str) -> pd.Series:
    if not isinstance(name, str) or name not in frame.columns:
        raise ValidationError(purpose, f"column {name!r} is not in the data")
    column = frame[name]
    if column.isna().any():
        raise ValidationError(purpose, f"column {name!r} has missing values")
    return column


class SplitScheme(ABC):
    """A way of separating rows into train and validation (or development and holdout)."""

    name: str
    deterministic: bool

    @abstractmethod
    def validate_params(self, params: dict, role: str) -> None:
        """Check parameters without data; role is ``holdout`` or ``validation``."""

    @abstractmethod
    def n_folds(self, params: dict) -> int:
        """Number of validation folds the parameters produce."""

    @abstractmethod
    def holdout_split(
        self, frame: pd.DataFrame, share: float, params: dict, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (development, holdout) row positions; ``share`` is the holdout fraction."""

    @abstractmethod
    def folds(
        self, frame: pd.DataFrame, params: dict, seed: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (train, validation) row positions for every fold."""


def _wrap(call):
    try:
        return call()
    except ValueError as exc:
        raise ValidationError("split", str(exc)) from exc


class RandomScheme(SplitScheme):
    """Random split, stratified on a column when ``stratify_by`` is given."""

    name = "random"
    deterministic = False

    def validate_params(self, params, role):
        required = ("n_folds",) if role == "validation" else ()
        check_params(params, role, required, ("stratify_by",))
        if role == "validation":
            check_fold_count(params["n_folds"])
        if "stratify_by" in params:
            require_text("stratify_by", params["stratify_by"])

    def n_folds(self, params):
        return params["n_folds"]

    def _strata(self, frame, params):
        if "stratify_by" not in params:
            return None
        return _column(frame, params["stratify_by"], "stratify_by").to_numpy()

    def holdout_split(self, frame, share, params, seed):
        positions = np.arange(len(frame))
        strata = self._strata(frame, params)
        dev, hold = _wrap(
            lambda: train_test_split(
                positions, test_size=share, random_state=seed, shuffle=True, stratify=strata
            )
        )
        return np.sort(dev), np.sort(hold)

    def folds(self, frame, params, seed):
        positions = np.zeros(len(frame))
        strata = self._strata(frame, params)
        n = params["n_folds"]
        if strata is None:
            splitter = KFold(n, shuffle=True, random_state=seed)
            return _wrap(lambda: list(splitter.split(positions)))
        splitter = StratifiedKFold(n, shuffle=True, random_state=seed)
        return _wrap(lambda: list(splitter.split(positions, strata)))


class GroupScheme(SplitScheme):
    """No group is on both sides of a split."""

    name = "group"
    deterministic = False

    def validate_params(self, params, role):
        required = ("group_column", "n_folds") if role == "validation" else ("group_column",)
        check_params(params, role, required)
        require_text("group_column", params["group_column"])
        if role == "validation":
            check_fold_count(params["n_folds"])

    def n_folds(self, params):
        return params["n_folds"]

    def holdout_split(self, frame, share, params, seed):
        groups = _column(frame, params["group_column"], "group_column").to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=share, random_state=seed)
        dev, hold = _wrap(lambda: next(splitter.split(np.zeros(len(frame)), groups=groups)))
        return np.sort(dev), np.sort(hold)

    def folds(self, frame, params, seed):
        groups = _column(frame, params["group_column"], "group_column").to_numpy()
        splitter = GroupKFold(params["n_folds"], shuffle=True, random_state=seed)
        return _wrap(lambda: list(splitter.split(np.zeros(len(frame)), groups=groups)))


class ChronologicalScheme(SplitScheme):
    """Later rows validate earlier ones: the holdout is the last share, folds expand a window.

    Rows with equal time stay on the same side of every boundary, so a
    validation block is strictly later than the training rows. Deterministic:
    the seed plays no role.
    """

    name = "chronological"
    deterministic = True

    def validate_params(self, params, role):
        required = ("time_column", "n_folds") if role == "validation" else ("time_column",)
        check_params(params, role, required)
        require_text("time_column", params["time_column"])
        if role == "validation":
            check_fold_count(params["n_folds"])

    def n_folds(self, params):
        return params["n_folds"]

    @staticmethod
    def _order(frame, params):
        column = _column(frame, params["time_column"], "time_column")
        usable = pd.api.types.is_datetime64_any_dtype(column) or (
            pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)
        )
        if not usable:
            raise ValidationError("time_column", "must be numeric or datetime")
        values = column.to_numpy()
        order = np.argsort(values, kind="stable")
        return order, values[order]

    @staticmethod
    def _boundary(values, cut):
        """Move ``cut`` forward so that equal times are not split."""
        while 0 < cut < len(values) and values[cut] == values[cut - 1]:
            cut += 1
        return cut

    def holdout_split(self, frame, share, params, seed):
        order, values = self._order(frame, params)
        n = len(values)
        cut = self._boundary(values, n - max(1, round(share * n)))
        if not 0 < cut < n:
            raise ValidationError("time_column", "has no time boundary for the holdout share")
        return order[:cut], order[cut:]

    def folds(self, frame, params, seed):
        order, values = self._order(frame, params)
        n, blocks = len(values), params["n_folds"] + 1
        bounds = [self._boundary(values, round(i * n / blocks)) for i in range(1, blocks)]
        bounds.append(n)
        if bounds[0] <= 0 or any(b <= a for a, b in zip(bounds, bounds[1:])):
            raise ValidationError("time_column", "has too few distinct times for the folds")
        return [(order[: bounds[i]], order[bounds[i] : bounds[i + 1]]) for i in range(blocks - 1)]


SCHEMES: dict[str, SplitScheme] = {}


def register_scheme(scheme: SplitScheme) -> None:
    """Add a scheme to the registry (fixed code only, never from the model)."""
    if not isinstance(scheme, SplitScheme):
        raise ValidationError("scheme", "must be a SplitScheme")
    require_text("scheme", scheme.name)
    if not isinstance(scheme.deterministic, bool):
        raise ValidationError("scheme", "must declare deterministic as a boolean")
    if scheme.name in SCHEMES:
        raise ValidationError("scheme", f"{scheme.name!r} is already registered")
    SCHEMES[scheme.name] = scheme


def get_scheme(name: object, field: str = "scheme") -> SplitScheme:
    if not isinstance(name, str):
        raise ValidationError(field, "must be the name of a registered scheme")
    if name not in SCHEMES:
        raise ValidationError(
            field, f"unknown scheme {name!r} (registered: {', '.join(sorted(SCHEMES))})"
        )
    return SCHEMES[name]


for _scheme in (RandomScheme(), GroupScheme(), ChronologicalScheme()):
    register_scheme(_scheme)
