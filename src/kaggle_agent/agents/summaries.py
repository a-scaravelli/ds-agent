"""Typed summaries shared by the roles, and the base of every typed input and output.

A summary holds numbers and short text, never rows of data: there is no field
for labels, holdout data, credentials, file paths or registry references.
Every type has ``to_dict``, ``from_dict``, ``to_json`` and ``from_json`` and
raises ``ValidationError`` on invalid data.
"""

import json
import math
from dataclasses import dataclass
from typing import ClassVar

from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import (
    check_keys,
    loads_object,
    require_text,
    require_text_list,
)


class JsonObject:
    """Mixin: JSON text on top of ``to_dict`` and ``from_dict``."""

    LABEL: ClassVar[str] = "object"

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(loads_object(cls.LABEL, text))


def require_count(field: str, value: object) -> int:
    """A non-negative integer (a boolean is not accepted)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(field, "must be an integer")
    if value < 0:
        raise ValidationError(field, "must not be negative")
    return value


def require_number(field: str, value: object) -> float:
    """A finite number (a boolean is not accepted)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(field, "must be a number")
    if not math.isfinite(value):
        raise ValidationError(field, "must be finite")
    return float(value)


def require_items(field: str, value: object, kind: type) -> tuple:
    """A list or tuple whose items are all instances of ``kind``, as a tuple."""
    if value is None:
        raise ValidationError(field, "is required")
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValidationError(field, "must be a list")
    items = tuple(value)
    for index, item in enumerate(items):
        if not isinstance(item, kind):
            raise ValidationError(f"{field}[{index}]", f"must be a {kind.__name__}")
    return items


def items_from(field: str, value: object, kind: type) -> tuple:
    """Build ``kind`` objects from a list of dicts with ``kind.from_dict``."""
    if value is None:
        raise ValidationError(field, "is required")
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValidationError(field, "must be a list")
    return tuple(item if isinstance(item, kind) else kind.from_dict(item) for item in value)


def object_from(field: str, value: object, kind: type):
    """Build one ``kind`` from a dict (or return it when it already is one)."""
    if isinstance(value, kind):
        return value
    if value is None:
        raise ValidationError(field, "is required")
    return kind.from_dict(value)


@dataclass(frozen=True)
class ColumnSummary(JsonObject):
    """Statistics of one column: name, type, share of missing values, distinct values."""

    LABEL: ClassVar[str] = "column"
    name: str
    dtype: str
    missing_fraction: float
    distinct_values: int

    def __post_init__(self) -> None:
        require_text("name", self.name)
        require_text("dtype", self.dtype)
        fraction = require_number("missing_fraction", self.missing_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValidationError("missing_fraction", "must be between 0 and 1")
        object.__setattr__(self, "missing_fraction", fraction)
        require_count("distinct_values", self.distinct_values)

    _FIELDS: ClassVar[tuple[str, ...]] = ("name", "dtype", "missing_fraction", "distinct_values")

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnSummary":
        check_keys("column", data, cls._FIELDS)
        return cls(**{name: data.get(name) for name in cls._FIELDS})


@dataclass(frozen=True)
class DataSummary(JsonObject):
    """What the roles may know about the data: sizes, target name, per-column statistics."""

    LABEL: ClassVar[str] = "data summary"
    rows: int
    target: str
    columns: tuple[ColumnSummary, ...]
    notes: tuple[str, ...] = ()

    _FIELDS: ClassVar[tuple[str, ...]] = ("rows", "target", "columns", "notes")

    def __post_init__(self) -> None:
        require_count("rows", self.rows)
        require_text("target", self.target)
        object.__setattr__(self, "columns", require_items("columns", self.columns, ColumnSummary))
        object.__setattr__(self, "notes", require_text_list("notes", self.notes, allow_empty=True))

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "target": self.target,
            "columns": [c.to_dict() for c in self.columns],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataSummary":
        check_keys("data summary", data, cls._FIELDS)
        kwargs = {
            "rows": data.get("rows"),
            "target": data.get("target"),
            "columns": items_from("columns", data.get("columns"), ColumnSummary),
        }
        if "notes" in data:
            kwargs["notes"] = data["notes"]
        return cls(**kwargs)


@dataclass(frozen=True)
class AttemptDigest(JsonObject):
    """One past attempt in a line: id, hypothesis, outcome and validation score (if any)."""

    LABEL: ClassVar[str] = "attempt digest"
    id: str
    hypothesis: str
    outcome: str
    validation_score: float | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = ("id", "hypothesis", "outcome", "validation_score")

    def __post_init__(self) -> None:
        for name in ("id", "hypothesis", "outcome"):
            require_text(name, getattr(self, name))
        if self.validation_score is not None:
            object.__setattr__(
                self, "validation_score", require_number("validation_score", self.validation_score)
            )

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "AttemptDigest":
        check_keys("attempt digest", data, cls._FIELDS)
        kwargs = {name: data[name] for name in cls._FIELDS if name in data}
        for name in ("id", "hypothesis", "outcome"):
            kwargs.setdefault(name, None)
        return cls(**kwargs)


@dataclass(frozen=True)
class RegistrySummary(JsonObject):
    """What the registry knows, boiled down for the roles: best attempts and recent failures.

    Built by the caller from the registry; the roles never see the registry.
    """

    LABEL: ClassVar[str] = "history"
    best: tuple[AttemptDigest, ...] = ()
    recent_failures: tuple[AttemptDigest, ...] = ()

    _FIELDS: ClassVar[tuple[str, ...]] = ("best", "recent_failures")

    def __post_init__(self) -> None:
        for name in self._FIELDS:
            object.__setattr__(self, name, require_items(name, getattr(self, name), AttemptDigest))

    def to_dict(self) -> dict:
        return {name: [d.to_dict() for d in getattr(self, name)] for name in self._FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "RegistrySummary":
        check_keys("history", data, cls._FIELDS)
        return cls(
            **{
                name: items_from(name, data[name], AttemptDigest)
                for name in cls._FIELDS
                if name in data
            }
        )


@dataclass(frozen=True)
class DataFindings(JsonObject):
    """Output of the data role: profile, quality issues, leakage hints, proposed cleaning."""

    LABEL: ClassVar[str] = "data findings"
    profile: str
    quality_issues: tuple[str, ...] = ()
    leakage_hints: tuple[str, ...] = ()
    proposed_cleaning: tuple[str, ...] = ()

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "profile", "quality_issues", "leakage_hints", "proposed_cleaning",
    )
    _LISTS: ClassVar[tuple[str, ...]] = _FIELDS[1:]

    def __post_init__(self) -> None:
        require_text("profile", self.profile)
        for name in self._LISTS:
            object.__setattr__(
                self, name, require_text_list(name, getattr(self, name), allow_empty=True)
            )

    @classmethod
    def contract_hints(cls) -> dict[str, str]:
        """Validation constraints per field, for the JSON contract asked of the model."""
        hints = {"profile": "non-empty string"}
        for name in cls._LISTS:
            hints[name] = "list of non-empty strings (may be empty)"
        return hints

    def to_dict(self) -> dict:
        return {
            name: list(v) if isinstance(v := getattr(self, name), tuple) else v
            for name in self._FIELDS
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataFindings":
        check_keys("data findings", data, cls._FIELDS)
        kwargs = {name: data[name] for name in cls._FIELDS if name in data}
        kwargs.setdefault("profile", None)
        return cls(**kwargs)


@dataclass(frozen=True)
class MetricValue(JsonObject):
    """A named metric value, as the evaluator computed it."""

    LABEL: ClassVar[str] = "metric value"
    name: str
    value: float

    _FIELDS: ClassVar[tuple[str, ...]] = ("name", "value")

    def __post_init__(self) -> None:
        require_text("name", self.name)
        object.__setattr__(self, "value", require_number("value", self.value))

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> "MetricValue":
        check_keys("metric value", data, cls._FIELDS)
        return cls(**{name: data.get(name) for name in cls._FIELDS})
