"""Typed guidance of the DS for the roles: steering, a 'Try this' request and a carry-over (ticket 30).

Frozen dataclasses with ``to_dict`` / ``from_dict`` and the same contract validation as the other
proposal types. They carry descriptions and numbers only, never code. Stdlib only, shared by the web
(validation of a form), the orchestrator (what is applied) and the agents (what a role is told).
"""

from dataclasses import dataclass

from .errors import ValidationError, check_keys, require_text
from .libraries import ALLOWED_LIBRARIES, LIBRARY_APIS

FOCUSES = ("feature_engineering", "model_tuning", "ensembling_free", "data_cleaning")
SKLEARN_FAMILIES = ("linear", "tree_ensemble", "nearest_neighbors", "svm", "naive_bayes")
# a closed vocabulary: the scikit-learn families plus the boosting libraries that may be used at all
MODEL_FAMILIES = SKLEARN_FAMILIES + tuple(name for name in LIBRARY_APIS if name in ALLOWED_LIBRARIES)

MAX_ITEMS = 20
MAX_NAME = 80
MAX_TEXT = 2000
MAX_CARRY_ITEMS = 5
MAX_CARRY_TEXT = 600


def _names(field: str, value: object, vocabulary: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValidationError(field, "must be a list of names")
    if len(value) > MAX_ITEMS:
        raise ValidationError(field, f"must have at most {MAX_ITEMS} items")
    out: list[str] = []
    for index, item in enumerate(value):
        text = require_text(f"{field}[{index}]", item).strip()
        if len(text) > MAX_NAME:
            raise ValidationError(f"{field}[{index}]", f"must be at most {MAX_NAME} characters")
        if vocabulary is not None and text not in vocabulary:
            raise ValidationError(f"{field}[{index}]", f"unknown model family {text!r} (known: {', '.join(vocabulary)})")
        if text not in out:
            out.append(text)
    return tuple(out)


def _text(field: str, value: object, limit: int, required: bool = False) -> str:
    if value is None or value == "":
        if required:
            raise ValidationError(field, "is required")
        return ""
    text = require_text(field, value).strip()
    if len(text) > limit:
        raise ValidationError(field, f"must be at most {limit} characters")
    return text


@dataclass(frozen=True)
class Steering:
    """What the DS wants the roles to favour or avoid; empty means no steering."""

    focus: str | None = None
    pin_features: tuple[str, ...] = ()
    avoid_features: tuple[str, ...] = ()
    try_families: tuple[str, ...] = ()
    avoid_families: tuple[str, ...] = ()
    allowed_model_families: tuple[str, ...] = ()
    notes: str = ""

    _FIELDS = ("focus", "pin_features", "avoid_features", "try_families", "avoid_families",
               "allowed_model_families", "notes")

    def __post_init__(self) -> None:
        if self.focus is not None and self.focus not in FOCUSES:
            raise ValidationError("focus", f"unknown focus {self.focus!r} (known: {', '.join(FOCUSES)})")
        object.__setattr__(self, "pin_features", _names("pin_features", self.pin_features))
        object.__setattr__(self, "avoid_features", _names("avoid_features", self.avoid_features))
        for name in ("try_families", "avoid_families", "allowed_model_families"):
            object.__setattr__(self, name, _names(name, getattr(self, name), MODEL_FAMILIES))
        object.__setattr__(self, "notes", _text("notes", self.notes, MAX_TEXT))
        both = sorted(set(self.pin_features) & set(self.avoid_features))
        if both:
            raise ValidationError("pin_features", f"{both[0]!r} is both pinned and avoided")
        both = sorted(set(self.try_families) & set(self.avoid_families))
        if both:
            raise ValidationError("try_families", f"{both[0]!r} is both to try and to avoid")

    @property
    def empty(self) -> bool:
        return self.focus is None and not self.notes and not any(
            getattr(self, n) for n in self._FIELDS[1:6])

    def to_dict(self) -> dict:
        return {"focus": self.focus, **{n: list(getattr(self, n)) for n in self._FIELDS[1:6]},
                "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict) -> "Steering":
        check_keys("steering", data, cls._FIELDS)
        return cls(**{n: data[n] for n in cls._FIELDS if n in data})


@dataclass(frozen=True)
class TryThis:
    """A hypothesis written by the DS, with optional constraints, to be built and evaluated as a candidate."""

    hypothesis: str
    constraints: tuple[str, ...] = ()

    _FIELDS = ("hypothesis", "constraints")

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis", _text("hypothesis", self.hypothesis, MAX_TEXT, required=True))
        raw = self.constraints
        if raw is None:
            raw = ()
        if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
            raise ValidationError("constraints", "must be a list of texts")
        if len(raw) > 10:
            raise ValidationError("constraints", "must have at most 10 items")
        object.__setattr__(self, "constraints", tuple(
            _text(f"constraints[{i}]", item, 500, required=True) for i, item in enumerate(raw)))

    def to_dict(self) -> dict:
        return {"hypothesis": self.hypothesis, "constraints": list(self.constraints)}

    @classmethod
    def from_dict(cls, data: dict) -> "TryThis":
        check_keys("request", data, cls._FIELDS)
        return cls(hypothesis=data.get("hypothesis"), constraints=data.get("constraints", ()))


def _digest_list(field: str, value: object) -> tuple[dict, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValidationError(field, "must be a list")
    items = []
    for index, item in enumerate(list(value)[:MAX_CARRY_ITEMS]):
        if not isinstance(item, dict):
            raise ValidationError(f"{field}[{index}]", "must be an object")
        clean = {}
        for key, val in item.items():
            if not isinstance(key, str):
                raise ValidationError(f"{field}[{index}]", "keys must be text")
            if isinstance(val, str):
                val = val[:MAX_CARRY_TEXT]
            elif val is not None and not isinstance(val, (int, float, bool)):
                raise ValidationError(f"{field}[{index}].{key}", "must be text, a number or null")
            clean[key] = val
        items.append(clean)
    return tuple(items)


def _text_list(field: str, value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValidationError(field, "must be a list of texts")
    return tuple(_text(f"{field}[{i}]", item, MAX_CARRY_TEXT, required=True)
                 for i, item in enumerate(list(value)[-MAX_CARRY_ITEMS:]))


@dataclass(frozen=True)
class CarryOver:
    """What a forked cycle inherits from its origin: descriptions and numbers, never code.

    ``same_protocol`` is False when the fork froze a new protocol: the carried scores were then
    measured under another protocol and are not comparable with the new ones.
    """

    origin_cycle: str
    origin_attempt: str | None = None
    same_protocol: bool = True
    protocol_note: str = ""
    attempt: tuple[dict, ...] = ()
    top_attempts: tuple[dict, ...] = ()
    failures: tuple[dict, ...] = ()
    guidance: tuple[str, ...] = ()
    reviewer: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    _FIELDS = ("origin_cycle", "origin_attempt", "same_protocol", "protocol_note", "attempt",
               "top_attempts", "failures", "guidance", "reviewer", "notes")

    def __post_init__(self) -> None:
        require_text("origin_cycle", self.origin_cycle)
        if self.origin_attempt is not None:
            require_text("origin_attempt", self.origin_attempt)
        if not isinstance(self.same_protocol, bool):
            raise ValidationError("same_protocol", "must be true or false")
        object.__setattr__(self, "protocol_note", _text("protocol_note", self.protocol_note, MAX_CARRY_TEXT))
        for name in ("attempt", "top_attempts", "failures"):
            object.__setattr__(self, name, _digest_list(name, getattr(self, name)))
        for name in ("guidance", "reviewer", "notes"):
            object.__setattr__(self, name, _text_list(name, getattr(self, name)))

    def to_dict(self) -> dict:
        data = {}
        for name in self._FIELDS:
            value = getattr(self, name)
            data[name] = [dict(i) for i in value] if name in ("attempt", "top_attempts", "failures") else (
                list(value) if isinstance(value, tuple) else value)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CarryOver":
        check_keys("carry-over", data, cls._FIELDS)
        return cls(**{n: data[n] for n in cls._FIELDS if n in data})
