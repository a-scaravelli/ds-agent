"""Attempt: a falsifiable hypothesis of the agent (design section 4)."""

import json
from dataclasses import dataclass

from .errors import ValidationError, check_keys, loads_object, require_text, require_text_list

_FIELDS = (
    "id",
    "phase",
    "parent_id",
    "hypothesis",
    "rationale",
    "evidence",
    "exploratory",
    "what_changed",
    "expected_effect",
    "measure",
    "estimated_cost",
    "risks",
    "artifacts",
)
_REQUIRED_TEXT = ("id", "phase", "hypothesis", "rationale", "what_changed", "measure")
_OPTIONAL_TEXT = ("parent_id", "expected_effect", "estimated_cost")
_LISTS = ("evidence", "risks", "artifacts")


@dataclass(frozen=True)
class Attempt:
    """One iteration of the cycle.

    Without ``evidence`` the attempt must be marked ``exploratory``.
    """

    id: str
    phase: str
    hypothesis: str
    rationale: str
    what_changed: str
    measure: str
    parent_id: str | None = None
    evidence: tuple[str, ...] = ()
    exploratory: bool = False
    expected_effect: str | None = None
    estimated_cost: str | None = None
    risks: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in _REQUIRED_TEXT:
            require_text(name, getattr(self, name))
        for name in _OPTIONAL_TEXT:
            if getattr(self, name) is not None:
                require_text(name, getattr(self, name))
        for name in _LISTS:
            value = getattr(self, name)
            object.__setattr__(
                self, name, require_text_list(name, value, allow_empty=True)
            )
        if not isinstance(self.exploratory, bool):
            raise ValidationError("exploratory", "must be a boolean")
        if self.parent_id == self.id:
            raise ValidationError("parent_id", "must differ from id")
        if not self.evidence and not self.exploratory:
            raise ValidationError(
                "evidence", "is required unless the attempt is marked exploratory"
            )

    @classmethod
    def contract_hints(cls) -> dict[str, str]:
        """Validation constraints per field, for the JSON contract asked of the model.

        ``__all__`` holds the rules on the whole object.
        """
        hints = {name: "non-empty string" for name in _REQUIRED_TEXT}
        hints["parent_id"] = "if present, a non-empty string different from id"
        for name in ("expected_effect", "estimated_cost"):
            hints[name] = "if present, a non-empty string"
        for name in _LISTS:
            hints[name] = "list of non-empty strings (may be empty)"
        hints["exploratory"] = "true or false"
        hints["__all__"] = "if evidence is empty, exploratory must be true"
        return hints

    def to_dict(self) -> dict:
        out = {}
        for name in _FIELDS:
            value = getattr(self, name)
            out[name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Attempt":
        check_keys("attempt", data, _FIELDS)
        kwargs = {name: data[name] for name in _FIELDS if name in data}
        for name in _REQUIRED_TEXT:
            kwargs.setdefault(name, None)
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Attempt":
        return cls.from_dict(loads_object("attempt", text))
