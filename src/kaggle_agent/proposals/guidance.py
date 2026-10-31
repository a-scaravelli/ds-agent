"""Guidance of the lead given after a round (design sections 2 and 7)."""

import json
from dataclasses import dataclass

from .errors import check_keys, loads_object, require_text

_FIELDS = ("id", "round_id", "text", "created_at")


@dataclass(frozen=True)
class Guidance:
    """Text of the lead, the round it follows and when it was given (ISO text)."""

    id: str
    round_id: str
    text: str
    created_at: str

    def __post_init__(self) -> None:
        for name in _FIELDS:
            require_text(name, getattr(self, name))

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "Guidance":
        check_keys("guidance", data, _FIELDS)
        return cls(**{name: data.get(name) for name in _FIELDS})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Guidance":
        return cls.from_dict(loads_object("guidance", text))
