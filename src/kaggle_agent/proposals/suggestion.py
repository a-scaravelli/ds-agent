"""User suggestion recorded at review time (design section 7)."""

import json
from dataclasses import dataclass

from .errors import ValidationError, check_keys, loads_object, require_text, require_text_list

_FIELDS = ("id", "text", "origin_cycle", "spawned_cycles")


@dataclass(frozen=True)
class Suggestion:
    """A suggestion with the cycle it came from and the cycles it spawned."""

    id: str
    text: str
    origin_cycle: str
    spawned_cycles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text("id", self.id)
        require_text("text", self.text)
        require_text("origin_cycle", self.origin_cycle)
        spawned = require_text_list("spawned_cycles", self.spawned_cycles, allow_empty=True)
        if len(set(spawned)) != len(spawned):
            raise ValidationError("spawned_cycles", "must not contain duplicates")
        if self.origin_cycle in spawned:
            raise ValidationError("spawned_cycles", "must not contain origin_cycle")
        object.__setattr__(self, "spawned_cycles", spawned)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "origin_cycle": self.origin_cycle,
            "spawned_cycles": list(self.spawned_cycles),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Suggestion":
        check_keys("suggestion", data, _FIELDS)
        kwargs = {name: data[name] for name in _FIELDS if name in data}
        for name in ("id", "text", "origin_cycle"):
            kwargs.setdefault(name, None)
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Suggestion":
        return cls.from_dict(loads_object("suggestion", text))
