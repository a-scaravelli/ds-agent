"""Round and reviewer report (design sections 2, 7 and 8)."""

import json
from dataclasses import dataclass

from .errors import ValidationError, check_keys, loads_object, require_text, require_text_list

_ROUND_FIELDS = ("id", "tasks", "candidates")
_REPORT_FIELDS = ("id", "round_id", "text", "created_at")


@dataclass(frozen=True)
class Round:
    """A round: the tasks assigned by the coordinator and its candidates.

    ``id`` is the round id. ``candidates`` are candidate ids, equal to
    attempt ids, without duplicates.
    """

    id: str
    tasks: tuple[str, ...]
    candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text("id", self.id)
        object.__setattr__(
            self, "tasks", require_text_list("tasks", self.tasks, allow_empty=True)
        )
        candidates = require_text_list("candidates", self.candidates, allow_empty=True)
        if len(set(candidates)) != len(candidates):
            raise ValidationError("candidates", "must not contain duplicates")
        object.__setattr__(self, "candidates", candidates)

    def to_dict(self) -> dict:
        return {"id": self.id, "tasks": list(self.tasks), "candidates": list(self.candidates)}

    @classmethod
    def from_dict(cls, data: dict) -> "Round":
        check_keys("round", data, _ROUND_FIELDS)
        kwargs = {name: data[name] for name in _ROUND_FIELDS if name in data}
        kwargs.setdefault("id", None)
        kwargs.setdefault("tasks", None)
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Round":
        return cls.from_dict(loads_object("round", text))


@dataclass(frozen=True)
class ReviewerReport:
    """The reviewer's interpretation of a round: text only, never a decision."""

    id: str
    round_id: str
    text: str
    created_at: str

    def __post_init__(self) -> None:
        for name in _REPORT_FIELDS:
            require_text(name, getattr(self, name))

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _REPORT_FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewerReport":
        check_keys("report", data, _REPORT_FIELDS)
        return cls(**{name: data.get(name) for name in _REPORT_FIELDS})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ReviewerReport":
        return cls.from_dict(loads_object("report", text))
