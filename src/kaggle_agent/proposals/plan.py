"""Short initial plan with five mandatory items (design section 4)."""

import json
from dataclasses import dataclass

from .errors import ValidationError, check_keys, loads_object, require_text, require_text_list

MAX_PLAN_WORDS = 300

_FIELDS = ("metric", "validation", "feature_engineering", "models", "baseline")


@dataclass(frozen=True)
class ShortPlan:
    """The plan the user approves before the autonomous cycle."""

    metric: str
    validation: str
    feature_engineering: tuple[str, ...]
    models: tuple[str, ...]
    baseline: str

    def __post_init__(self) -> None:
        require_text("metric", self.metric)
        require_text("validation", self.validation)
        require_text("baseline", self.baseline)
        object.__setattr__(
            self,
            "feature_engineering",
            require_text_list("feature_engineering", self.feature_engineering, allow_empty=False),
        )
        object.__setattr__(
            self, "models", require_text_list("models", self.models, allow_empty=False)
        )
        words = self.word_count()
        if words > MAX_PLAN_WORDS:
            raise ValidationError(
                "plan", f"has {words} words, over the limit of {MAX_PLAN_WORDS}"
            )

    @classmethod
    def contract_hints(cls) -> dict[str, str]:
        """Validation constraints per field, for the JSON contract asked of the model.

        ``__all__`` holds the rules on the whole object.
        """
        return {
            "metric": "non-empty string",
            "validation": "non-empty string",
            "baseline": "non-empty string",
            "feature_engineering": "non-empty list of non-empty strings",
            "models": "non-empty list of non-empty strings",
            "__all__": (
                f"the five items together must stay within {MAX_PLAN_WORDS} "
                "whitespace-separated words"
            ),
        }

    def word_count(self) -> int:
        """Number of whitespace-separated words across the five items."""
        parts = [self.metric, self.validation, self.baseline]
        parts += self.feature_engineering + self.models
        return sum(len(p.split()) for p in parts)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "validation": self.validation,
            "feature_engineering": list(self.feature_engineering),
            "models": list(self.models),
            "baseline": self.baseline,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShortPlan":
        check_keys("plan", data, _FIELDS)
        return cls(**{name: data.get(name) for name in _FIELDS})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ShortPlan":
        return cls.from_dict(loads_object("plan", text))
