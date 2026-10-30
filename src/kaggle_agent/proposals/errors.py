"""Validation error shared by the proposal data model."""

import json


# object-level labels the validators use as ``field`` (not a field of any type)
OBJECT_LABELS = (
    "plan", "attempt", "suggestion", "goal", "criterion", "guidance", "round", "report",
)


class ValidationError(ValueError):
    """Raised when plan, attempt or suggestion data is invalid.

    Carries the offending ``field`` and the ``reason`` so the loop can pass
    the reason for the rejection back to the model (design section 2).
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


def require_text(field: str, value: object) -> str:
    """Return ``value`` if it is a string that is not empty after ``strip``."""
    if value is None:
        raise ValidationError(field, "is required")
    if not isinstance(value, str):
        raise ValidationError(field, "must be a string")
    if not value.strip():
        raise ValidationError(field, "must not be empty")
    return value


def require_text_list(field: str, value: object, *, allow_empty: bool) -> tuple[str, ...]:
    """Return ``value`` as a tuple of non-empty strings."""
    if value is None:
        raise ValidationError(field, "is required")
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValidationError(field, "must be a list of strings")
    if not value and not allow_empty:
        raise ValidationError(field, "must not be empty")
    for index, item in enumerate(value):
        require_text(f"{field}[{index}]", item)
    return tuple(value)


def check_keys(cls_name: str, data: object, allowed: tuple[str, ...]) -> None:
    """Reject non-dict input and unknown keys."""
    if not isinstance(data, dict):
        raise ValidationError(cls_name, "must be a JSON object")
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValidationError(unknown[0], f"unknown key (allowed: {', '.join(allowed)})")


def loads_object(cls_name: str, text: object) -> dict:
    """Parse a JSON object, turning any failure into ``ValidationError``."""
    if not isinstance(text, str):
        raise ValidationError(cls_name, "JSON input must be a string")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValidationError(cls_name, f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(cls_name, "JSON input must be an object")
    return data
