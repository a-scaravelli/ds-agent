"""The five roles of the agents (design section 8)."""

from .errors import ValidationError

ROLES = ("coordinator", "data", "feature_engineering", "modeling", "reviewer")


def validate_role(value: object, field: str = "role") -> str:
    """Return ``value`` if it is one of ``ROLES``, else raise ``ValidationError``."""
    if value is None:
        raise ValidationError(field, "is required")
    if not isinstance(value, str) or value not in ROLES:
        raise ValidationError(field, f"must be one of {', '.join(ROLES)}")
    return value
