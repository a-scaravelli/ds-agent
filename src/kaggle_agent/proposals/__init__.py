"""Proposal data model: short plan, attempt and suggestion, with validation and JSON."""

from .attempt import Attempt
from .errors import ValidationError
from .plan import MAX_PLAN_WORDS, ShortPlan
from .suggestion import Suggestion

__all__ = ["Attempt", "MAX_PLAN_WORDS", "ShortPlan", "Suggestion", "ValidationError"]
