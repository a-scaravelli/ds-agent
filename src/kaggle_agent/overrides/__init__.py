"""What the DS may override, and how it is recorded and labelled (cockpit ticket 4/4).

A neutral, stdlib-only package imported by BOTH the web and the orchestrator, so the two sides share one
definition of the decision kinds, the labels, the overlay of the improvement rule and the holdout seal.
Nothing here reads or writes a file: callers pass the records they read.
"""

from .labels import LABELS, run_labels
from .learnings import search_items
from .overlay import apply_bar_overrides, resolve_choice, valid_bar_overrides
from .records import (
    KINDS,
    REASON_REQUIRED,
    normalize_payload,
    records_for_cycle,
)
from .sealing import classify_read, seal_state

__all__ = [
    "KINDS", "LABELS", "REASON_REQUIRED", "apply_bar_overrides", "classify_read", "normalize_payload",
    "records_for_cycle", "resolve_choice", "run_labels", "search_items", "seal_state", "valid_bar_overrides",
]
