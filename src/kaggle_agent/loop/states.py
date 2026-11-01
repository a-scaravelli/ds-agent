"""The states of the cycle (design section 2) and the transitions the fixed code allows."""

from .errors import TransitionError

PLANNING = "PLANNING"
AWAITING_PLAN_APPROVAL = "AWAITING_PLAN_APPROVAL"
SPLITTING = "SPLITTING"
BASELINE = "BASELINE"
ITERATING = "ITERATING"
AWAITING_GUIDANCE = "AWAITING_GUIDANCE"
REPORTING = "REPORTING"
AWAITING_REVIEW = "AWAITING_REVIEW"
STOPPED = "STOPPED"

STATES = (
    PLANNING,
    AWAITING_PLAN_APPROVAL,
    SPLITTING,
    BASELINE,
    ITERATING,
    AWAITING_GUIDANCE,
    REPORTING,
    AWAITING_REVIEW,
    STOPPED,
)

# ITERATING -> ITERATING is the end of a round with no pause for guidance
TRANSITIONS: dict[str, tuple[str, ...]] = {
    PLANNING: (AWAITING_PLAN_APPROVAL,),
    AWAITING_PLAN_APPROVAL: (PLANNING, SPLITTING),
    SPLITTING: (BASELINE,),
    BASELINE: (ITERATING,),
    ITERATING: (ITERATING, AWAITING_GUIDANCE, REPORTING),
    AWAITING_GUIDANCE: (ITERATING, REPORTING),
    REPORTING: (AWAITING_REVIEW,),
    AWAITING_REVIEW: (STOPPED,),
    STOPPED: (),
}


def check_transition(current: str, new: str) -> None:
    """Raise ``TransitionError`` unless ``current`` -> ``new`` is allowed."""
    if current not in TRANSITIONS:
        raise TransitionError(f"unknown state {current!r}")
    if new not in TRANSITIONS[current]:
        raise TransitionError(f"transition {current} -> {new} is not allowed")
