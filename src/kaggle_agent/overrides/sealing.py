"""The state of the holdout panel: sealed until the DS unseals it (the engine allows ONE holdout session)."""

AT_REVIEW = ("AWAITING_REVIEW", "STOPPED")
SEALED, READY, UNSEALED = "sealed", "ready", "unsealed"
FIRST, EARLY, SECOND = "first", "early", "second"


def classify_read(cycle_state: str, reads: list[dict]) -> str:
    """first (at review, no read yet), second (a read exists) or early (the run is still iterating)."""
    if reads:
        return SECOND
    return FIRST if cycle_state in AT_REVIEW else EARLY


def seal_state(reads: list[dict], outcome_recorded: bool) -> str:
    """sealed: no unseal record; ready: sealed but the outcome exists (review); unsealed: a read exists."""
    if reads:
        return UNSEALED
    return READY if outcome_recorded else SEALED
