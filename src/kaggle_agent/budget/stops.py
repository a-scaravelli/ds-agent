"""Stop reasons and the pure rules behind them (design section 11)."""

TARGET_REACHED = "target_reached"
CONSECUTIVE_FAILURES = "consecutive_failures"
MAX_ROUNDS = "max_rounds"
STAGNATION = "stagnation"

# fixed priority when several reasons apply at once
STOP_REASONS = (TARGET_REACHED, CONSECUTIVE_FAILURES, MAX_ROUNDS, STAGNATION)


def pause_due(round_position: int, m: int) -> bool:
    """True when the cycle pauses for guidance after the round at ``round_position`` (from 1)."""
    if m < 1 or round_position < 1:
        raise ValueError("round_position and m must be at least 1")
    return round_position % m == 0


def rounds_without_improvement(improved: list[bool]) -> int:
    """Rounds without real improvement at the end of ``improved`` (one flag per finished round)."""
    count = 0
    for flag in reversed(improved):
        if flag:
            break
        count += 1
    return count


def stop_reason(
    *,
    target_met: bool,
    rounds_done: int,
    max_rounds: int,
    stagnant_rounds: int,
    stagnation_k: int,
    consecutive_failures: int,
    max_consecutive_failures: int,
) -> str | None:
    """The reason to stop, or None. Priority: target, failures, rounds, stagnation."""
    if target_met:
        return TARGET_REACHED
    if consecutive_failures >= max_consecutive_failures:
        return CONSECUTIVE_FAILURES
    if rounds_done >= max_rounds:
        return MAX_ROUNDS
    if stagnant_rounds >= stagnation_k:
        return STAGNATION
    return None
