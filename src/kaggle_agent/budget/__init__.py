"""Limits and stop criteria of the autonomous cycle (design sections 11 and 14, level 1).

Level 1 has caps on rounds and candidates only: the number of rounds N, the
candidates per round C, the rounds without real improvement K, the rounds in a
row without a pause for guidance M, the fix retries per attempt, the cap on
consecutive failed attempts and the cap on consecutive rejected proposals per
candidate. The defaults are provisional and configurable. Time, submission and
subscription limits arrive at level 2. This package is pure: standard library
and ``proposals`` only.
"""

from .limits import (
    DEFAULT_CANDIDATES,
    DEFAULT_FIX_RETRIES,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    DEFAULT_MAX_REJECTIONS,
    DEFAULT_PAUSE_EVERY,
    DEFAULT_ROUNDS,
    DEFAULT_STAGNATION_ROUNDS,
    CycleLimits,
)
from .stops import (
    CONSECUTIVE_FAILURES,
    MAX_ROUNDS,
    STAGNATION,
    STOP_REASONS,
    TARGET_REACHED,
    pause_due,
    rounds_without_improvement,
    stop_reason,
)

__all__ = [
    "CONSECUTIVE_FAILURES",
    "CycleLimits",
    "DEFAULT_CANDIDATES",
    "DEFAULT_FIX_RETRIES",
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "DEFAULT_MAX_REJECTIONS",
    "DEFAULT_PAUSE_EVERY",
    "DEFAULT_ROUNDS",
    "DEFAULT_STAGNATION_ROUNDS",
    "MAX_ROUNDS",
    "STAGNATION",
    "STOP_REASONS",
    "TARGET_REACHED",
    "pause_due",
    "rounds_without_improvement",
    "stop_reason",
]
