"""Pause every M rounds and the stop reasons with their priority (cases 2 and 3)."""

import pytest

from kaggle_agent.budget import (
    CONSECUTIVE_FAILURES,
    MAX_ROUNDS,
    STAGNATION,
    STOP_REASONS,
    TARGET_REACHED,
    pause_due,
    rounds_without_improvement,
    stop_reason,
)


def test_pause_due_every_m_rounds():
    n = 6
    assert [pause_due(p, 1) for p in range(1, n + 1)] == [True] * n
    assert [p for p in range(1, n + 1) if pause_due(p, 3)] == [3, 6]
    assert [p for p in range(1, n + 1) if pause_due(p, n)] == [n]
    with pytest.raises(ValueError):
        pause_due(0, 1)
    with pytest.raises(ValueError):
        pause_due(1, 0)


def quiet(**changes):
    values = dict(
        target_met=False, rounds_done=1, max_rounds=5, stagnant_rounds=0, stagnation_k=2,
        consecutive_failures=0, max_consecutive_failures=3,
    )
    return stop_reason(**{**values, **changes})


def test_each_stop_reason_alone_and_the_priority():
    assert quiet() is None
    assert quiet(target_met=True) == TARGET_REACHED
    assert quiet(rounds_done=5) == MAX_ROUNDS
    assert quiet(stagnant_rounds=2) == STAGNATION
    assert quiet(consecutive_failures=3) == CONSECUTIVE_FAILURES
    assert STOP_REASONS == (TARGET_REACHED, CONSECUTIVE_FAILURES, MAX_ROUNDS, STAGNATION)
    everything = dict(target_met=True, rounds_done=5, stagnant_rounds=2, consecutive_failures=3)
    assert quiet(**everything) == TARGET_REACHED
    everything["target_met"] = False
    assert quiet(**everything) == CONSECUTIVE_FAILURES
    everything["consecutive_failures"] = 0
    assert quiet(**everything) == MAX_ROUNDS
    everything["rounds_done"] = 1
    assert quiet(**everything) == STAGNATION


def test_rounds_without_improvement_counts_the_trailing_run():
    assert rounds_without_improvement([]) == 0
    assert rounds_without_improvement([True, False, False]) == 2
    assert rounds_without_improvement([False, True]) == 0
