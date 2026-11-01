"""Parameters and caps of the cycle: provisional, configurable defaults (case 1)."""

import pytest

from kaggle_agent.budget import (
    DEFAULT_CANDIDATES,
    DEFAULT_FIX_RETRIES,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    DEFAULT_MAX_REJECTIONS,
    DEFAULT_PAUSE_EVERY,
    DEFAULT_ROUNDS,
    CycleLimits,
)
from kaggle_agent.proposals import ValidationError
from registry_helpers import make_cycle


def test_defaults_overrides_cycle_values_and_validation():
    limits = CycleLimits()
    assert (limits.rounds, limits.candidates, limits.pause_every, limits.fix_retries) == (5, 3, 1, 3)
    assert limits.max_consecutive_failures == DEFAULT_MAX_CONSECUTIVE_FAILURES >= 1
    assert limits.max_rejections == DEFAULT_MAX_REJECTIONS >= 1
    assert (DEFAULT_ROUNDS, DEFAULT_CANDIDATES, DEFAULT_PAUSE_EVERY, DEFAULT_FIX_RETRIES) == (5, 3, 1, 3)
    custom = CycleLimits(rounds=8, candidates=4, pause_every=2, max_rejections=5)
    assert (custom.rounds, custom.candidates, custom.pause_every, custom.max_rejections) == (8, 4, 2, 5)

    cycle = make_cycle("C1", iterations=7, stagnation_k=2, max_fix_retries=1)
    resolved = CycleLimits.resolve(cycle, CycleLimits(rounds=99, stagnation_rounds=9, fix_retries=9))
    assert (resolved.rounds, resolved.stagnation_rounds, resolved.fix_retries) == (7, 2, 1)
    assert resolved.candidates == DEFAULT_CANDIDATES  # injected fallback: no stored parameters

    for bad in ({"candidates": 0}, {"pause_every": 0}, {"max_consecutive_failures": -1},
                {"max_rejections": 0}, {"rounds": 0}, {"fix_retries": -1}, {"candidates": True},
                {"candidates": 1.5}):
        with pytest.raises(ValidationError):
            CycleLimits(**bad)
    assert CycleLimits(fix_retries=0).fix_retries == 0


def test_to_dict_from_dict_round_trip_and_stored_values_win():
    limits = CycleLimits(candidates=2, pause_every=3, max_consecutive_failures=4, max_rejections=5)
    stored = limits.to_dict()
    assert stored == {
        "candidates": 2, "pause_every": 3, "max_consecutive_failures": 4, "max_rejections": 5,
    }
    cycle = make_cycle("C1", iterations=6, stagnation_k=2, max_fix_retries=2, parameters=stored)
    resolved = CycleLimits.resolve(cycle, CycleLimits(candidates=9, pause_every=9))
    assert resolved == CycleLimits(
        rounds=6, candidates=2, stagnation_rounds=2, pause_every=3, fix_retries=2,
        max_consecutive_failures=4, max_rejections=5,
    )
    assert CycleLimits.from_dict(stored, rounds=6, stagnation_rounds=2, fix_retries=2) == resolved
    with pytest.raises(ValidationError):
        CycleLimits.from_dict({"unknown": 1})
