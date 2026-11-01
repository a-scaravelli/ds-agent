"""The limits are stored on the cycle at the first start and resume uses them (case 32)."""

import pytest
from loop_helpers import Boom, CYCLE, make_setup

from kaggle_agent.budget import CycleLimits
from kaggle_agent.review import SimulatedLead

FIRST = CycleLimits(candidates=2, pause_every=2, max_consecutive_failures=4, max_rejections=3)
OTHER = CycleLimits(candidates=5, pause_every=1, max_consecutive_failures=9, max_rejections=7)


class CrashAtPlan(SimulatedLead):
    def plan_decision(self, context):
        raise Boom


def test_limits_are_written_once_and_a_resumed_run_ignores_the_injected_ones(tmp_path):
    setup = make_setup(tmp_path, rounds=4, candidates=2, stagnation_k=9, lead=CrashAtPlan())
    setup.ctx = setup.ctx.__class__(**{**setup.ctx.__dict__, "limits": FIRST})
    assert setup.cycle().parameters is None
    with pytest.raises(Boom):
        setup.run()
    assert setup.cycle().parameters == FIRST.to_dict()  # written at the first start

    machine = setup.machine(limits=OTHER, lead=SimulatedLead())
    outcome = machine.run()
    assert outcome.status == "stopped"
    assert setup.cycle().parameters == FIRST.to_dict()  # not overwritten
    assert machine.limits == CycleLimits(
        rounds=4, candidates=2, stagnation_rounds=9, pause_every=2, fix_retries=3,
        max_consecutive_failures=4, max_rejections=3,
    )
    rounds = [r for r in setup.registry.list_rounds_with_candidates(CYCLE) if not r.round.id.endswith("-R0")]
    assert [len(r.candidates) for r in rounds] == [2, 2, 2, 2]  # C=2, not 5
    assert [c.round_position for c in machine.ctx.lead.asked("guidance")] == [2, 4]  # M=2, not 1


def test_an_older_cycle_without_parameters_takes_the_injected_limits_and_stores_them(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=9)
    assert setup.cycle().parameters is None
    machine = setup.machine(limits=FIRST)
    machine.run()
    assert setup.cycle().parameters == FIRST.to_dict()
    assert machine.limits.candidates == 2 and machine.limits.pause_every == 2
    rounds = [r for r in setup.registry.list_rounds_with_candidates(CYCLE) if not r.round.id.endswith("-R0")]
    assert [len(r.candidates) for r in rounds] == [2, 2]


def test_a_cycle_with_stored_parameters_ignores_the_injected_ones_from_the_start(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=9, parameters=FIRST.to_dict())
    machine = setup.machine(limits=OTHER)
    machine.run()
    assert machine.limits.candidates == 2 and machine.limits.max_rejections == 3
