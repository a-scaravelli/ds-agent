"""The state machine: allowed transitions, persistence and the explicit plan approval (cases 4, 25)."""

import pytest
from loop_helpers import CYCLE, make_setup, record_states

from kaggle_agent.loop import Hooks, TransitionError, check_transition, STATES, TRANSITIONS
from kaggle_agent.loop.states import (
    AWAITING_GUIDANCE,
    AWAITING_PLAN_APPROVAL,
    AWAITING_REVIEW,
    BASELINE,
    ITERATING,
    PLANNING,
    REPORTING,
    SPLITTING,
    STOPPED,
)
from kaggle_agent.review import Approve, Revise, SimulatedLead


def test_every_transition_is_persisted_and_only_allowed_ones_occur(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5)
    seen = record_states(setup.registry)
    outcome = setup.run()
    assert outcome.status == "stopped" and outcome.state == STOPPED
    path = [PLANNING] + seen
    assert path == [
        PLANNING, AWAITING_PLAN_APPROVAL, SPLITTING, BASELINE, ITERATING, AWAITING_GUIDANCE,
        ITERATING, AWAITING_GUIDANCE, REPORTING, AWAITING_REVIEW, STOPPED,
    ]
    for current, new in zip(path, path[1:]):
        check_transition(current, new)
    assert setup.cycle().state == STOPPED
    assert set(path) == set(STATES)


def test_an_illegal_transition_raises():
    assert set(TRANSITIONS) == set(STATES) and TRANSITIONS[STOPPED] == ()
    for current, new in ((PLANNING, ITERATING), (STOPPED, PLANNING), (SPLITTING, REPORTING),
                         (AWAITING_REVIEW, ITERATING), (REPORTING, STOPPED)):
        with pytest.raises(TransitionError):
            check_transition(current, new)
    with pytest.raises(TransitionError):
        check_transition("NOWHERE", PLANNING)


def test_unknown_saved_state_is_refused(tmp_path):
    setup = make_setup(tmp_path, state="SOMEWHERE")
    with pytest.raises(TransitionError):
        setup.run()


def test_revise_returns_to_planning_and_freeze_happens_once_on_approval(tmp_path):
    plans = []
    hooks = Hooks(plan=lambda ctx, revisions: plans.append(revisions))
    lead = SimulatedLead(plans=[Revise("use another metric"), Approve()])
    setup = make_setup(tmp_path, rounds=1, candidates=1, hooks=hooks, lead=lead)
    seen = []
    original = setup.registry.set_cycle_state

    def spy(cycle_id, state):
        seen.append((state, setup.registry.get_cycle(cycle_id).protocol_hash, len(setup.freezes)))
        original(cycle_id, state)

    setup.registry.set_cycle_state = spy
    setup.run()
    assert [state for state, _, _ in seen][:4] == [
        AWAITING_PLAN_APPROVAL, PLANNING, AWAITING_PLAN_APPROVAL, SPLITTING,
    ]
    assert plans == [(), ("use another metric",)]  # the hook runs again, with the lead's request
    assert len(setup.freezes) == 1
    # before the plan is approved nothing is frozen; the hash is set when SPLITTING is saved
    assert all(hash_ is None and freezes == 0 for state, hash_, freezes in seen[:3])
    state, hash_, freezes = seen[3]
    assert state == SPLITTING and hash_ is not None and freezes == 1
    assert setup.cycle().protocol_hash == hash_


def test_no_answer_to_the_plan_keeps_the_state_and_freezes_nothing(tmp_path):
    setup = make_setup(tmp_path, rounds=1, candidates=1, lead=SimulatedLead(plans=[None]))
    outcome = setup.run()
    assert (outcome.status, outcome.state) == ("waiting", AWAITING_PLAN_APPROVAL)
    assert setup.cycle().state == AWAITING_PLAN_APPROVAL
    assert setup.freezes == [] and setup.cycle().protocol_hash is None
    again = setup.run()  # the lead answers now (the script is exhausted: the default approves)
    assert again.status == "stopped" and len(setup.freezes) == 1
