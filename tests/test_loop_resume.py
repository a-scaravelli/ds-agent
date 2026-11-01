"""Resume from every persisted state and in the middle of a round (cases 26, 27)."""

import dataclasses

import pytest
from loop_helpers import Boom, CYCLE, FakeRunnerFactory, RoleClient, crash, make_setup, closed_outcomes

from kaggle_agent.loop import FakeBundleWriter, Hooks, RoleSet, client_roles
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
from kaggle_agent.review import SimulatedLead


class Tripwire:
    """Raises ``Boom`` the first time it is hit, then lets everything through."""

    def __init__(self, armed=True) -> None:
        self.armed = armed

    def hit(self):
        if self.armed:
            self.armed = False
            raise Boom


class CrashingLead(SimulatedLead):
    def __init__(self, trip, where, **kwargs):
        super().__init__(**kwargs)
        self.trip, self.where = trip, where

    def plan_decision(self, context):
        if self.where == AWAITING_PLAN_APPROVAL:
            self.trip.hit()
        return super().plan_decision(context)

    def guidance(self, context):
        if self.where == AWAITING_GUIDANCE:
            self.trip.hit()
        return super().guidance(context)

    def review_decision(self, context):
        if self.where == AWAITING_REVIEW:
            self.trip.hit()
        return super().review_decision(context)


def crashing_roles(roles, trip, role, nth):
    """The role set with a crash injected before the ``nth`` call (from 1) of one role."""
    counter = {"n": 0}

    def wrap(name, function):
        def call(*args):
            if name == role:
                counter["n"] += 1
                if counter["n"] == nth:
                    trip.hit()
            return function(*args)
        return call

    return RoleSet(**{f.name: wrap(f.name, getattr(roles, f.name)) for f in dataclasses.fields(RoleSet)})


# state -> where the crash is injected: a role call (role, nth), a hook, or the lead
CRASH_POINTS = {
    PLANNING: ("role", "coordinator", 1),
    AWAITING_PLAN_APPROVAL: ("lead",),
    SPLITTING: ("hook", "split"),
    BASELINE: ("role", "data", 1),
    ITERATING: ("role", "data", 2),
    AWAITING_GUIDANCE: ("lead",),
    REPORTING: ("hook", "report"),
    AWAITING_REVIEW: ("lead",),
}


def build(tmp_path, state=None):
    """A setup with counting hooks and, when ``state`` is given, a one-shot crash in that state."""
    trip = Tripwire(armed=state is not None)
    calls = {"split": 0, "report": 0}

    def count(name):
        def hook(*args):
            if state is not None and CRASH_POINTS[state] == ("hook", name):
                trip.hit()
            calls[name] += 1
        return hook

    client = RoleClient()
    setup = make_setup(
        tmp_path, rounds=2, candidates=1, stagnation_k=5, client=client,
        hooks=Hooks(split=count("split"), report=count("report")),
        lead=CrashingLead(trip, state),
    )
    point = CRASH_POINTS.get(state)
    if point and point[0] == "role":
        setup.ctx = dataclasses.replace(
            setup.ctx, roles=crashing_roles(setup.ctx.roles, trip, point[1], point[2])
        )
    return setup, calls


def snapshot(setup):
    registry = setup.registry
    return {
        "state": setup.cycle().state,
        "roles": list(setup.client.roles_called),
        "attempts": closed_outcomes(registry),
        "bundles": [b.id for b in registry.list_bundles(CYCLE)],
        "reports": [r.id for r in registry.list_reviewer_reports(CYCLE)],
        "comparisons": registry.count_comparisons(CYCLE),
        "protocol": setup.cycle().protocol_hash,
        "runs": list(setup.factory.runs),
        "traces": len(registry.list_role_traces(CYCLE)),
        "freezes": len(setup.freezes),
    }


@pytest.mark.parametrize("state", list(CRASH_POINTS))
def test_resume_from_every_persisted_state_repeats_nothing(tmp_path, state):
    (tmp_path / "full").mkdir()
    (tmp_path / "crash").mkdir()
    reference_setup, reference_calls = build(tmp_path / "full")
    assert reference_setup.run().status == "stopped"
    expected = snapshot(reference_setup)
    assert expected["state"] == STOPPED and expected["freezes"] == 1

    setup, calls = build(tmp_path / "crash", state)
    with pytest.raises(Boom):
        setup.run()
    assert setup.cycle().state == state  # the state was saved before the crash
    freezes_at_crash = len(setup.freezes)
    outcome = setup.machine().run()  # a brand-new machine on the same registry
    assert outcome.status == "stopped"
    assert snapshot(setup) == expected  # the same records as an uninterrupted run
    assert calls == reference_calls  # finished hooks are not called again
    if state not in (PLANNING, AWAITING_PLAN_APPROVAL):
        assert freezes_at_crash == 1 and len(setup.freezes) == 1  # never frozen a second time


def test_a_stopped_cycle_is_inert(tmp_path):
    setup, calls = build(tmp_path)
    setup.run()
    before = snapshot(setup)
    saved = []
    original = setup.registry.set_cycle_state
    setup.registry.set_cycle_state = lambda *a: (saved.append(a), original(*a))
    outcome = setup.run()
    assert outcome.status == "stopped" and outcome.state == STOPPED
    assert snapshot(setup) == before and saved == []
    assert calls == {"split": 1, "report": 1}


def finish_and_check(setup, expected_attempts):
    outcome = setup.machine().run()
    assert outcome.status == "stopped"
    ids = [a.attempt_id for a in setup.registry.list_attempts(CYCLE)]
    assert ids == expected_attempts and len(set(ids)) == len(ids)
    assert len(set(setup.factory.runs)) == len(setup.factory.runs)  # no execution twice
    assert [r.id for r in setup.registry.list_reviewer_reports(CYCLE)] == ["C1-R1-rep", "C1-R2-rep"]
    assert [g.id for g in setup.registry.list_guidance(CYCLE)] == []


ATTEMPTS = ["C1-R0-c1", "C1-R1-c1", "C1-R2-c1"]


def test_resume_after_the_attempt_is_registered_but_before_it_ran(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5)
    trip = Tripwire()
    factory = setup.factory

    def guarded(loaded):
        if loaded.bundle_id == "C1-R1-c1-b1":
            trip.hit()
        return factory(loaded)

    setup.ctx = dataclasses.replace(setup.ctx, runner_factory=guarded)
    with pytest.raises(Boom):
        setup.run()
    assert setup.registry.get_attempt("C1-R1-c1").executions == ()
    writes = len(setup.writer.requests)
    finish_and_check(setup, ATTEMPTS)
    assert len(setup.registry.get_attempt("C1-R1-c1").executions) == 1
    assert len(setup.writer.requests) == writes + 1  # only the second round's candidate


def test_resume_between_a_failed_execution_and_its_fix(tmp_path):
    setup = make_setup(
        tmp_path, rounds=2, candidates=1, stagnation_k=5, fix_retries=2,
        rule=lambda b, n: crash() if b == "C1-R1-c1-b1" else 0.3,
    )
    trip = Tripwire()
    writer = setup.writer
    original = writer.write

    def write(request):
        if request.fix is not None:
            trip.hit()
        return original(request)

    writer.write = write
    with pytest.raises(Boom):
        setup.run()
    assert len(setup.registry.get_attempt("C1-R1-c1").executions) == 1
    finish_and_check(setup, ATTEMPTS)
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert [e.outcome for e in detail.executions] == ["failed", "succeeded"]
    assert len(detail.executions) <= 1 + 2
    assert [r.fix.retry for r in setup.writer.requests if r.fix] == [1]


def test_resume_after_the_reviewer_report_but_before_guidance(tmp_path):
    trip = Tripwire()

    class Crashing(SimulatedLead):
        def guidance(self, context):
            trip.hit()
            return super().guidance(context)

    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5, lead=Crashing())
    with pytest.raises(Boom):
        setup.run()
    assert [r.id for r in setup.registry.list_reviewer_reports(CYCLE)] == ["C1-R1-rep"]
    assert setup.cycle().state == AWAITING_GUIDANCE
    finish_and_check(setup, ATTEMPTS)


def test_the_trace_loss_window_between_add_attempt_and_record_trace(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5)
    trip = Tripwire()
    original = setup.registry.add_role_trace

    def add_role_trace(trace):
        if trace.candidate_id == "C1-R1-c1":
            trip.hit()
        return original(trace)

    setup.registry.add_role_trace = add_role_trace
    with pytest.raises(Boom):
        setup.run()
    assert [a.attempt_id for a in setup.registry.list_attempts(CYCLE)][-1] == "C1-R1-c1"
    calls_before = list(setup.client.roles_called)
    finish_and_check(setup, ATTEMPTS)
    # accepted behaviour: the roles' traces of that candidate stay absent, nothing is repeated
    assert [t for t in setup.registry.list_role_traces(CYCLE) if t.candidate_id == "C1-R1-c1"] == []
    assert setup.client.roles_called[: len(calls_before)] == calls_before
    assert setup.client.roles_called.count("modeling") == 3  # baseline + two rounds, none twice
    assert setup.registry.get_attempt("C1-R1-c1").outcome == "succeeded"
