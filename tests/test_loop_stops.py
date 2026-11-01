"""Stop rules: stagnation, target and the failure cap (cases 11, 12, 13, 19)."""

import pytest
from loop_helpers import Boom, CYCLE, by_round, make_setup

from kaggle_agent.bundle import BundleRunError
from kaggle_agent.loop import Hooks
from kaggle_agent.review import LeadResponse, SimulatedLead

TIMEOUT = BundleRunError("timed_out", "timeout", "over the limit")


def iteration_ids(registry):
    return [r.round.id for r in registry.list_rounds_with_candidates(CYCLE) if not r.round.id.endswith("-R0")]


def run_to_report(setup):
    reasons = []
    setup.ctx = setup.ctx.__class__(**{
        **setup.ctx.__dict__, "hooks": Hooks(report=lambda ctx, reason: reasons.append(reason))
    })
    outcome = setup.run()
    return outcome, reasons


def test_stagnation_stops_after_k_rounds_against_the_baseline(tmp_path):
    setup = make_setup(tmp_path, rounds=10, candidates=2, stagnation_k=2, rule=by_round([], 0.3))
    outcome, reasons = run_to_report(setup)
    assert reasons == ["stagnation"] and outcome.reason == "stagnation"
    assert iteration_ids(setup.registry) == ["C1-R1", "C1-R2"]
    comparisons = setup.registry.list_comparisons(CYCLE)
    assert setup.registry.count_comparisons(CYCLE) == 2 == len(comparisons)  # one per round
    assert [c.reference_attempt_id for c in comparisons] == ["C1-R0-c1", "C1-R0-c1"]
    assert [c.candidate_attempt_id for c in comparisons] == ["C1-R1-c1", "C1-R2-c1"]
    assert not any(c.is_real for c in comparisons)


def test_a_round_without_a_succeeded_candidate_counts_and_records_no_comparison(tmp_path):
    def rule(bundle_id, n):
        if "-R1-" in bundle_id:
            return TIMEOUT
        return 0.3

    setup = make_setup(
        tmp_path, rounds=10, candidates=2, stagnation_k=2, max_failures=10, rule=rule
    )
    outcome, reasons = run_to_report(setup)
    assert reasons == ["stagnation"]
    assert iteration_ids(setup.registry) == ["C1-R1", "C1-R2"]
    comparisons = setup.registry.list_comparisons(CYCLE)
    assert [c.candidate_attempt_id for c in comparisons] == ["C1-R2-c1"]  # none for round 1


def test_a_real_improvement_resets_the_counter_and_becomes_the_reference(tmp_path):
    setup = make_setup(
        tmp_path, rounds=10, candidates=2, stagnation_k=2, rule=by_round([0.3, 0.1], default=0.1)
    )
    outcome, reasons = run_to_report(setup)
    assert reasons == ["stagnation"]
    assert iteration_ids(setup.registry) == ["C1-R1", "C1-R2", "C1-R3", "C1-R4"]
    comparisons = setup.registry.list_comparisons(CYCLE)
    assert [c.is_real for c in comparisons] == [False, True, False, False]
    assert comparisons[0].reference_attempt_id == "C1-R0-c1"
    assert comparisons[2].reference_attempt_id == comparisons[1].candidate_attempt_id == "C1-R2-c1"


def test_the_stagnation_counter_comes_from_the_registry_not_from_memory(tmp_path):
    class Crashing(SimulatedLead):
        def guidance(self, context):
            raise Boom

    setup = make_setup(
        tmp_path, rounds=10, candidates=1, stagnation_k=2, rule=by_round([], 0.3),
        lead=Crashing(),
    )
    with pytest.raises(Boom):
        setup.run()
    assert iteration_ids(setup.registry) == ["C1-R1"]  # one stagnant round, then the crash
    assert setup.cycle().state == "AWAITING_GUIDANCE"
    outcome = setup.run(lead=SimulatedLead())  # a brand-new machine on the same registry
    assert outcome.reason == "stagnation"
    assert iteration_ids(setup.registry) == ["C1-R1", "C1-R2"]  # exactly one more stagnant round


GOAL = {"criteria": [
    {"metric": "accuracy", "threshold": 0.8, "direction": "maximize"},
    {"metric": "balanced_accuracy", "threshold": 0.9, "direction": "maximize"},
]}


def test_stop_on_target_needs_every_criterion_on_validation(tmp_path):
    setup = make_setup(
        tmp_path, rounds=6, candidates=1, stagnation_k=6, goal=GOAL,
        rule=by_round([0.15, 0.02], default=0.02),
    )
    outcome, reasons = run_to_report(setup)
    assert reasons == ["target_reached"] and outcome.reason == "target_reached"
    assert iteration_ids(setup.registry) == ["C1-R1", "C1-R2"]
    values = [
        setup.engine.goal_values_from_execution(setup.registry.get_attempt(a).executions[-1])
        for a in ("C1-R1-c1", "C1-R2-c1")
    ]
    assert values[0]["accuracy"] >= 0.8 and values[0]["balanced_accuracy"] < 0.9  # only one met
    assert all(values[1][m] >= t for m, t in (("accuracy", 0.8), ("balanced_accuracy", 0.9)))
    assert setup.cycle().state == "STOPPED"


def test_failure_cap_stops_at_once_in_the_middle_of_a_round(tmp_path):
    def rule(bundle_id, n):
        return TIMEOUT if "-R1-c1-" in bundle_id or "-R1-c2-" in bundle_id else 0.3

    setup = make_setup(tmp_path, rounds=3, candidates=3, max_failures=2, stagnation_k=5, rule=rule)
    outcome, reasons = run_to_report(setup)
    assert reasons == ["consecutive_failures"] and outcome.reason == "consecutive_failures"
    ids = [a.attempt_id for a in setup.registry.list_attempts(CYCLE)]
    assert ids == ["C1-R0-c1", "C1-R1-c1", "C1-R1-c2"]  # the third candidate never ran
    assert setup.registry.list_reviewer_reports(CYCLE) == []
    assert setup.lead.asked("guidance") == []  # straight to REPORTING, no pause


def test_a_succeeded_attempt_in_between_resets_the_failure_count(tmp_path):
    def rule(bundle_id, n):
        failing = ("-R1-c1-", "-R1-c3-", "-R2-c1-")
        return TIMEOUT if any(f in bundle_id for f in failing) else 0.3

    setup = make_setup(tmp_path, rounds=2, candidates=3, max_failures=2, stagnation_k=5, rule=rule)
    outcome, reasons = run_to_report(setup)
    # fail, succeed, fail | fail, ...: the second round's first failure follows round 1's last one
    assert reasons == ["consecutive_failures"]
    ids = [a.attempt_id for a in setup.registry.list_attempts(CYCLE)]
    assert ids[-1] == "C1-R2-c1"

    (tmp_path / "q").mkdir()
    quiet = make_setup(
        tmp_path / "q", rounds=2, candidates=3, max_failures=2, stagnation_k=5,
        rule=lambda b, n: TIMEOUT if "-R1-c1-" in b or "-R1-c3-" in b else 0.3,
    )
    outcome, reasons = run_to_report(quiet)
    assert reasons == ["max_rounds"]  # the success of c2 and of round 2 kept the count below 2
