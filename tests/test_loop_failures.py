"""Unhappy cases of the cycle: code errors, timeouts, rejected proposals, suspension (cases 14-17, 20-22)."""

from collections import deque

import pytest
from loop_helpers import (
    CYCLE,
    FakeRoles,
    RoleClient,
    crash,
    make_setup,
)
from registry_helpers import make_attempt, make_execution

from kaggle_agent.agents import RoleRejectedError
from kaggle_agent.bundle import BundleRunError
from kaggle_agent.llm.fake import Malformed, QuotaExhausted, RateLimited, Reply
from kaggle_agent.loop import CycleView, FakeBundleWriter, is_placeholder

TIMEOUT = BundleRunError("timed_out", "timeout", "over the limit")


def one_round(tmp_path, rule, **kwargs):
    kwargs.setdefault("max_failures", 10)
    kwargs.setdefault("stagnation_k", 5)
    setup = make_setup(tmp_path, rounds=1, candidates=1, rule=rule, **kwargs)
    setup.run()
    return setup


def always_crash(bundle_id, n):
    return crash("boom") if "-R1-c1-" in bundle_id else 0.3


def test_code_errors_are_fixed_inside_the_attempt_up_to_the_cap_then_closed_failed(tmp_path):
    setup = one_round(tmp_path, always_crash, fix_retries=3)
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert len(detail.executions) == 4  # 1 + 3 fix retries
    assert detail.closed and detail.outcome == "failed" and detail.failure_kind == "code_error"
    fixes = [r.fix for r in setup.writer.requests if r.fix is not None]
    assert [f.retry for f in fixes] == [1, 2, 3] and all(f.reason == "boom" for f in fixes)
    assert [f.previous_bundle_id for f in fixes] == [
        "C1-R1-c1-b1", "C1-R1-c1-b1-f1", "C1-R1-c1-b1-f2"
    ]
    bundle_ids = [b.id for b in setup.registry.list_bundles(CYCLE) if b.id.startswith("C1-R1-c1")]
    assert bundle_ids == ["C1-R1-c1-b1", "C1-R1-c1-b1-f1", "C1-R1-c1-b1-f2", "C1-R1-c1-b1-f3"]
    assert len({e.code_hash for e in detail.executions}) == 4  # each fix is its own bundle
    summaries = setup.registry.list_attempts(CYCLE)
    assert [a.iteration for a in summaries] == [1, 2]  # fixes consumed no iteration, no attempt


def test_a_code_error_fixed_before_the_cap_succeeds_in_the_same_attempt(tmp_path):
    setup = one_round(tmp_path, lambda b, n: crash() if b == "C1-R1-c1-b1" else 0.3)
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert [(e.outcome, e.failure_kind) for e in detail.executions] == [
        ("failed", "code_error"), ("succeeded", None)
    ]
    assert detail.closed and detail.outcome == "succeeded"
    assert detail.attempt.bundle_id == "C1-R1-c1-b1"
    assert [a.iteration for a in setup.registry.list_attempts(CYCLE)] == [1, 2]


def test_a_timeout_closes_the_attempt_without_a_fix(tmp_path):
    setup = one_round(tmp_path, lambda b, n: TIMEOUT if "-R1-c1-" in b else 0.3)
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert len(detail.executions) == 1 and detail.executions[0].failure_kind == "timeout"
    assert detail.closed and (detail.outcome, detail.failure_kind) == ("failed", "timeout")
    assert all(r.fix is None for r in setup.writer.requests)


@pytest.mark.parametrize("status", ["protected_modified", "diff_rejected", "folder_changed"])
def test_integrity_failures_are_never_retried(tmp_path, status):
    error = BundleRunError(status, "code_error", f"{status} happened")
    setup = one_round(tmp_path, lambda b, n: error if "-R1-c1-" in b else 0.3)
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert len(detail.executions) == 1
    assert detail.executions[0].resources == {"sandbox_status": status, "retryable": False}
    assert detail.closed and (detail.outcome, detail.failure_kind) == ("failed", "code_error")
    assert all(r.fix is None for r in setup.writer.requests)
    assert setup.factory.runs.count("C1-R1-c1-b1") == 1


def test_a_rejected_proposal_goes_back_with_the_reason_and_a_duplicate_too(tmp_path):
    class DuplicateOnce(FakeBundleWriter):
        def write(self, request):  # the baseline and the first try of C1-R1-c1 share their code
            self.duplicate = len(self.requests) < 2
            return super().write(request)

    setup = make_setup(
        tmp_path, rounds=1, candidates=1, stagnation_k=5, writer=DuplicateOnce(), max_rejections=3
    )
    rejection = RoleRejectedError("feature_engineering", 1, "evidence: is required")
    fake = FakeRoles(
        setup.registry, setup.clock, same_change=True,
        reject={"feature_engineering": {"C1-R1-c1": [rejection]}},
    )
    setup.run(roles=fake.roleset())
    tasks = [i.task for name, i in fake.calls if name == "feature_engineering"]
    # baseline, rejected try, retry (rejected as duplicate), retry
    assert "evidence: is required" in tasks[2] and "rejected" in tasks[2]
    assert "duplicate of the earlier attempt C1-R0-c1" in tasks[3]
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert detail.outcome == "succeeded" and detail.attempt.bundle_id == "C1-R1-c1-b2"
    assert [a.attempt_id for a in setup.registry.list_attempts(CYCLE)] == ["C1-R0-c1", "C1-R1-c1"]


def rejected(role):
    return RoleRejectedError(role, 1, "hypothesis: not falsifiable")


def test_consecutive_rejections_end_as_inconclusive_and_placeholders_are_excluded(tmp_path):
    setup = make_setup(tmp_path, rounds=1, candidates=2, stagnation_k=5, max_rejections=2)
    reg = setup.registry
    fake = FakeRoles(
        setup.registry, setup.clock,
        reject={"modeling": {"C1-R1-c1": [rejected("modeling"), rejected("modeling")]}},
    )
    # ordinary neighbours in the same cycle: a falsified hypothesis and an inconclusive attempt
    reg.add_attempt(CYCLE, make_attempt("X1", what_changed="falsified"), "fp-x1")
    reg.add_execution(make_execution("X1", outcome="failed", failure_kind="hypothesis_falsified",
                                     reason="did not help"))
    reg.close_attempt("X1", "falsified")
    reg.add_attempt(CYCLE, make_attempt("X2", what_changed="unclear"), "fp-x2")
    reg.add_execution(make_execution("X2", outcome="inconclusive", failure_kind="below_threshold",
                                     reason="not enough rows"))
    reg.close_attempt("X2", "inconclusive")
    setup.run(roles=fake.roleset())

    placeholder = reg.get_attempt("C1-R1-c1")
    assert placeholder.closed and placeholder.outcome == "inconclusive"
    assert len(placeholder.executions) == 1
    assert placeholder.executions[0].reason.startswith("no valid proposal after 2 rejections")
    assert reg.get_attempt("C1-R1-c2").outcome == "succeeded"  # the round went on
    assert is_placeholder(placeholder)
    assert not is_placeholder(reg.get_attempt("X1")) and not is_placeholder(reg.get_attempt("X2"))
    assert "C1-R1-c1" in {f.attempt_id for f in reg.list_failures("hypothesis_falsified")}

    view = CycleView(reg, CYCLE, setup.engine)
    summary = view.registry_summary()
    assert "C1-R1-c1" not in {d.id for d in summary.best + summary.recent_failures}
    assert {"X1", "X2"} <= {d.id for d in summary.recent_failures}  # ordinary ones are kept
    assert view.round_states()[0].best.attempt.id == "C1-R1-c2"
    # the reviewer still sees the placeholder, as inconclusive
    reviewer_input = [i for name, i in fake.calls if name == "reviewer"][0]
    assert {r.candidate_id: r.outcome for r in reviewer_input.results} == {
        "C1-R1-c1": "inconclusive", "C1-R1-c2": "succeeded"
    }


def test_a_placeholder_never_becomes_the_baseline_and_does_not_touch_the_failure_count(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=2, stagnation_k=5, max_rejections=1)
    fake = FakeRoles(
        setup.registry, setup.clock,
        reject={"modeling": {"C1-R0-c1": [rejected("modeling")]}},
    )
    setup.run(roles=fake.roleset())
    view = CycleView(setup.registry, CYCLE, setup.engine)
    assert is_placeholder(setup.registry.get_attempt("C1-R0-c1"))
    assert view.baseline() is None
    # without a baseline the first round's best is the reference and is not compared
    comparisons = setup.registry.list_comparisons(CYCLE)
    assert [c.candidate_attempt_id for c in comparisons] == ["C1-R2-c1"]
    assert comparisons[0].reference_attempt_id == "C1-R1-c1"


def test_placeholders_neither_increment_nor_reset_the_failure_count(tmp_path):
    def rule(bundle_id, n):
        return TIMEOUT if "-R1-c1-" in bundle_id or "-R1-c3-" in bundle_id else 0.3

    setup = make_setup(
        tmp_path, rounds=2, candidates=3, stagnation_k=5, max_rejections=1, max_failures=2, rule=rule
    )
    fake = FakeRoles(
        setup.registry, setup.clock,
        reject={"modeling": {"C1-R1-c2": [rejected("modeling")]}},
    )
    outcome = setup.run(roles=fake.roleset())
    # failed, placeholder, failed: two failures in a row, the placeholder in between is ignored
    assert outcome.reason == "consecutive_failures"
    ids = [a.attempt_id for a in setup.registry.list_attempts(CYCLE)]
    assert ids == ["C1-R0-c1", "C1-R1-c1", "C1-R1-c2", "C1-R1-c3"]


def test_rate_limit_and_quota_suspend_the_cycle_without_losing_state(tmp_path):
    client = RoleClient(overrides={"data": [RateLimited()]})
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5, client=client)
    outcome = setup.run()
    assert outcome.status == "suspended" and "rate limited" in outcome.reason
    assert outcome.state == "BASELINE" == setup.cycle().state
    assert setup.registry.list_attempts(CYCLE) == []
    traces = setup.registry.list_role_traces(CYCLE, role="data")
    assert [(t.outcome, t.output) for t in traces] == [("rate_limited", None)]  # the failed call is traced
    assert [c.outcome for c in setup.registry.list_llm_calls(CYCLE, role="data")] == ["rate_limited"]

    # later, with the client working again (its override is used up), the run continues
    client.overrides["coordinator"] = deque([QuotaExhausted()])  # the next coordinator call
    resumed = setup.run()
    assert resumed.status == "suspended" and "quota" in resumed.reason
    assert resumed.state == "AWAITING_GUIDANCE"  # round 1 finished; its direction call failed
    runs_before = list(setup.factory.runs)
    writes_before = len(setup.writer.requests)
    final = setup.run()  # the quota is back
    assert final.status == "stopped"
    assert [r for r in setup.factory.runs if r in runs_before] == runs_before
    assert len(setup.factory.runs) == len(set(setup.factory.runs))  # no bundle ran twice
    assert len(setup.writer.requests) > writes_before
    assert len({a.attempt_id for a in setup.registry.list_attempts(CYCLE)}) == 3


def test_a_rejected_coordinator_suspends_the_cycle(tmp_path):
    client = RoleClient()
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5, client=client)
    client.overrides["coordinator"] = deque(
        [Reply(client.default_reply("coordinator", 1)), Malformed()]
    )
    outcome = setup.run()
    assert outcome.status == "suspended" and outcome.state == "AWAITING_GUIDANCE"
    assert "no valid output" in outcome.reason
    assert setup.registry.list_reviewer_reports(CYCLE) != []  # nothing finished was lost
