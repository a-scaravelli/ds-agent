"""The round pipeline end to end with fakes (cases 5 to 10, 23, 24)."""

import json

from loop_helpers import (
    CYCLE,
    FakeRoles,
    RoleClient,
    by_round,
    closed_outcomes,
    make_setup,
    record_states,
)

from kaggle_agent.bundle import load_bundle
from kaggle_agent.llm.fake import Malformed, Reply
from kaggle_agent.loop import FakeBundleWriter, Hooks
from kaggle_agent.loop.states import AWAITING_GUIDANCE, ITERATING
from kaggle_agent.review import LeadResponse, SimulatedLead


def iteration_rounds(registry):
    return [r for r in registry.list_rounds_with_candidates(CYCLE) if not r.round.id.endswith("-R0")]


def test_happy_path_over_several_rounds_with_fake_roles_and_a_simulated_lead(tmp_path):
    reports = []
    hooks = Hooks(report=lambda ctx, reason: reports.append(reason))
    setup = make_setup(
        tmp_path, rounds=3, candidates=2, pause_every=1, hooks=hooks,
        rule=by_round([0.25, 0.2, 0.15]),
    )
    seen = record_states(setup.registry)
    outcome = setup.run()
    assert (outcome.status, outcome.state, outcome.reason) == ("stopped", "STOPPED", "max_rounds")
    assert reports == ["max_rounds"]  # REPORTING is entered and left through the hook
    rounds = iteration_rounds(setup.registry)
    assert [len(r.candidates) for r in rounds] == [2, 2, 2]
    for round_detail in rounds:
        for candidate in round_detail.candidates:
            detail = setup.registry.get_attempt(candidate.attempt_id)
            assert detail.closed and detail.outcome == "succeeded"
            assert detail.attempt.bundle_id and len(detail.executions) == 1
        assert [r.id for r in setup.registry.list_reviewer_reports(CYCLE, round_detail.round.id)] == [
            f"{round_detail.round.id}-rep"
        ]
    assert len(setup.lead.asked("guidance")) == 3  # after every round, the last one included
    assert seen[-3:] == ["REPORTING", "AWAITING_REVIEW", "STOPPED"]
    assert setup.registry.count_comparisons(CYCLE) == 3  # one comparison per round


def test_each_round_passes_outputs_between_roles_in_the_documented_order(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=2, stagnation_k=5)
    fake = FakeRoles(setup.registry, setup.clock)
    setup.run(roles=fake.roleset())
    names = fake.names()
    one_round = ["data", "feature_engineering", "modeling"] * 2 + ["reviewer", "coordinator"]
    assert names == ["coordinator", "data", "feature_engineering", "modeling"] + one_round + (
        ["data", "feature_engineering", "modeling"] * 2 + ["reviewer"]
    )
    by_name = {}
    for name, role_input in fake.calls:
        by_name.setdefault(name, []).append(role_input)
    first_data = fake.calls[1:4]
    # the loop hands the data findings to feature engineering and modeling, and the FE attempt on
    fe_input, modeling_input = first_data[1][1], first_data[2][1]
    assert fe_input.findings.profile == "profile 2" == modeling_input.findings.profile
    assert modeling_input.previous_attempt.what_changed == "feature_engineering change 3"
    assert modeling_input.previous_attempt.id == modeling_input.candidate_id == "C1-R0-c1"
    # the Round row holds the tasks of the coordinator and its candidates are the attempts
    round_one = setup.registry.get_round("C1-R1")
    assert round_one.tasks == ("task 1",)
    assert round_one.candidates == ("C1-R1-c1", "C1-R1-c2")
    second = setup.registry.get_round("C1-R2")
    assert second.tasks != round_one.tasks
    traces = setup.registry.list_role_traces(CYCLE, role="coordinator", round_id="C1-R1")
    assert tuple(traces[-1].output["tasks"]) == second.tasks


def test_bundle_production_goes_through_the_open_writer_interface(tmp_path):
    setup = make_setup(tmp_path, rounds=1, candidates=1)
    setup.run()
    request = setup.writer.requests[1]  # 0 is the baseline
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert request.attempt.id == "C1-R1-c1" and request.outputs.modeling.id == "C1-R1-c1"
    assert request.outputs.findings.profile and request.outputs.feature_engineering.id == "C1-R1-c1"
    assert request.fix is None
    loaded = load_bundle(request.folder)
    assert loaded.bundle_id == request.bundle_id == detail.attempt.bundle_id
    record = setup.registry.get_bundle(detail.attempt.bundle_id)
    assert record.folder_hash == loaded.folder_hash and record.cycle_id == CYCLE
    assert setup.factory.runs.count(request.bundle_id) == 1  # scored through the runner factory
    execution = detail.executions[0]
    assert execution.code_hash == loaded.folder_hash and execution.outcome == "succeeded"
    assert any(m.partition == "validation" for m in execution.metrics)


def test_an_invalid_bundle_layout_is_rejected_and_retried_with_the_reason(tmp_path):
    first = []

    def invalid_once(request):  # the first bundle of the first iteration candidate is broken
        if request.bundle_id == "C1-R1-c1-b1" and not first:
            first.append(request)
            return True
        return False

    writer = FakeBundleWriter(invalid_for=invalid_once)
    setup = make_setup(tmp_path, rounds=1, candidates=1, writer=writer)
    setup.run()
    detail = setup.registry.get_attempt("C1-R1-c1")
    assert detail.attempt.bundle_id == "C1-R1-c1-b1" and detail.outcome == "succeeded"
    fe_prompts = [p for role, p in setup.client.prompts if role == "feature_engineering"]
    assert "was rejected" in fe_prompts[-1] and "not found in the bundle folder" in fe_prompts[-1]
    assert [a.attempt_id for a in setup.registry.list_attempts(CYCLE)].count("C1-R1-c1") == 1


def test_guidance_is_a_suggestion_and_is_read_by_the_next_round(tmp_path):
    lead = SimulatedLead(guidance=[LeadResponse("guidance", "focus on tree models")])
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5, lead=lead)
    fake = FakeRoles(setup.registry, setup.clock)
    setup.run(roles=fake.roleset())
    guidance = setup.registry.list_guidance(CYCLE)
    assert [(g.id, g.round_id, g.text) for g in guidance] == [
        ("C1-R1-g", "C1-R1", "focus on tree models")
    ]
    suggestions = setup.registry.list_suggestions()
    assert [(s.id, s.text, s.origin_cycle) for s in suggestions] == [
        ("C1-R1-g", "focus on tree models", CYCLE)
    ]
    coordinator_inputs = [i for name, i in fake.calls if name == "coordinator"]
    assert coordinator_inputs[0].guidance == ()  # the plan, before any guidance
    assert coordinator_inputs[1].guidance == ("focus on tree models",)
    assert [r.id for r in coordinator_inputs[1].reports] == ["C1-R1-rep"]


def test_continue_is_an_input_and_silence_is_never_an_approval(tmp_path):
    setup = make_setup(
        tmp_path, rounds=3, candidates=1, stagnation_k=5,
        lead=SimulatedLead(guidance=[LeadResponse("continue"), None]),
    )
    outcome = setup.run()
    assert (outcome.status, outcome.state) == ("waiting", AWAITING_GUIDANCE)
    assert setup.registry.list_guidance(CYCLE) == []  # continue records nothing
    assert len(iteration_rounds(setup.registry)) == 2  # round 2 started after the continue
    coordinator_traces = setup.registry.list_role_traces(CYCLE, role="coordinator")
    assert [t.round_id for t in coordinator_traces] == ["C1-R0", "C1-R1"]  # none after round 2
    again = setup.run()
    assert again.status == "stopped" and len(iteration_rounds(setup.registry)) == 3


def test_m_rounds_in_a_row_without_a_pause(tmp_path):
    setup = make_setup(tmp_path, rounds=4, candidates=1, pause_every=2, stagnation_k=9)
    seen = record_states(setup.registry)
    setup.run()
    asked = [c.round_position for c in setup.lead.asked("guidance")]
    assert asked == [2, 4]
    start = seen.index(ITERATING)
    assert seen[start:start + 3] == [ITERATING, ITERATING, AWAITING_GUIDANCE]
    # the coordinator still gave a direction after round 1, with no pause
    after_first = setup.registry.list_role_traces(CYCLE, role="coordinator", round_id="C1-R1")
    assert len(after_first) == 1 and after_first[0].output is not None


def test_the_reviewer_is_never_a_gate(tmp_path):
    def run(name, overrides):
        folder = tmp_path / name
        folder.mkdir()
        setup = make_setup(
            folder, rounds=2, candidates=1, stagnation_k=5, client=RoleClient(overrides=overrides)
        )
        setup.run()
        return setup

    normal = run("normal", {})
    rejected = run("rejected", {"reviewer": [Malformed()]})
    negative = run("negative", {"reviewer": [Reply(json.dumps({"text": "everything failed, stop now"}))]})
    # the rejected reviewer leaves a fixed-code report, the negative one is stored as written
    assert rejected.registry.list_reviewer_reports(CYCLE)[0].text.startswith("Automatic summary")
    assert negative.registry.list_reviewer_reports(CYCLE)[0].text == "everything failed, stop now"
    reference = closed_outcomes(normal.registry)
    assert closed_outcomes(rejected.registry) == reference == closed_outcomes(negative.registry)
    for setup in (normal, rejected, negative):
        assert setup.cycle().state == "STOPPED"
        assert len(setup.registry.list_reviewer_reports(CYCLE)) == 2


class _CheckingLead(SimulatedLead):
    def __init__(self, registry, **kwargs):
        super().__init__(**kwargs)
        self.registry = registry
        self.states = []

    def guidance(self, context):
        self.states.append(self.registry.get_cycle(CYCLE).state)
        return super().guidance(context)


def test_a_protocol_change_is_refused_and_needs_a_new_plan_approval(tmp_path):
    setup = make_setup(tmp_path, rounds=2, candidates=1, stagnation_k=5)
    lead = _CheckingLead(
        setup.registry,
        guidance=[LeadResponse("protocol_change", "switch to log loss"), LeadResponse("continue")],
    )
    setup.ctx = setup.ctx.__class__(**{**setup.ctx.__dict__, "lead": lead})
    outcome = setup.run()
    assert outcome.status == "stopped"
    assert lead.states[:2] == [AWAITING_GUIDANCE, AWAITING_GUIDANCE]
    assert len(lead.notices) == 1 and "new plan approval" in lead.notices[0]
    assert setup.registry.list_guidance(CYCLE) == [] and setup.registry.list_suggestions() == []
    assert setup.cycle().protocol_hash == setup.engine.current_hash()  # unchanged
    assert len(setup.freezes) == 1
