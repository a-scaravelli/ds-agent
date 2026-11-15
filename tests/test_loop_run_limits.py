"""The roles are told the run limits, from the stored cycle parameters (cases 2, 3, 4, 6, 10, 11, 12)."""

import json
import re
from pathlib import Path

from loop_helpers import CYCLE, Boom, RoleClient, by_round, make_setup
from orchestrator_helpers import World

from kaggle_agent.agents import CoordinatorInput, RegistrySummary, RunLimits
from kaggle_agent.budget import CycleLimits
from kaggle_agent.llm.fake import RateLimited, Reply, valid_plan_json
from kaggle_agent.loop import Hooks, ids, role_limits
from kaggle_agent.orchestrator import build_context, load_config
from kaggle_agent.orchestrator.lead_file import Ledger, make_plan_hook
from kaggle_agent.proposals import ALLOWED_LIBRARIES, Round
from kaggle_agent.registry import RoleTrace
from kaggle_agent.review import Revise, SimulatedLead

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
OTHER = CycleLimits(rounds=9, candidates=7, stagnation_rounds=5, pause_every=1, fix_retries=1)
STORED = {"candidates": 2, "pause_every": 1, "max_consecutive_failures": 3, "max_rejections": 3}


class CrashAtPlan(SimulatedLead):
    def plan_decision(self, context):
        raise Boom


def sub(path):
    path.mkdir(parents=True)
    return path


def plan_reply(baseline="majority class"):
    plan = json.loads(valid_plan_json())
    plan["baseline"] = baseline
    return Reply(json.dumps({"plan": plan, "tasks": ["task"], "goal": None}))


def inputs(registry, role, round_id=None):
    return [t.input for t in registry.list_role_traces(CYCLE, role=role, round_id=round_id)]


def limits_of(traces_input):
    return traces_input["limits"]


# 2
def test_the_limits_come_from_the_stored_parameters_and_fall_back_to_the_context(tmp_path):
    setup = make_setup(sub(tmp_path / "a"), rounds=3, candidates=2, parameters=STORED, lead=CrashAtPlan())
    injected = setup.ctx.__class__(**{**setup.ctx.__dict__, "limits": CycleLimits()})  # 5 rounds, 3 candidates
    try:
        setup.machine(limits=CycleLimits()).run()
    except Boom:
        pass
    (first,) = inputs(setup.registry, "coordinator")
    assert (limits_of(first)["rounds"], limits_of(first)["candidates"]) == (3, 2)
    assert role_limits(injected, 4).current_round == 4 and role_limits(injected, 4).candidates == 2

    bare = make_setup(sub(tmp_path / "b"), rounds=3, candidates=2, parameters=None, lead=CrashAtPlan())
    fallback = role_limits(bare.ctx, 0)  # nothing stored yet: the injected limits are used
    assert fallback.candidates == bare.ctx.limits.candidates == 2 and fallback.rounds == 3
    wide = bare.ctx.__class__(**{**bare.ctx.__dict__, "limits": CycleLimits(candidates=4, rounds=3)})
    assert bare.registry.get_cycle(CYCLE).parameters is None
    assert role_limits(wide, 0).candidates == 4
    try:
        bare.machine(limits=CycleLimits(candidates=4, rounds=3)).run()
    except Boom:
        pass
    assert limits_of(inputs(bare.registry, "coordinator")[0])["candidates"] == 4


# 3
def test_a_resumed_cycle_sends_the_same_limits(tmp_path):
    client = RoleClient(overrides={"coordinator": [plan_reply(), RateLimited(retry_after=1)]})
    setup = make_setup(tmp_path, rounds=3, candidates=2, client=client)
    assert setup.run().status == "suspended"  # the direction after round 1 hit a rate limit
    first = inputs(setup.registry, "coordinator")
    assert len(first) == 2 and limits_of(first[0])["current_round"] == 0
    assert limits_of(first[1])["current_round"] == 1
    outcome = setup.run(limits=OTHER)  # rebuilt context, different injected limits
    assert outcome.status == "stopped"
    traces = inputs(setup.registry, "coordinator")
    again = traces[2]  # the direction after round 1, asked again
    assert limits_of(again) == limits_of(first[1])
    for value in traces:
        assert {k: v for k, v in limits_of(value).items() if k != "current_round"} == {
            k: v for k, v in limits_of(first[0]).items() if k != "current_round"}


def test_the_plan_hook_after_a_revision_and_a_resume_keeps_the_limits(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.append(1, "keep it simple", "2026-11-01T00:00:00+00:00")
    client = RoleClient(overrides={"coordinator": [RateLimited(retry_after=1)]})
    setup = make_setup(sub(tmp_path / "w"), rounds=3, candidates=2, client=client,
                       hooks=Hooks(plan=make_plan_hook(ledger)), lead=SimulatedLead(plans=[Revise("more trees")]))
    assert setup.run().status == "suspended"
    assert setup.run(limits=OTHER).status == "stopped"
    first, second, third = inputs(setup.registry, "coordinator")[:3]
    assert limits_of(first) == limits_of(second) == limits_of(third) and limits_of(first)["rounds"] == 3
    assert first["guidance"] == ["keep it simple"]


# 4
def test_the_current_round_is_right_at_every_call_site(tmp_path):
    setup = make_setup(tmp_path, rounds=3, candidates=2, rule=by_round([0.25, 0.2, 0.15]))
    assert setup.run().status == "stopped"
    seen = {}
    for trace in setup.registry.list_role_traces(CYCLE):
        index = ids.round_index(CYCLE, trace.round_id)
        if trace.role == "data":
            assert "limits" not in trace.input
            continue
        assert trace.input["limits"]["current_round"] == index, (trace.role, trace.round_id)
        assert trace.input["limits"]["rounds"] == 3 and trace.input["limits"]["candidates"] == 2
        seen.setdefault(trace.role, set()).add(index)
    assert seen["coordinator"] == {0, 1, 2}  # no direction call after the last round
    assert seen["feature_engineering"] == seen["modeling"] == {0, 1, 2, 3}  # 0 is the baseline
    assert seen["reviewer"] == {1, 2, 3}


# 11
def test_a_plan_beyond_the_last_round_is_not_rejected_by_code(tmp_path):
    client = RoleClient(overrides={"coordinator": [plan_reply("Round 4 onward: tune the trees")]})
    setup = make_setup(tmp_path, rounds=3, candidates=2, client=client, lead=CrashAtPlan())
    try:
        setup.run()
    except Boom:
        pass
    (trace,) = setup.registry.list_role_traces(CYCLE, role="coordinator")
    assert trace.outcome == "ok" and "Round 4 onward" in json.dumps(trace.output)
    assert client.roles_called == ["coordinator"]  # no retry
    assert "Never plan beyond the last round" in " ".join(client.prompts[0][1].split())


# 10
def test_a_stored_trace_without_limits_reads_back_and_parses(tmp_path):
    setup = make_setup(tmp_path)
    old = CoordinatorInput(
        data=setup.ctx.data_summary, metric="accuracy", history=RegistrySummary(),
    ).to_dict()
    assert "limits" not in old
    setup.registry.add_round(CYCLE, Round(ids.setup_round_id(CYCLE), ("t",)))
    setup.registry.add_role_trace(RoleTrace(
        cycle_id=CYCLE, role="coordinator", round_id=ids.setup_round_id(CYCLE),
        input_type="CoordinatorInput", input=old, outcome="invalid_output",
        created_at="2026-10-30T10:00:00+00:00",
    ))
    (stored,) = setup.registry.list_role_traces(CYCLE, role="coordinator")
    assert stored.input == old
    parsed = CoordinatorInput.from_dict(stored.input)
    assert parsed.limits is None and "limits" not in parsed.to_dict()


# 12
def test_the_plan_wrapper_and_the_default_path_build_identical_limits(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.append(1, "use trees", "2026-11-01T00:00:00+00:00")
    wrapped = make_setup(sub(tmp_path / "w"), hooks=Hooks(plan=make_plan_hook(ledger)), lead=CrashAtPlan())
    plain = make_setup(sub(tmp_path / "p"), lead=CrashAtPlan())
    for setup in (wrapped, plain):
        try:
            setup.run()
        except Boom:
            pass
    (a,) = inputs(wrapped.registry, "coordinator")
    (b,) = inputs(plain.registry, "coordinator")
    assert a["limits"] == b["limits"] and RunLimits.from_dict(a["limits"]).current_round == 0
    assert a["guidance"] == ["use trees"]


# 6
def test_a_configured_list_reaches_the_roles_and_the_writer(tmp_path):
    listed = '["numpy", "pandas", "scikit-learn", "xgboost"]'
    world = World(tmp_path, extra=f"allowed_dependencies = {listed}")
    built = build_context(load_config(world.config), world.deps)
    try:
        assert built.ctx.allowed_libraries == ("numpy", "pandas", "scikit-learn", "xgboost")
        assert built.ctx.writer.allowed_dependencies == built.ctx.allowed_libraries
        assert role_limits(built.ctx, 0).allowed_libraries == built.ctx.allowed_libraries
    finally:
        built.close()
    default = World(tmp_path / "d")
    built = build_context(load_config(default.config), default.deps)
    try:
        assert built.ctx.allowed_libraries == ALLOWED_LIBRARIES == built.ctx.writer.allowed_dependencies
    finally:
        built.close()


def test_the_list_of_libraries_is_written_in_one_place_only():
    pattern = re.compile(r"""["']scikit-learn["']\s*,\s*["']xgboost["']""")
    found = [p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py") if pattern.search(p.read_text("utf-8"))]
    assert found == ["proposals/libraries.py"]


# 5
def test_the_limits_take_the_protocol_values_and_the_rule_names_the_threshold(tmp_path):
    setup = make_setup(tmp_path, threshold_k=1.5)
    protocol = setup.engine.protocol
    built = role_limits(setup.ctx, 2)
    assert built.threshold_k == 1.5 and built.validation_scheme == protocol.validation_scheme
    assert built.repetitions == protocol.repetitions and built.current_round == 2
    assert "1.5 times the baseline dispersion" in built.significance_rule
