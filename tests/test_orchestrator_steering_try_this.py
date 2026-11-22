"""steering_roles: typed steering, 'Try this' bound once, resume after a crash (cases 38, 40, 47, 48, 52)."""

from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, World
from overrides_helpers import applied, attempts, decide

from kaggle_agent.agents.prompting import REQUEST_NOTE, STEERING_NOTE

STEERING = {"focus": "feature_engineering", "pin_features": ["x1"], "avoid_features": ["id"], "try_families": [],
            "avoid_families": ["svm"], "allowed_model_families": ["tree_ensemble", "linear"], "notes": "keep it small"}
TRY = {"hypothesis": "interaction of x1 and x2", "constraints": ["no new libraries"]}


def started(tmp_path, *flags) -> World:
    world = World(tmp_path)
    assert world.run("start", *flags)[0] == 10
    return world


def prompts_of(world, kind, since=0):
    return [p for k, p in zip(world.claude.kinds[since:], world.claude.prompts[since:]) if k == kind]


# 38
def test_steering_reaches_the_three_roles_as_typed_input_and_a_clear_removes_it(tmp_path):
    world = started(tmp_path, "--supervised")
    decide(world, "steering", steering=STEERING, clear=False)
    mark = len(world.claude.prompts)
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    for kind in ("coordinator", "feature_engineering", "modeling"):
        for prompt in prompts_of(world, kind, mark):
            assert '"steering"' in prompt and '"focus": "feature_engineering"' in prompt, kind
            assert '"pin_features": ["x1"]' in prompt and '"allowed_model_families": ["tree_ensemble", "linear"]' in prompt
            assert STEERING_NOTE in prompt
    assert not any('"steering"' in p for p in prompts_of(world, "reviewer", mark) + prompts_of(world, "data", mark))
    assert [m["outcome"] for m in applied(world) if m["kind"] == "steering"] == ["applied"]
    decide(world, "steering", clear=True, steering={})
    mark = len(world.claude.prompts)
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume", "--supervised")[0] == 10
    after = [p for kind in ("coordinator", "feature_engineering", "modeling") for p in prompts_of(world, kind, mark)]
    assert after and not any('"steering"' in p or STEERING_NOTE in p for p in after)
    outcomes = {m["decision"]: m["outcome"] for m in applied(world) if m["kind"] == "steering"}
    assert list(outcomes.values()) == ["applied", "applied"]  # each reached a role before the next replaced it


# 52
def test_steering_applies_to_a_run_that_is_not_supervised(tmp_path):
    world = started(tmp_path)
    decide(world, "steering", steering=STEERING, clear=False)
    world.lead(PLAN_APPROVE)
    assert world.run("resume")[0] == 10
    assert any('"steering"' in p for p in prompts_of(world, "modeling"))
    assert [m["outcome"] for m in applied(world)] == ["applied"]


# 40
def test_try_this_binds_to_the_first_proposal_of_the_round_and_the_next_round_has_none(tmp_path):
    world = started(tmp_path, "--supervised")
    record = decide(world, "try_this", request=TRY, from_round=1)
    mark = len(world.claude.prompts)
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    bound = [m for m in applied(world) if m["kind"] == "try_this"]
    assert len(bound) == 1 and bound[0]["decision"] == record["id"] and bound[0]["bound_to"] == "toy-1-R1-c1"
    carrying = [(k, p) for k, p in zip(world.claude.kinds[mark:], world.claude.prompts[mark:]) if REQUEST_NOTE in p]
    assert [k for k, _ in carrying] == ["feature_engineering", "modeling"]
    assert all('"hypothesis": "interaction of x1 and x2"' in p for _, p in carrying)
    mark = len(world.claude.prompts)
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume", "--supervised")[0] == 10
    assert not any(REQUEST_NOTE in p for p in world.claude.prompts[mark:])  # round 2 has no request
    assert len([m for m in applied(world) if m["kind"] == "try_this"]) == 1
    # the candidate was evaluated by the same engine: its attempt sits in the same cycle and protocol
    assert attempts(world) == ["toy-1-R0-c1", "toy-1-R1-c1", "toy-1-R2-c1"]


# 47
def test_a_crash_between_the_binding_and_the_effect_reuses_the_same_candidate_on_resume(tmp_path):
    world = started(tmp_path, "--supervised")
    record = decide(world, "try_this", request=TRY, from_round=1)
    world.claude.errors[8] = RuntimeError("the machine went to sleep")  # the modeling call of round 1
    world.lead(PLAN_APPROVE)
    code, _ = world.run("resume", "--supervised")
    assert code == 11  # suspended: the binding was written BEFORE the failing call
    assert [(m["decision"], m["bound_to"]) for m in applied(world)] == [(record["id"], "toy-1-R1-c1")]
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    assert [(m["decision"], m["bound_to"]) for m in applied(world)] == [(record["id"], "toy-1-R1-c1")]
    assert attempts(world) == ["toy-1-R0-c1", "toy-1-R1-c1"]
    final = [p for k, p in zip(world.claude.kinds, world.claude.prompts) if k == "modeling" and REQUEST_NOTE in p]
    assert final and all('"hypothesis": "interaction of x1 and x2"' in p for p in final)


# 48
def test_a_request_for_a_later_round_waits_and_is_not_applied_early(tmp_path):
    world = started(tmp_path, "--supervised")
    decide(world, "try_this", request=TRY, from_round=2)
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    assert applied(world) == []  # round 1 is before from_round: nothing bound, nothing stale
    assert not any(REQUEST_NOTE in p for p in world.claude.prompts)
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume", "--supervised")[0] == 10
    assert [m["bound_to"] for m in applied(world)] == ["toy-1-R2-c1"]


# 48
def test_a_steering_that_a_later_one_replaced_before_any_role_read_it_is_marked_stale(tmp_path):
    world = started(tmp_path, "--supervised")
    first = decide(world, "steering", steering=STEERING, clear=False)
    second = decide(world, "steering", steering={**STEERING, "focus": "model_tuning"}, clear=False)
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    outcomes = {m["decision"]: m for m in applied(world)}
    assert outcomes[first["id"]]["outcome"] == "stale" and "superseded" in outcomes[first["id"]]["reason"]
    assert outcomes[second["id"]]["outcome"] == "applied"
    assert any('"focus": "model_tuning"' in p for p in prompts_of(world, "modeling"))
    assert not any('"focus": "feature_engineering"' in p for p in prompts_of(world, "modeling"))
    import json
    stale = [json.loads(line) for line in (world.project / "cockpit" / "events.jsonl").read_text().splitlines()]
    assert [e["data"]["decision"] for e in stale if e["type"] == "decision_stale"] == [first["id"]]
