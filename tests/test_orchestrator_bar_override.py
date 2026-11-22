"""The bar override: overlay on the comparisons, stale records, cycle rule, every run (cases 34, 35, 36, 47, 48, 52)."""

import json
from types import SimpleNamespace

from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, World
from overrides_helpers import FIRST, applied, attempts, decide, files, registry

from kaggle_agent.loop import CycleView
from kaggle_agent.orchestrator import overrides_registry
from kaggle_agent.orchestrator.overrides_registry import OverridingRegistry
from kaggle_agent.orchestrator.project import read_project
from kaggle_agent.orchestrator.config import load_config
from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.registry import ComparisonRecord

R1 = "toy-1-R1-c1"


def after_round(tmp_path, rounds=1) -> World:
    world = World(tmp_path)
    assert world.run("start", "--supervised")[0] == 10
    world.lead(PLAN_APPROVE)
    assert world.run("resume", "--supervised")[0] == 10
    for n in range(1, rounds):
        world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=n))
        assert world.run("resume", "--supervised")[0] == 10
    return world


def view_over(world, overlay: bool):
    reg = registry(world)
    config = load_config(str(world.config))
    state = read_project(config.paths)
    engine = EvaluationEngine(reg, state.cycle_id, Protocol.from_dict(state.protocol))
    inner = OverridingRegistry(reg, files(world), None, FIRST, FIRST) if overlay else reg
    return CycleView(inner, state.cycle_id, engine), reg, inner


def events(world) -> list[dict]:
    path = world.project / "cockpit" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# 34
def test_a_round_override_moves_the_derived_reference_and_leaves_the_stored_row_alone(tmp_path):
    world = after_round(tmp_path)
    plain, reg, _ = view_over(world, overlay=False)
    state = plain.round_states()[0]
    assert state.complete and state.improved is False and state.comparison.is_real is False
    stored = (state.comparison.difference, state.comparison.threshold, state.comparison.is_real)
    assert plain.reference().attempt.id == "toy-1-R0-c1"
    reg.close()
    decide(world, "bar_override", scope="round", round=1, attempt=R1, reason="the gain is real on fold 2")
    over, reg, inner = view_over(world, overlay=True)
    state = over.round_states()[0]
    assert state.improved is True and over.reference().attempt.id == R1
    assert (state.comparison.difference, state.comparison.threshold) == stored[:2] and state.comparison.is_real
    raw = reg.list_comparisons(FIRST)  # what the file stores is unchanged
    assert [(c.difference, c.threshold, c.is_real) for c in raw if c.candidate_attempt_id == R1] == [stored]
    assert [m["outcome"] for m in applied(world) if m["kind"] == "bar_override"] == ["applied"]
    reg.close()


# 52
def test_a_run_without_supervised_honours_the_override_and_writes_the_marker(tmp_path):
    world = World(tmp_path)
    assert world.run("start")[0] == 10
    world.lead(PLAN_APPROVE)
    assert world.run("resume")[0] == 10
    decide(world, "bar_override", scope="round", round=1, attempt=R1, reason="because")
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume")[0] == 10  # no --supervised: no events, no heartbeat
    assert not (world.project / "cockpit" / "events.jsonl").exists()
    marks = [m for m in applied(world) if m["kind"] == "bar_override"]
    assert len(marks) == 1 and marks[0]["outcome"] == "applied"
    view, reg, _ = view_over(world, overlay=True)
    assert view.round_states()[0].improved is True
    reg.close()


# 35 48
def test_a_hand_made_override_for_an_old_round_is_marked_stale_once_and_ignored(tmp_path):
    world = after_round(tmp_path, rounds=2)
    old = decide(world, "bar_override", scope="round", round=1, attempt=R1, reason="too late")
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    world.run("resume", "--supervised")
    world.run("resume", "--supervised")
    marks = [m for m in applied(world) if m["decision"] == old["id"]]
    assert len(marks) == 1 and marks[0]["outcome"] == "stale" and "round 1" in marks[0]["reason"]
    stale = [e for e in events(world) if e["type"] == "decision_stale"]
    assert len(stale) == 1 and stale[0]["data"]["decision"] == old["id"]
    view, reg, _ = view_over(world, overlay=True)
    assert all(not s.improved for s in view.round_states())
    reg.close()


# 36
def test_a_cycle_rule_counts_positive_sub_bar_rounds_from_round_k_and_ends_with_the_cycle(monkeypatch):
    class Inner:
        def __init__(self, rounds, comparisons):
            self.rounds, self.comparisons = rounds, comparisons

        def list_comparisons(self, cycle_id):
            return self.comparisons

        def list_rounds_with_candidates(self, cycle_id):
            return self.rounds

        def list_reviewer_reports(self, cycle_id):
            return [SimpleNamespace(round_id=r.round.id) for r in self.rounds]

    def rnd(n, attempt):
        return SimpleNamespace(round=SimpleNamespace(id=f"c-R{n}"), candidates=[SimpleNamespace(attempt_id=attempt)])

    comparisons = [ComparisonRecord("c", "a1", "a0", 0.01, 0.05, False), ComparisonRecord("c", "a2", "a0", 0.02, 0.05, False),
                   ComparisonRecord("c", "a3", "a0", -0.02, 0.05, False)]
    records = [{"id": "d1", "kind": "bar_override", "scope": "cycle", "round": 2, "reason": "the bar is too strict"}]
    marked: dict = {}
    monkeypatch.setattr(overrides_registry, "read_jsonl", lambda path: (records, 0))
    monkeypatch.setattr(overrides_registry, "mark", lambda files, emitter, record, outcome, reason="": marked.setdefault(record["id"], outcome))
    monkeypatch.setattr(overrides_registry, "applied_outcomes", lambda files: dict(marked))
    registry = lambda *rounds: OverridingRegistry(  # noqa: E731
        Inner(list(rounds), comparisons[:len(rounds)]), SimpleNamespace(decisions="x"), None, "c", "c")
    # given while round 2 is the last complete one: it counts round 2 (and rounds 1 stays as it was)
    out = registry(rnd(1, "a1"), rnd(2, "a2")).list_comparisons("c")
    assert [c.is_real for c in out] == [False, True]
    # later rounds: the rule is applied from round k on; a positive round counts, a negative one does not
    out = registry(rnd(1, "a1"), rnd(2, "a2"), rnd(3, "a3")).list_comparisons("c")
    assert [c.is_real for c in out] == [False, True, False]
    assert [(c.difference, c.threshold) for c in out] == [(0.01, 0.05), (0.02, 0.05), (-0.02, 0.05)]
    # the rule ends with the cycle: another cycle is not touched
    assert registry(rnd(1, "a1")).list_comparisons("other") == comparisons[:1]
    assert marked == {"d1": "applied"}


# 47
def test_a_decision_is_marked_once_and_a_resume_does_not_apply_it_again(tmp_path):
    world = after_round(tmp_path)
    decide(world, "bar_override", scope="round", round=1, attempt=R1, reason="because")
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert world.run("resume", "--supervised")[0] == 10
    # a second resume (the run was interrupted and resumed) re-derives the same state
    assert world.run("resume", "--supervised")[0] == 10
    assert [m["outcome"] for m in applied(world) if m["kind"] == "bar_override"] == ["applied"]
    assert len([e for e in events(world) if e["type"] == "decision_applied" and e["data"]["kind"] == "bar_override"]) == 1
    assert attempts(world) == ["toy-1-R0-c1", R1, "toy-1-R2-c1"]
