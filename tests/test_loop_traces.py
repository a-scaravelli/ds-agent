"""Role traces carry the candidate id, and everything external is injected (cases 28, 29)."""

from pathlib import Path

from loop_helpers import CYCLE, RoleClient, make_setup

from kaggle_agent.llm.fake import Malformed
from kaggle_agent.review import LeadResponse, SimulatedLead


def test_candidate_roles_carry_candidate_id_and_every_trace_links_to_a_call(tmp_path):
    client = RoleClient(overrides={"feature_engineering": [Malformed()]})  # the first FE reply is rejected
    setup = make_setup(tmp_path, rounds=1, candidates=2, stagnation_k=5, client=client)
    setup.run()
    reg = setup.registry
    traces = reg.list_role_traces(CYCLE)
    call_count = len(reg.list_llm_calls(CYCLE))
    for trace in traces:
        assert trace.llm_call_id is not None and 1 <= trace.llm_call_id <= call_count
        if trace.role in ("feature_engineering", "modeling") and trace.output is not None:
            assert trace.candidate_id == trace.output["id"]
            assert trace.candidate_id in {a.attempt_id for a in reg.list_attempts(CYCLE)}
            assert trace.round_id == trace.candidate_id.rsplit("-c", 1)[0]
        else:
            assert trace.candidate_id is None
    ok = [t for t in traces if t.outcome == "ok"]
    assert sorted({(t.role, t.candidate_id) for t in ok if t.candidate_id}) == sorted(
        {(role, f"C1-{r}-c{c}") for role in ("feature_engineering", "modeling")
         for r, c in (("R0", 1), ("R1", 1), ("R1", 2))}
    )
    assert {t.role for t in ok if t.candidate_id is None} == {"coordinator", "data", "reviewer"}
    rejected = [t for t in traces if t.outcome == "invalid_output"]
    assert len(rejected) == 1 and rejected[0].role == "feature_engineering"
    assert rejected[0].candidate_id is None and rejected[0].output is None  # traced, with no candidate
    assert [t.role for t in traces if t.role == "reviewer"] == ["reviewer"]


def test_the_whole_cycle_runs_on_injected_fakes_only(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an external resource was used")

    monkeypatch.setattr("subprocess.Popen", forbidden)  # no real process
    monkeypatch.setattr("socket.socket", forbidden)  # no network
    lead = SimulatedLead(guidance=[LeadResponse("guidance", "try trees")])
    setup = make_setup(tmp_path, rounds=2, candidates=2, stagnation_k=5, lead=lead)
    assert setup.run().status == "stopped"
    reg = setup.registry
    stamps = [t.created_at for t in reg.list_role_traces(CYCLE)]
    stamps += [r.created_at for r in reg.list_reviewer_reports(CYCLE)]
    stamps += [g.created_at for g in reg.list_guidance(CYCLE)]
    assert stamps and all(s.startswith("2026-11-01T09:") for s in stamps)  # from the manual clock
    assert all(Path(b.reference).is_relative_to(tmp_path) for b in reg.list_bundles(CYCLE))


def test_the_new_tests_avoid_platform_specific_paths():
    forbidden = ("/proc", "/tmp", "RealClaudeClient", "SubprocessSandbox", "mkdtemp")
    this = Path(__file__).name
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name == this or not path.name.startswith(
            ("test_loop_", "loop_helpers", "test_budget_", "test_review_")
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in text, (path.name, word)
