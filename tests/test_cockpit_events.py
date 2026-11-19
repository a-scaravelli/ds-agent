"""The event log: ids, types and payloads, fold progress without changing results, no prompts or secrets."""

import json
import re

from cockpit_live_helpers import drive_supervised, live, read_events  # noqa: F401  (fixtures)
from orchestrator_helpers import World, drive_to_stopped

from kaggle_agent.orchestrator.cockpit import read_jsonl
from kaggle_agent.orchestrator.events import ALLOWED, clean_payload
from kaggle_agent.sandbox.protected import hash_path


# 1
def test_the_event_log_is_append_only_with_monotonic_ids_across_processes_and_a_torn_tail(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    first = live.events()
    assert [e["id"] for e in first] == list(range(1, len(first) + 1))
    log = live.cockpit_file("events.jsonl")
    with open(log, "ab") as handle:  # a crash in the middle of a line
        handle.write(b'{"id": 999, "type": "stage_st')
    live.decide_plan()
    live.ok("/api/run/resume")  # a NEW orchestrator process continues the ids
    live.wait_exit(10)
    records, skipped = read_jsonl(log)
    ids = [r["id"] for r in records]
    assert skipped == 1 and ids == list(range(1, len(ids) + 1)) and len(ids) > len(first)
    assert live.events()[-1]["id"] == ids[-1]  # the reader ignores the torn line


# 2
def test_event_types_and_payloads_cover_the_whole_cycle(tmp_path):
    world = World(tmp_path)
    drive_supervised(world)
    events = read_events(world.project)
    types = {e["type"] for e in events}
    assert {"run_start", "cycle_state", "stage_start", "stage_end", "round_start", "candidate_start",
            "candidate_end", "llm_call", "waiting", "finished", "fold_start", "fold_end"} <= types
    assert "decision_applied" not in types  # these answers come from lead.toml, not from the page
    for event in events:
        assert set(event) == {"id", "at", "type", "cycle", "data"} and event["at"].endswith("+00:00")
        assert event["cycle"] == "toy-1" and len(json.dumps(event["data"])) <= 2048
    roles = {e["data"].get("role") for e in events if e["type"] == "stage_start"}
    assert {"coordinator", "data", "feature_engineering", "modeling", "reviewer", "writer"} <= roles
    call = next(e for e in events if e["type"] == "llm_call")
    assert set(call["data"]) >= {"role", "outcome", "input_tokens", "output_tokens", "seconds"}
    assert any(e["type"] == "candidate_end" and e["data"]["outcome"] == "succeeded" for e in events)
    assert next(e for e in events if e["type"] == "waiting")["data"]["question"].startswith("plan approval")
    assert events[-1]["type"] == "finished"
    assert {e["type"] for e in events} <= set(ALLOWED)


def _dump(world):
    registry = world.registry()
    try:
        cycle = registry.list_cycles()[0]
        rows = []
        for summary in registry.list_attempts(cycle.id):
            detail = registry.get_attempt(summary.attempt_id)
            rows.append((summary.attempt_id, summary.outcome,
                         [(m.partition, m.mean, m.dispersion) for e in detail.executions for m in e.metrics]))
        comparisons = [(c.candidate_attempt_id, c.difference, c.threshold, c.is_real)
                       for c in registry.list_comparisons(cycle.id)]
        guidance = [(g.round_id, g.text) for g in registry.list_guidance(cycle.id)]
        return cycle.state, cycle.protocol_hash, rows, comparisons, guidance
    finally:
        registry.close()


# 3
def test_fold_progress_by_wrapping_the_candidate_leaves_the_results_unchanged(tmp_path):
    from orchestrator_helpers import World as W

    plain, watched = W(tmp_path / "plain"), W(tmp_path / "watched")
    drive_to_stopped(plain)
    drive_supervised(watched)
    assert _dump(plain) == _dump(watched)
    events = read_events(watched.project)
    ends = [e["data"] for e in events if e["type"] == "fold_end"]
    assert ends and all(d["folds"] == 3 and 1 <= d["fold"] <= 3 for d in ends)
    assert all(0.0 <= d["score"] <= 1.0 for d in ends)  # the partial score of the validation fold only
    assert len([e for e in events if e["type"] == "fold_start"]) == len(ends)
    assert not (watched.project / "cockpit" / "events.jsonl").read_text().count("holdout")
    from kaggle_agent.orchestrator import lead_file  # noqa: F401
    import test_orchestrator_guards as guards

    assert hash_path(guards.SRC / "evaluation") == guards.PINNED["evaluation"]


# 4
def test_events_never_contain_prompts_or_secrets_and_the_console_tail_is_redacted(live, monkeypatch):
    secret = "kgat_" + "a1B2c3D4" * 5
    monkeypatch.setenv("KAGGLE_API_TOKEN", secret)
    live.ok("/api/run/start")
    live.wait_exit(10)
    live.decide_plan()
    live.ok("/api/run/resume")
    live.wait_exit(10)
    text = live.cockpit_file("events.jsonl").read_text(encoding="utf-8")
    assert secret not in text and secret not in live.console()
    for word in ("You are the", "hypothesis ", "round summary", "a small table", "Bundle id:", "majority class"):
        assert word not in text, word  # no prompt, response or hypothesis text
    with open(live.cockpit_file("console.log"), "a", encoding="utf-8") as handle:
        handle.write(f"debug token={secret}\nAuthorization: Bearer {secret}\n")
    tail = live.api("/api/console?tail=2048")["text"]
    assert secret not in tail and "[redacted]" in tail
    for event in live.events():  # every payload is scalars only
        assert all(isinstance(v, (str, int, float, bool)) for v in event["data"].values())
    assert clean_payload("llm_call", {"role": "x", "prompt": "SECRET", "outcome": "ok"}) == {"role": "x", "outcome": "ok"}
    assert len(json.dumps(clean_payload("error", {"kind": "x" * 5000}))) < 400
