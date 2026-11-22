"""The cockpit over a registry written by the existing fake end-to-end run (no network)."""

import json

from cockpit_helpers import api, client_for, loopback_only, page, sha256_of  # noqa: F401  (fixture)
from orchestrator_helpers import World, drive_to_stopped

from kaggle_agent.registry import SqliteRegistry


# 30
def test_every_page_and_endpoint_answers_over_a_registry_from_a_fake_end_to_end_run(tmp_path):
    world = World(tmp_path)
    drive_to_stopped(world)
    db = world.project / "registry.db"
    assert (world.project / "project.json").is_file()
    before = sha256_of(db)
    with SqliteRegistry(db, read_only=True) as reg:
        cycle = reg.list_cycles()[0]
        attempts = reg.list_attempts(cycle.id)
        outcome = reg.get_holdout_outcome(cycle.id)
        report = reg.get_final_report(cycle.id)
        scores = [p.mean for p in reg.metric_series(cycle.id, "validation")]
        rounds = reg.list_rounds_with_candidates(cycle.id)
        round_id = rounds[-1].round.id
        attempt_id = attempts[0].attempt_id
    assert attempts and scores and outcome is not None and report is not None
    client = client_for(db, world.project)
    pages = ["/", f"/cycles/{cycle.id}", f"/cycles/{cycle.id}/rounds", f"/cycles/{cycle.id}/rounds/{round_id}",
             f"/cycles/{cycle.id}/attempts", f"/cycles/{cycle.id}/traces", f"/cycles/{cycle.id}/report",
             f"/cycles/{cycle.id}/guidance", f"/attempts/{attempt_id}", "/suggestions"]
    for url in pages:
        page(client, url)
    base = f"/api/runs/{cycle.id}"
    for suffix in ("", "/chart", "/rounds", "/attempts", "/executions", "/comparisons", "/reviewer-reports",
                   "/guidance", "/submissions", "/holdout", "/final-report", "/bundles", "/llm-calls",
                   "/role-traces"):
        body = api(client, base + suffix)
        assert body["api_version"] == 1 and body["warnings"] == [], suffix
    run = api(client, base)["data"]
    protocol = json.loads((world.project / "project.json").read_text(encoding="utf-8"))
    assert run["state"] == "STOPPED" and run["status"] in ("succeeded", "failed")
    assert run["attempts_total"] == len(attempts) and run["rounds_planned"] == cycle.iterations
    assert run["best"]["mean"] == (max(scores) if protocol["direction"] == "maximize" else min(scores))
    assert run["protocol"]["direction_source"] == "project.json"
    assert run["protocol"]["metric"] == protocol["metric"]
    assert run["protocol"]["hash"] == cycle.protocol_hash
    assert run["stop_reason"] == report.content["stop_reason"]
    assert run["chosen"]["attempt_id"] == report.content["chosen"]["attempt_id"]
    holdout = api(client, base + "/holdout")["data"]
    assert holdout["state"] == "ready" and holdout["status"] == "sealed" and holdout["score"] is None  # ticket 30
    cockpit = world.project / "cockpit"
    cockpit.mkdir(exist_ok=True)
    (cockpit / "holdout-reads.jsonl").write_text(json.dumps({"id": "u1", "cycle": cycle.id, "kind": "first"}) + "\n",
                                                 encoding="utf-8")
    holdout = api(client, base + "/holdout")["data"]
    assert holdout["status"] == outcome.status and holdout["score"] == outcome.score
    chart = api(client, base + "/chart")["data"]
    assert len(chart["points"]) == len(scores) and chart["baseline"] is not None
    assert sha256_of(db) == before
