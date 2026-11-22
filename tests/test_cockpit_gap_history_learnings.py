"""The leaderboard gap, the run history, the learnings notebook and unused requests (cases 22, 23, 29, 30, 41)."""

from cockpit_finish_helpers import FinishWorld
from orchestrator_helpers import SLUG, competition
from overrides_helpers import FIRST, SECOND, decide, fork

TOKEN_HEADERS = ("Origin", "X-CSRF-Token")


def send(fw):
    sha = fw.wait_job(fw.ok("/api/submission/preview")["job"])["result"]["data"]["sha256"]
    return fw.wait_job(fw.ok("/api/submission/confirm", {"sha256": sha, "confirm": True, "typed": "SEND"})["job"])


# 22
def test_the_gap_view_compares_public_cv_and_holdout_and_explains_empty_and_pending_states(tmp_path):
    fw = FinishWorld(tmp_path, comp=competition(score_script=(None,)))
    empty = fw.get_json("/api/runs/toy-1/gap")
    assert empty["rows"] == [] and "No submission yet" in empty["state_text"]
    assert "small" in empty["explanation"] and "overfit" in empty["explanation"] and "CV optimism" in empty["explanation"]
    assert send(fw)["result"]["code"] == 23
    [row] = fw.get_json("/api/runs/toy-1/gap")["rows"]
    assert row["state"] == "pending" and row["public"] is None and row["gap_cv"] is None
    fw.world.kaggle.competitions[SLUG] = competition(score_script=(None, 0.7))
    send(fw)
    [row] = fw.get_json("/api/runs/toy-1/gap")["rows"]
    assert row["state"] == "scored" and row["public"] == 0.7 and abs(row["gap_cv"] - (0.7 - 0.7291666666666666)) < 1e-9
    assert row["holdout"] == "sealed" and row["gap_holdout"] is None
    for body in (fw.client.get("/api/runs/toy-1/gap").get_data(as_text=True), fw.page("/cycles/toy-1/gap")):
        assert "0.4167" not in body and "0.41666" not in body
    fw.ok("/api/holdout/unseal", {"kind": "first", "confirm": True})
    [row] = fw.get_json("/api/runs/toy-1/gap")["rows"]
    assert abs(row["holdout"] - 0.4166666666666667) < 1e-9 and abs(row["gap_holdout"] - (0.7 - row["holdout"])) < 1e-9


# 23
def test_the_history_lists_what_each_run_tried_and_learned_with_origin_protocol_and_labels(tmp_path):
    fw = FinishWorld(tmp_path)
    fw.ok("/api/notes", {"target": {"type": "run", "id": "toy-1"}, "text": "the id column is useless"})
    assert fork(fw.world, "toy-1-R1-c1", "r-history", "--threshold-k", "0.5", "--reason", "looser")[0] == 10
    runs = {r["id"]: r for r in fw.get_json("/api/history")}
    assert set(runs) == {FIRST, SECOND}
    first = runs[FIRST]
    assert [t["hypothesis"] for t in first["tried"]] and all(t["outcome"] == "succeeded" for t in first["tried"])
    assert first["learned"]["reviewer"] and first["learned"]["suggestions"] and first["learned"]["guidance"] == []
    assert first["learned"]["notes"] == ["the id column is useless"] and first["origin"] is None
    second = runs[SECOND]
    assert second["origin"] == {"cycle_id": FIRST, "attempt_id": "toy-1-R1-c1"} and second["protocol_label"] == "changed"
    assert "new protocol (set by the DS)" in [x["label"] for x in second["labels"]]
    page = fw.page("/history")
    assert "toy-1" in page and "toy-2" in page and "looser" in page.lower() or "new protocol" in page


# 29
def test_the_notebook_searches_notes_tags_and_suggestions_case_insensitively_and_escapes(tmp_path):
    fw = FinishWorld(tmp_path)
    fw.ok("/api/notes", {"target": {"type": "competition", "id": SLUG}, "text": "Beware the LEAKY id column <b>x</b>"})
    fw.ok("/api/notes", {"target": {"type": "attempt", "id": "toy-1-R1-c1"}, "text": "tree depth hurt"})
    fw.ok("/api/tags", {"target": {"type": "run", "id": "toy-1"}, "tags": ["Leakage", "baseline"]})
    everything = fw.get_json("/api/learnings")
    assert everything["total"] >= 4 and everything["q"] == ""
    found = fw.get_json("/api/learnings?q=leaky")
    assert [i["source"] for i in found["items"]] == ["note"] and found["items"][0]["target_type"] == "competition"
    assert fw.get_json("/api/learnings?q=LEAKAGE")["items"][0]["tags"] == ["Leakage", "baseline"]
    assert any(i["source"] == "suggestion" for i in fw.get_json("/api/learnings?q=rounds")["items"])
    assert fw.get_json("/api/learnings?q=nothing-matches-this")["items"] == []
    page = fw.page("/learnings?q=leaky")
    assert "<b>x</b>" not in page
    paged = fw.get_json("/api/learnings?limit=1&offset=1")
    assert len(paged["items"]) == 1 and paged["limit"] == 1 and paged["offset"] == 1


# 30
def test_competition_notes_are_guarded_and_audited_and_the_notebook_shows_the_change(tmp_path):
    fw = FinishWorld(tmp_path)
    target = {"type": "competition", "id": SLUG}
    assert fw.post("/api/notes", {"target": target, "text": "x"}, token=False).status_code == 403
    assert fw.post("/api/notes", {"target": {"type": "competition", "id": "other"}, "text": "x"}).status_code == 400
    assert fw.lines("notes.jsonl") == [] and fw.lines("audit.jsonl") == []
    note = fw.ok("/api/notes", {"target": target, "text": "first idea"})["note"]
    fw.ok("/api/notes", {"target": target, "op": "edit", "note_id": note, "text": "second idea"})
    assert [i["text"] for i in fw.get_json("/api/learnings?q=idea")["items"]] == ["second idea"]
    fw.ok("/api/notes", {"target": target, "op": "delete", "note_id": note})
    assert fw.get_json("/api/learnings?q=idea")["items"] == []
    assert [a["what"] for a in fw.lines("audit.jsonl")] == ["note.add", "note.edit", "note.delete"]
    assert len(fw.lines("notes.jsonl")) == 3


# 41
def test_a_try_this_that_never_ran_shows_not_used_and_nothing_was_applied(tmp_path):
    fw = FinishWorld(tmp_path)
    assert fw.post("/api/decisions/try-this", {"request": {"hypothesis": "interaction of x1 and x2"}}
                   ).status_code == 409  # the cycle is stopped: nothing would run it
    decide(fw.world, "try_this", request={"hypothesis": "interaction of x1 and x2", "constraints": []}, from_round=3)
    [item] = fw.get_json("/api/overrides/pending")["overrides"]
    assert item["kind"] == "try_this" and item["status"] == "not used" and "stopped" in item["reason"]
    assert fw.lines("decisions-applied.jsonl") == [] and "not used" in fw.page("/cycles/toy-1/rounds/toy-1-R2")
