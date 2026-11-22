"""The holdout panel: sealed until unsealed, the early and second read, the guard (cases 1 to 5)."""

import hashlib

from cockpit_finish_helpers import GUIDE_CONTINUE, PLAN_APPROVE, FinishWorld, iterating_world

SCORE_TEXTS = ("0.41666", "0.4167", "12 rows")


def surfaces(fw: FinishWorld) -> dict[str, str]:
    return {
        "api holdout": fw.client.get("/api/runs/toy-1/holdout").get_data(as_text=True),
        "api report": fw.client.get("/api/runs/toy-1/final-report").get_data(as_text=True),
        "api submissions": fw.client.get("/api/runs/toy-1/submissions/preview").get_data(as_text=True),
        "api gap": fw.client.get("/api/runs/toy-1/gap").get_data(as_text=True),
        "page": fw.page("/cycles/toy-1/report"),
        "overview": fw.page("/cycles/toy-1"),
        "print": fw.page("/cycles/toy-1/report/print"),
        "markdown": fw.client.get("/api/runs/toy-1/report.md").get_data(as_text=True),
    }


def unseal(fw, kind="first", **body):
    return fw.post("/api/holdout/unseal", {"kind": kind, "confirm": True, **body})


# 1
def test_a_sealed_holdout_shows_a_plain_status_and_no_number_anywhere(tmp_path):
    fw = FinishWorld(tmp_path)
    registry_before = fw.registry_hash()
    # the pipeline did record an outcome: the registry holds it, the cockpit does not show it
    assert fw.report()["holdout"]["score"] is not None
    view = fw.get_json("/api/runs/toy-1/holdout")
    assert view["status"] == "sealed" and view["state"] == "ready" and view["score"] is None and view["rows"] is None
    assert "sealed" in view["text"] and "ONE holdout session" in view["text"]
    for name, body in surfaces(fw).items():
        for text in SCORE_TEXTS:
            assert text not in body, (name, text)
    assert "sealed" in fw.page("/cycles/toy-1/report").lower()
    assert "attachment" in fw.client.get("/api/runs/toy-1/report.md").headers["Content-Disposition"]
    assert fw.registry_hash() == registry_before and fw.lines("holdout-reads.jsonl") == []


# 2
def test_unseal_and_read_at_the_review_needs_no_reason_and_shows_the_number(tmp_path):
    fw = FinishWorld(tmp_path)
    registry_before = fw.registry_hash()
    data = fw.ok("/api/holdout/unseal", {"kind": "first", "confirm": True})
    assert data["kind"] == "first" and data["label"] is None and data["duplicate"] is False
    view = fw.get_json("/api/runs/toy-1/holdout")
    assert view["state"] == "unsealed" and view["status"] == "succeeded" and view["rows"] == 12
    assert abs(view["score"] - 0.4166666666666667) < 1e-9
    assert "0.4167" in fw.page("/cycles/toy-1/report") or "0.41667" in fw.page("/cycles/toy-1/report")
    assert "0.4167" in fw.client.get("/api/runs/toy-1/report.md").get_data(as_text=True)
    [read] = fw.lines("holdout-reads.jsonl")
    assert read["kind"] == "first" and read["cycle"] == "toy-1" and read["reason"] == ""
    assert [a["what"] for a in fw.lines("audit.jsonl")] == ["holdout.first"]
    assert fw.get_json("/api/runs/toy-1/labels")["labels"] == []  # a first read is not a re-read
    assert fw.registry_hash() == registry_before


# 3
def test_an_early_read_warns_wants_a_reason_is_logged_finishes_the_run_and_labels_it(tmp_path):
    fw = iterating_world(tmp_path)
    response = unseal(fw, "early")
    assert response.status_code == 400
    message = response.get_json()["error"]["message"]
    assert "reason" in message and "before the cycle ended" in message and "re-read" in message
    assert fw.lines("holdout-reads.jsonl") == [] and fw.lines("audit.jsonl") == []
    data = fw.ok("/api/holdout/unseal", {"kind": "early", "confirm": True, "reason": "the client wants a number"})
    assert data["label"] == "holdout re-read" and data["finish_now"]
    [change] = fw.lines("limit-changes.jsonl")
    assert change["values"] == {"rounds": 1} and change["reason"] == "the client wants a number"
    assert [r["kind"] for r in fw.lines("holdout-reads.jsonl")] == ["early"]
    view = fw.get_json("/api/runs/toy-1/holdout")
    assert view["state"] == "unsealed" and view["score"] is None  # nothing recorded yet: the number comes at the report
    fw.world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    assert fw.world.run("resume", "--supervised")[0] == 10 and fw.world.state() == "AWAITING_REVIEW"
    view = fw.get_json("/api/runs/toy-1/holdout")
    assert view["score"] is not None and view["rows"] == 12
    labels = fw.get_json("/api/runs/toy-1/labels")["labels"]
    assert [x["label"] for x in labels] == ["holdout re-read"] and labels[0]["reason"] == "the client wants a number"
    assert "holdout re-read" in fw.page("/cycles/toy-1/report")


# 4
def test_a_second_read_is_labelled_and_logged_and_the_registry_file_is_identical(tmp_path):
    fw = FinishWorld(tmp_path)
    fw.ok("/api/holdout/unseal", {"kind": "first", "confirm": True})
    before = fw.registry_hash()
    response = unseal(fw, "second")
    assert response.status_code == 400 and "reason" in response.get_json()["error"]["message"]
    response = unseal(fw, "first")  # the situation changed: it is a second read now
    assert response.status_code == 409
    data = fw.ok("/api/holdout/unseal", {"kind": "second", "confirm": True, "reason": "check the split again"})
    assert data["label"] == "holdout re-read" and data["finish_now"] is None
    assert [r["kind"] for r in fw.lines("holdout-reads.jsonl")] == ["first", "second"]
    assert [a["what"] for a in fw.lines("audit.jsonl")] == ["holdout.first", "holdout.second"]
    assert [x["label"] for x in fw.get_json("/api/runs/toy-1/labels")["labels"]] == ["holdout re-read"]
    assert fw.registry_hash() == before
    # the same request_id twice is one read
    again = fw.ok("/api/holdout/unseal", {"kind": "second", "confirm": True, "reason": "x", "request_id": "same-request-1"})
    dup = fw.ok("/api/holdout/unseal", {"kind": "second", "confirm": True, "reason": "x", "request_id": "same-request-1"})
    assert dup["duplicate"] is True and dup["read"] == again["read"] and len(fw.lines("holdout-reads.jsonl")) == 3


# 5
def test_unseal_is_a_guarded_write_without_a_token_or_with_a_foreign_origin(tmp_path):
    fw = FinishWorld(tmp_path)
    before = fw.registry_hash()
    body = {"kind": "first", "confirm": True}
    assert fw.post("/api/holdout/unseal", body, token=False).status_code == 403
    assert fw.post("/api/holdout/unseal", body, token="not-the-token").status_code == 403
    assert fw.post("/api/holdout/unseal", body, origin="http://evil.example").status_code == 403
    assert fw.client.get("/api/holdout/unseal").status_code == 405
    assert fw.lines("audit.jsonl") == [] and fw.lines("holdout-reads.jsonl") == []
    assert fw.get_json("/api/runs/toy-1/holdout")["state"] == "ready" and fw.registry_hash() == before
    assert unseal(fw, "first", confirm=False).status_code == 400  # confirm is required: nothing was logged
    assert fw.lines("holdout-reads.jsonl") == [] and hashlib.sha256(b"").hexdigest()
