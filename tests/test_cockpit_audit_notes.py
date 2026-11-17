"""The audit log, notes and tags (single-writer files) and concurrent appends."""

import getpass
import json
import multiprocessing
import threading

from cockpit_live_helpers import live, live_factory, wait_for_stage  # noqa: F401  (fixtures)

from kaggle_agent.web import store

RUN = "toy-1"


def audit_entries(project):
    return project.api("/api/audit")["entries"]


def to_a_run_with_attempts(project):
    project.ok("/api/run/start")
    project.wait_exit(10)
    project.decide_plan()
    project.ok("/api/run/resume")
    project.wait_exit(10)
    return project.api(f"/api/runs/{RUN}/attempts")[0]["id"]


# 39
def test_every_ds_action_is_audited_before_it_takes_effect_and_merged_with_system_events(live):
    block = live.tmp / "block"
    live.set_scenario({"block_file": str(block)})
    block.write_text("x", encoding="utf-8")
    live.ok("/api/run/start")
    wait_for_stage(live)
    live.ok("/api/run/pause")
    block.unlink()
    live.wait_exit(10)
    live.decide_plan()
    block.write_text("x", encoding="utf-8")
    last = live.events()[-1]["id"]
    live.ok("/api/run/resume")
    wait_for_stage(live, last)
    live.ok("/api/run/stop", {"confirm": True, "reason": "testing the audit"})
    block.unlink()
    live.wait_exit(15)
    live.set_scenario({})
    live.ok("/api/run/resume")
    live.wait_exit(10)
    live.ok("/api/limits", {"values": {"rounds": 3}, "reason": "one more round", "confirm": True})
    live.ok("/api/notes", {"target": {"type": "run", "id": RUN}, "text": "a note"})
    live.ok("/api/settings", {"auto_resume": False})
    entries = audit_entries(live)
    mine = [e for e in entries if e["source"] == "ds"]
    names = [e["what"] for e in mine]
    for expected in ("run.start", "run.pause", "decision.plan_approve", "run.resume", "run.stop", "limits.extend",
                     "note.add", "settings.change"):
        assert expected in names, expected
    assert all(e["who"] == getpass.getuser() and e["at"] and e["id"] for e in mine)
    why = {e["what"]: e["why"] for e in mine}
    assert why["run.stop"] == "testing the audit" and why["limits.extend"] == "one more round"
    assert [e["at"] for e in entries] == sorted(e["at"] for e in entries)  # chronological
    system = {e["what"] for e in entries if e["source"] == "system"}
    assert {"run_start", "waiting", "paused"} & system or {"run_start", "waiting"} <= system
    page = live.get("/audit")[1]
    assert "run.stop" in page and "testing the audit" in page and "waiting" in page
    # written BEFORE it takes effect: the audit entry of the stop precedes the stop event of the run
    stop_audit = next(e for e in mine if e["what"] == "run.stop")
    stopped = next(e for e in live.events() if e["type"] == "stopped")
    assert stop_audit["at"] <= stopped["at"]


# 39
def test_an_action_is_refused_when_its_audit_entry_cannot_be_written(live):
    (live.dir / "cockpit" / "audit.jsonl").mkdir(parents=True)  # a directory: the append fails
    status, body = live.post("/api/settings", {"auto_resume": True})
    assert status == 503 and "audit" in body["error"]["message"].lower()
    assert not (live.dir / "cockpit" / "settings.json").exists()
    status, _ = live.post("/api/run/start")
    assert status == 503 and not (live.dir / "cockpit" / "run.json").exists()  # nothing was started


# 40
def test_notes_and_tags_on_a_run_and_an_attempt_are_shown_and_searchable(live):
    attempt = to_a_run_with_attempts(live)
    run_target, attempt_target = {"type": "run", "id": RUN}, {"type": "attempt", "id": attempt}
    note = live.ok("/api/notes", {"target": run_target, "text": "watch the validation folds"})["note"]
    live.ok("/api/notes", {"target": attempt_target, "text": "suspiciously fast"})
    live.ok("/api/tags", {"target": attempt_target, "tags": ["promising", "trees", "promising"]})
    assert live.ok("/api/tags", {"target": run_target, "tags": ["baseline-ok"]})["tags"] == ["baseline-ok"]
    assert "watch the validation folds" in live.get(f"/cycles/{RUN}")[1] and "baseline-ok" in live.get("/")[1]
    page = live.get(f"/attempts/{attempt}")[1]
    assert "suspiciously fast" in page and "promising" in page and "trees" in page
    assert page.count("promising") >= 1
    assert [r["id"] for r in live.api("/api/runs?q=validation%20folds")] == [RUN]
    assert [r["id"] for r in live.api("/api/runs?q=baseline-ok")] == [RUN]
    assert [a["id"] for a in live.api(f"/api/runs/{RUN}/attempts?q=promising")] == [attempt]
    assert [a["id"] for a in live.api(f"/api/runs/{RUN}/attempts?q=suspiciously")] == [attempt]
    assert live.api(f"/api/runs/{RUN}/attempts?q=nothing-like-this") == []
    assert "data-text" in live.get(f"/cycles/{RUN}/attempts")[1]  # the filter box searches the same text
    live.ok("/api/notes", {"target": run_target, "op": "edit", "note_id": note, "text": "folds look fine now"})
    assert live.api("/api/runs?q=folds%20look%20fine")[0]["id"] == RUN and live.api("/api/runs?q=watch%20the") == []
    live.ok("/api/notes", {"target": run_target, "op": "delete", "note_id": note})
    assert live.api("/api/runs?q=folds%20look%20fine") == []
    assert "folds look fine" not in live.get(f"/cycles/{RUN}")[1]
    ops = [json.loads(line)["op"] for line in (live.dir / "cockpit" / "notes.jsonl").read_text("utf-8").splitlines()]
    assert ops == ["add", "add", "tags", "tags", "edit", "delete"]  # the log keeps the history
    for bad in ({"target": {"type": "run", "id": "nope"}, "text": "x"}, {"target": {"type": "other", "id": RUN}},
                {"target": run_target, "text": ""}, {"target": run_target, "op": "edit", "note_id": "none", "text": "x"}):
        assert live.post("/api/notes", bad)[0] in (400, 404, 409), bad
    assert live.post("/api/tags", {"target": run_target, "tags": ["x" * 41]})[0] == 400
    assert live.post("/api/tags", {"target": run_target, "tags": "tag"})[0] == 400


def _append(path, tag, count):
    for n in range(count):
        store.append_jsonl(path, {"id": f"{tag}-{n}", "text": "x" * 200}, sync=(n % 10 == 0))


# 44
def test_concurrent_appends_from_threads_and_processes_lose_nothing(tmp_path):
    folder = tmp_path / "cockpit"
    paths = [folder / "decisions.jsonl", folder / "audit.jsonl"]
    per, failures, stop = 40, [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                for path in paths:
                    records, _ = store.read_jsonl(path)
                    assert all(isinstance(r, dict) and r["id"] for r in records)
            except Exception as exc:  # a reader must never fail, whatever the writers do
                failures.append(exc)
                return

    readers = [threading.Thread(target=reader) for _ in range(2)]
    writers = [threading.Thread(target=_append, args=(paths[n % 2], f"t{n}", per)) for n in range(16)]
    procs = [multiprocessing.get_context("spawn").Process(target=_append, args=(paths[n], f"p{n}", per)) for n in range(2)]
    for thread in readers + writers:
        thread.start()
    for proc in procs:
        proc.start()
    for thread in writers:
        thread.join(60)
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0
    stop.set()
    for thread in readers:
        thread.join(10)
    assert failures == []
    for n, path in enumerate(paths):
        raw = path.read_text(encoding="utf-8").splitlines()
        records, skipped = store.read_jsonl(path)
        ids = [r["id"] for r in records]
        expected = {f"t{t}-{i}" for t in range(16) if t % 2 == n for i in range(per)} | {f"p{n}-{i}" for i in range(per)}
        assert skipped == 0 and len(raw) == len(ids) == len(set(ids)) == len(expected) and set(ids) == expected
