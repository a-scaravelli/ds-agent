"""The fork form and its refusals, frozen settings (cases 26, 27, 37), old projects (28)."""

import json

from cockpit_finish_helpers import FinishWorld, iterating_world
from orchestrator_helpers import World

from kaggle_agent.orchestrator.cycles import read_cycles

FORK = {"from_run": "toy-1", "from_attempt": "toy-1-R1-c1"}


def rows(fw):
    reg = fw.registry()
    try:
        return [c.id for c in reg.list_cycles()]
    finally:
        reg.close()


# 26
def test_a_fork_with_a_new_threshold_is_labelled_and_needs_a_reason(tmp_path):
    fw = FinishWorld(tmp_path)
    before = (fw.project / "config.toml").read_bytes() if (fw.project / "config.toml").exists() else None
    response = fw.post("/api/run/fork", {**FORK, "threshold_k": 0.5, "confirm": True})
    assert response.status_code == 400 and "reason" in response.get_json()["error"]["message"]
    assert read_cycles(fw.project) == []
    data = fw.ok("/api/run/fork", {**FORK, "threshold_k": 0.5, "reason": "try a looser bar", "confirm": True})
    assert data["protocol"] == "changed" and data["duplicate"] is False
    [entry] = read_cycles(fw.project)
    assert entry["protocol_label"] == "changed" and entry["changes"]["threshold_k"][1] == 0.5
    assert entry["reason"] == "try a looser bar" and entry["origin"] == {"cycle_id": "toy-1", "attempt_id": "toy-1-R1-c1"}
    origin = fw.get_json("/api/runs/toy-2/origin")
    assert origin["entry"]["protocol_label"] == "changed"
    assert "new protocol" in str(fw.get_json("/api/runs/toy-2/labels")["labels"])
    assert [a for a in fw.lines("audit.jsonl") if a["what"] == "run.fork"][0]["why"] == "try a looser bar"
    assert before == ((fw.project / "config.toml").read_bytes() if (fw.project / "config.toml").exists() else None)
    # the same request id is one fork
    request_id = fw.controller.argvs[-1][fw.controller.argvs[-1].index("--request-id") + 1]
    again = fw.ok("/api/run/fork", {**FORK, "threshold_k": 0.5, "reason": "try a looser bar", "confirm": True,
                                    "request_id": request_id})
    assert again["duplicate"] is True and rows(fw).count("toy-2") == 1


def test_a_fork_with_the_same_protocol_needs_no_reason(tmp_path):
    fw = FinishWorld(tmp_path)
    data = fw.ok("/api/run/fork", FORK)
    assert data["protocol"] == "same"
    [entry] = read_cycles(fw.project)
    assert entry["protocol_label"] == "same" and rows(fw) == ["toy-1", "toy-2"]
    assert fw.get_json("/api/runs/toy-2/origin")["carry"]["origin_attempt"] == "toy-1-R1-c1"
    assert "toy-2" in fw.page("/fork") or "Start from" in fw.page("/fork")


# 27
def test_a_fork_is_refused_only_for_technical_reasons(tmp_path):
    fw = iterating_world(tmp_path)  # the current cycle is not stopped
    before = fw.registry_hash()
    response = fw.post("/api/run/fork", FORK)
    assert response.status_code == 409 and "not stopped" in response.get_json()["error"]["message"]
    assert fw.post("/api/run/fork", {"from_run": "toy-1", "from_attempt": "toy-1-nope"}).status_code == 404
    assert fw.post("/api/run/fork", {"from_run": "toy-9"}).status_code == 404
    assert fw.post("/api/run/fork", {"from_attempt": "x"}).status_code == 400
    assert fw.registry_hash() == before and read_cycles(fw.project) == []
    fw = FinishWorld(tmp_path / "b")
    assert fw.post("/api/run/fork", {**FORK, "threshold_k": 0.7}).status_code == 400  # a protocol change needs a reason
    fw.controller.active = lambda: True  # a run holds the project
    response = fw.post("/api/run/fork", FORK)
    assert response.status_code == 409 and "run is active" in response.get_json()["error"]["message"]
    assert read_cycles(fw.project) == [] and rows(fw) == ["toy-1"]


# 37
def test_threshold_k_is_not_changed_in_place_and_the_fork_flow_is_the_way(tmp_path):
    fw = FinishWorld(tmp_path)
    response = fw.post("/api/config/save", {"config": {"protocol": {"threshold_k": 0.5}}})
    body = response.get_json()
    text = json.dumps(body)
    assert "never edited in place" in text and "new plan version or a new run" in text
    assert (body.get("data") or {}).get("saved") is not True
    assert fw.get_json("/api/runs/toy-1/origin")["entry"] is None and not (fw.project / "cycles.json").exists()
    fw.ok("/api/run/fork", {**FORK, "threshold_k": 0.5, "reason": "looser", "confirm": True})
    reg = fw.registry()
    try:
        hashes = {c.id: c.protocol_hash for c in reg.list_cycles()}
    finally:
        reg.close()
    assert hashes["toy-2"] != hashes["toy-1"]  # a new protocol hash, only through the fork


# 28
def test_old_projects_open_everywhere_with_empty_new_sections(tmp_path):
    world = World(tmp_path)
    assert world.run("start")[0] == 10
    fw = FinishWorld.__new__(FinishWorld)
    from kaggle_agent.registry import SqliteRegistry
    from kaggle_agent.web import create_app
    from cockpit_finish_helpers import InProcessController

    fw.world, fw.project = world, world.project
    fw.controller = InProcessController(world)
    fw.app = create_app(lambda: SqliteRegistry(world.project / "registry.db", read_only=True),
                        project_dir=world.project, controller=fw.controller)
    fw.client = fw.app.test_client()
    assert not (world.project / "cycles.json").exists() and not (world.project / "cockpit" / "holdout-reads.jsonl").exists()
    for path in ("/api/runs/toy-1/holdout", "/api/runs/toy-1/labels", "/api/runs/toy-1/bundle", "/api/history",
                 "/api/learnings", "/api/overrides/pending", "/api/runs/toy-1/gap", "/api/runs/toy-1/origin",
                 "/api/runs/toy-1/submissions/preview", "/api/runs/toy-1/final-report", "/api/runs/toy-1/bar-override"):
        response = fw.client.get(path)
        assert response.status_code == 200, (path, response.status_code)
    for page in ("/", "/cycles/toy-1", "/cycles/toy-1/report", "/cycles/toy-1/report/print", "/cycles/toy-1/bundle",
                 "/cycles/toy-1/predict", "/cycles/toy-1/submission", "/cycles/toy-1/gap", "/history", "/learnings",
                 "/fork"):
        assert fw.client.get(page).status_code == 200, page
    assert fw.get_json("/api/runs/toy-1/labels")["labels"] == [] and fw.get_json("/api/overrides/pending")["overrides"] == []
    code, text = world.run("status")
    assert code == 0 and "cycle: toy-1" in text
    world.lead("")  # nothing to apply: the legacy cycle resumes as before
    assert not (world.project / "cycles.json").exists()
