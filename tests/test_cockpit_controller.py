"""The run controller over real HTTP: detach, one run, stale pids, states, pause, stop, resume, heartbeat."""

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from cockpit_live_helpers import (  # noqa: F401  (fixtures)
    FAKE_RUN,
    finish_cycle,
    live,
    live_factory,
    registry_summary,
    wait_for_stage,
)
from orchestrator_helpers import World

from kaggle_agent.orchestrator import codes
from kaggle_agent.web.codes_text import CODE_TEXT, describe_code
from kaggle_agent.web.controller import RunController


def blocking(project, name="block"):
    """Make every fake Claude call wait while the returned file exists."""
    path = project.tmp / name
    project.set_scenario({**json.loads(project.scenario_file.read_text(encoding="utf-8")), "block_file": str(path)})
    return path


def kill_hard(project):
    info = project.controller._run_info()
    os.killpg(info["pgid"], signal.SIGKILL)
    project.wait_for(lambda: not project.controller.is_alive(info["pid"]), what="the process to die")
    return info


# 20
def test_a_detached_run_writes_the_console_and_a_new_server_reattaches(live):
    block = blocking(live)
    block.write_text("x", encoding="utf-8")
    live.ok("/api/run/start")
    state = live.wait_state("running")
    info = json.loads((live.dir / "cockpit" / "run.json").read_text(encoding="utf-8"))
    pid = info["pid"]
    assert state["pid"] == pid and info["start_time"] and info["pgid"] == pid
    assert os.getpgid(pid) == pid != os.getpgid(0)  # its own process group
    assert os.getsid(pid) == pid != os.getsid(0)  # and its own session: a web restart does not kill it
    assert live.controller.process_start_time(pid) == info["start_time"]
    # the web server object is replaced by a new one over the same folder
    live.stop_server()
    again = RunController(live.dir, orchestrator_argv=live.controller.orchestrator_argv,
                          claude_executable=live.claude, grace_seconds=1.0)
    live.controller = again
    live.start_server(controller=again)
    assert live.controller.is_alive(pid)
    attached = live.run_state()
    assert attached["state"] == "running" and attached["pid"] == pid and attached["heartbeat_age"] is not None
    block.unlink()
    live.wait_exit(10)
    assert live.console().strip()  # the captured output of the orchestrator process
    assert live.api("/api/console?tail=4096")["text"].strip()


# 21
def test_only_one_run_is_active_at_a_time(live):
    block = blocking(live)
    block.write_text("x", encoding="utf-8")
    results = []
    threads = [threading.Thread(target=lambda: results.append(live.post("/api/run/start")[0])) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(results) == [202, 409]
    pid = live.run_state()["pid"]
    status, body = live.post("/api/run/resume")
    assert status == 409
    with_lock = live.controller
    try:
        with_lock.launch("resume")
        raise AssertionError("a second launch must be refused")
    except Exception as exc:
        assert "already active" in str(exc)
    # a start from a terminal: the second orchestrator exits at once with code 1 (run.lock)
    sys.path.insert(0, str(FAKE_RUN.parent))
    import fake_run_main

    code = fake_run_main.main(["resume", "--config", str(live.dir / "config.toml"), "--supervised"],
                              deps=fake_run_main.build({}))
    assert code == codes.ERROR
    assert live.run_state()["pid"] == pid and live.controller.is_alive(pid)
    block.unlink()
    live.wait_exit(10)
    assert len(live.api("/api/runs")) == 1  # one cycle, no second run was created


# 22
def test_stale_pid_and_lock_recovery(live):
    block = blocking(live)
    block.write_text("x", encoding="utf-8")
    live.ok("/api/run/start")
    live.wait_state("running")
    wait_for_stage(live)  # the run has started its cycle: a kill before that would leave nothing to resume
    info = kill_hard(live)
    assert info["pid"] > 0
    state = live.run_state()
    assert state["state"] == "failed" and state["exit"] is None and "disappeared" in state["text"]
    assert not (live.dir / "cockpit" / "exit.json").exists()
    assert live.controller.lock_held() is False
    block.unlink()
    live.ok("/api/run/resume")  # the lock is free, the registry is transactional: the new start works
    live.wait_exit(10)
    # a pid that was reused by another process: the start time does not match, so it is never signalled
    run_json = live.dir / "cockpit" / "run.json"
    original = run_json.read_text(encoding="utf-8")  # restored below: the cleanup of the fixture signals run.json's group
    run_json.write_text(json.dumps({**json.loads(run_json.read_text(encoding="utf-8")), "pid": os.getpid(),
                                    "pgid": os.getpgid(0), "start_time": "Thu Jan  1 00:00:00 1970",
                                    "stop_requested": False}), encoding="utf-8")
    assert live.controller.is_alive(os.getpid()) and not live.controller._verified(live.controller._run_info())
    assert live.run_state()["alive"] is False
    status, body = live.post("/api/run/stop", {"confirm": True, "reason": "test"})
    assert status == 409 and "nothing is running" in body["error"]["message"]
    assert live.post("/api/run/pause")[0] == 409  # nothing is running
    run_json.write_text(original, encoding="utf-8")
    assert os.getpid() > 0  # this test process is still here: it received no signal


# 23
def test_every_state_is_shown_in_plain_words_and_every_exit_code_has_text(live_factory):
    assert set(CODE_TEXT) == set(codes.ALL)
    for code in codes.ALL:
        state, text = describe_code(code)
        assert state and len(text) > 20
    assert describe_code(99)[0] == "failed"
    project = live_factory()
    assert project.run_state()["state"] == "idle" and "Nothing is running" in project.run_state()["text"]
    project.ok("/api/run/start")
    waiting = project.wait_exit(10)
    assert waiting["state"] == "waiting" and waiting["label"] == "waiting for you (your approval of the plan)"
    assert waiting["exit"]["code"] == 10
    # crafted exit files for the codes a test cannot easily produce (the process id must match run.json)
    pid = waiting["pid"]
    for code, expected in ((13, "failed"), (14, "paused"), (15, "stopped"), (12, "failed"), (0, "finished")):
        (project.dir / "cockpit" / "exit.json").write_text(
            json.dumps({"code": code, "pid": pid, "at": "2026-01-01T00:00:00+00:00", "meaning": "x"}),
            encoding="utf-8")
        shown = project.run_state()
        assert shown["state"] == expected and shown["text"].startswith(describe_code(code)[1][:20]), (code, shown)
    page = project.get("/run")[1]
    assert "Start" in page and "Resume" in page and "Pause" in page and "Stop" in page


# 23
def test_suspended_for_quota_and_for_login_and_the_lead_file_error(live_factory):
    quota = live_factory(scenario={"errors": {"1": "quota"}})
    quota.ok("/api/run/start")
    state = quota.wait_exit(11)
    assert state["state"] == "suspended" and state["suspended_kind"] == "quota"
    assert state["label"] == "suspended (quota)"
    quota.cleanup()
    login = live_factory(scenario={"errors": {"1": "login"}})
    login.ok("/api/run/start")
    state = login.wait_exit(11)
    assert state["suspended_kind"] == "login" and "claude auth login" in state["text"]
    login.cleanup()
    lead = live_factory()
    lead.ok("/api/run/start")
    lead.wait_exit(10)
    (lead.dir / "lead.toml").write_text("[plan]\ndecision = 12\n", encoding="utf-8")
    lead.ok("/api/run/resume")
    state = lead.wait_exit(12)
    assert state["state"] == "failed" and "lead.toml" in state["text"]


# 24
def test_pause_finishes_the_call_exits_with_14_and_the_resumed_registry_equals_the_uninterrupted_one(tmp_path):
    from cockpit_live_helpers import LiveProject

    base = LiveProject(tmp_path / "plain")
    paused = LiveProject(tmp_path / "paused")
    try:
        for project in (base,):
            project.ok("/api/run/start")
            project.wait_exit(10)
            project.decide_plan()
            project.ok("/api/run/resume")
            project.wait_exit(10)
            finish_cycle(project)
        paused.ok("/api/run/start")
        paused.wait_exit(10)
        paused.decide_plan()
        block = blocking(paused)
        block.write_text("x", encoding="utf-8")
        last = paused.events()[-1]["id"]
        paused.ok("/api/run/resume")
        wait_for_stage(paused, last)
        assert paused.ok("/api/run/pause")["pause_requested"] is True
        assert paused.run_state()["pause_requested"] is True
        time.sleep(0.2)
        assert paused.run_state()["alive"] is True  # not interrupted inside the call
        block.unlink()
        state = paused.wait_exit(14)
        assert state["state"] == "paused" and "Resume" in state["text"]
        assert "paused" in paused.event_types()
        assert all(a[3] is None for a in registry_summary(paused.registry_path())["attempts"])  # no failed attempt
        paused.set_scenario({})
        paused.ok("/api/run/resume")
        paused.wait_exit(10)
        finish_cycle(paused)
        got, want = registry_summary(paused.registry_path()), registry_summary(base.registry_path())
        assert {k: v for k, v in got.items() if k != "calls"} == {k: v for k, v in want.items() if k != "calls"}
        # a role call whose result was not yet committed when the pause came is made again on resume (the same as
        # after a crash): the paused run may hold ONE extra call row, never fewer
        extra = list(got["calls"])
        for role in want["calls"]:
            extra.remove(role)
        assert len(extra) <= 1, extra
    finally:
        base.cleanup()
        paused.cleanup()


# 24
def test_a_pause_pending_when_the_cycle_waits_ends_as_waiting_and_the_request_is_removed(live):
    block = blocking(live)
    block.write_text("x", encoding="utf-8")
    live.ok("/api/run/start")
    wait_for_stage(live)
    live.ok("/api/run/pause")
    block.unlink()
    state = live.wait_exit(10)  # the cycle reached AWAITING_PLAN_APPROVAL: waiting, never paused
    assert state["state"] == "waiting"
    live.wait_for(lambda: not (live.dir / "cockpit" / "pause-request.json").exists(), what="the stale request to go")
    assert "paused" not in live.event_types()


# 24
def test_a_pause_pending_when_the_cycle_stops_ends_as_finished(tmp_path):
    from cockpit_live_helpers import drive_supervised

    world = World(tmp_path)
    drive_supervised(world, to="AWAITING_REVIEW")
    from orchestrator_helpers import GUIDE_CONTINUE, PLAN_APPROVE, REVIEW_ACCEPT

    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2) + REVIEW_ACCEPT)
    cockpit = world.project / "cockpit"
    cockpit.mkdir(exist_ok=True)
    (cockpit / "pause-request.json").write_text('{"at": "2026-01-01T00:00:00+00:00"}', encoding="utf-8")
    code, text = world.run("resume", "--supervised")
    assert code == codes.FINISHED and world.state() == "STOPPED", text  # never 14


# 25
def test_stop_is_sigterm_the_sandbox_child_dies_and_the_cycle_resumes(live_factory):
    project = live_factory(scenario={}, grace=20.0)
    pid_file = project.tmp / "bundle.pid"
    scenario = {"real_sandbox": True, "slow_bundle": 120, "bundle_pid_file": str(pid_file)}
    project.set_scenario(scenario)
    project.ok("/api/run/start")
    project.wait_exit(10)
    project.decide_plan()
    project.ok("/api/run/resume")
    project.wait_for(lambda: pid_file.exists() and pid_file.read_text(encoding="utf-8").strip(), what="the bundle child")
    child = int(pid_file.read_text(encoding="utf-8"))
    assert project.controller.is_alive(child) and os.getpgid(child) != os.getpgid(project.run_state()["pid"])
    assert project.post("/api/run/stop", {"reason": "test"})[0] == 400  # confirmation first
    stopping = project.ok("/api/run/stop", {"confirm": True, "reason": "stop for the test"})
    assert stopping["pid"]
    state = project.wait_exit(15)
    assert state["state"] == "stopped" and state["exit"]["code"] == 15
    project.wait_for(lambda: not project.controller.is_alive(child), 10, "the bundle child to be gone")  # no orphan
    exit_json = json.loads((project.dir / "cockpit" / "exit.json").read_text(encoding="utf-8"))
    assert exit_json["code"] == 15
    project.set_scenario({})
    project.ok("/api/run/resume")
    project.wait_exit(10)
    assert registry_summary(project.registry_path())["state"] == "AWAITING_GUIDANCE"


# 25
def test_a_run_that_ignores_sigterm_is_killed_after_the_grace_period_and_resumes(live_factory):
    project = live_factory(scenario={"ignore_sigterm": True}, grace=1.0)
    block = blocking(project)
    block.write_text("x", encoding="utf-8")
    project.ok("/api/run/start")
    state = project.wait_state("running")
    started = time.monotonic()
    project.ok("/api/run/stop", {"confirm": True, "reason": "it ignores the signal"})
    time.sleep(0.4)
    assert project.controller.is_alive(state["pid"])  # SIGTERM alone did not stop it
    project.wait_for(lambda: not project.controller.is_alive(state["pid"]), 10, "SIGKILL after the grace period")
    assert time.monotonic() - started >= 0.9
    assert project.run_state()["state"] == "failed"  # killed: no exit.json (the documented known limit)
    block.unlink()
    project.set_scenario({})
    project.ok("/api/run/resume")
    project.wait_exit(10)


# 52
def test_a_fresh_heartbeat_shows_running_and_a_stale_one_shows_not_responding(live_factory):
    project = live_factory(scenario={"heartbeat": 0.2}, controller_kwargs={"heartbeat_stale_seconds": 1.5})
    block = blocking(project)
    block.write_text("x", encoding="utf-8")
    project.ok("/api/run/start")
    project.wait_for(lambda: (project.dir / "cockpit" / "heartbeat.json").exists(), what="the first heartbeat")
    state = project.wait_state("running")
    assert state["responsive"] is True and state["heartbeat_age"] < 1.5
    pid = state["pid"]
    os.kill(pid, signal.SIGSTOP)  # frozen: the heartbeat thread stops too
    try:
        frozen = project.wait_for(lambda: (s := project.run_state())["responsive"] is False and s, 10, "not responding")
        assert frozen["label"] == "not responding" and frozen["alive"] is True
        assert project.controller.is_alive(pid)  # nothing was killed
    finally:
        os.kill(pid, signal.SIGCONT)
    block.unlink()
    project.wait_exit(10)


# 54
def test_liveness_uses_kill_zero_and_ps_never_the_proc_filesystem(live):
    source = (Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "web" / "controller.py")
    text = source.read_text(encoding="utf-8")
    assert "os.kill(int(pid), 0)" in text and '"lstart="' in text and "start_new_session" in text
    assert "os.killpg" in text
    assert ("/" + "proc") not in text
    assert live.controller.is_alive(os.getpid()) and live.controller.process_start_time(os.getpid())
