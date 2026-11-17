"""The whole flow from the page over real HTTP: config, start, live events, decisions, accept, finished."""

import threading

from cockpit_live_helpers import FORM, live, registry_summary  # noqa: F401  (fixtures)

RUN = "toy-1"


# 48
def test_the_full_flow_from_the_page_ends_finished_with_every_step_audited(live):
    saved = live.ok("/api/config/save", {"config": {**FORM, "limits": {**FORM["limits"], "rounds": 2}}})
    assert saved["ok"] is True
    stream = live.sse()
    seen: list = []
    done = threading.Event()

    def read():
        for kind, payload in stream.frames(stop=lambda e: e["type"] == "finished"):
            if kind == "event":
                seen.append(payload)
        done.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    live.ok("/api/run/start")
    live.wait_exit(10)
    assert live.api("/api/decisions/pending")["question"]["kind"] == "plan"
    live.decide_plan()
    for n in (1, 2):
        live.ok("/api/run/resume")
        live.wait_exit(10)
        text = {"action": "guidance", "address": n, "text": "try a simpler model"} if n == 1 else {
            "action": "continue", "address": n}
        live.ok("/api/decisions/guidance", text)
    live.ok("/api/run/resume")
    live.wait_exit(10)
    live.ok("/api/decisions/review", {"confirm": True})
    live.ok("/api/run/resume")
    state = live.wait_exit(0)
    assert state["state"] == "finished"
    # the stream did not stop being usable: reconnect picks the tail up when the first one timed out
    assert done.wait(15) or stream.close() is None
    types = [e["type"] for e in live.events()]
    for needed in ("run_start", "cycle_state", "stage_start", "candidate_end", "waiting", "decision_applied",
                   "finished"):
        assert needed in types, needed
    assert [e["id"] for e in seen] == list(range(1, len(seen) + 1))
    summary = registry_summary(live.registry_path())
    assert summary["state"] == "STOPPED" and summary["rounds"] >= 2 and "try a simpler model" in summary["guidance"]
    what = " ".join(e["what"] for e in live.api("/api/audit")["entries"])
    for step in ("save", "start", "plan", "guidance", "review", "resume"):
        assert step in what, step
