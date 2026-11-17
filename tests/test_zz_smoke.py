from cockpit_live_helpers import live  # noqa


def test_smoke(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    print(live.event_types())
    print(live.console())
    assert live.registry_state() == "AWAITING_PLAN_APPROVAL"
    live.decide_plan()
    print(live.api("/api/decisions/pending"))
    live.ok("/api/run/resume")
    live.wait_exit(10)
    print(live.event_types())
