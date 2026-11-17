from cockpit_live_helpers import live  # noqa


def test_pages(live):
    for url in ("/", "/live", "/run", "/decisions", "/setup", "/environment", "/audit", "/suggestions"):
        status, body = live.get(url)
        assert status == 200, (url, status, body[:500] if isinstance(body, str) else body)
    live.ok("/api/run/start")
    live.wait_exit(10)
    for url in ("/", "/live", "/run", "/decisions", "/cycles/toy-1", "/cycles/toy-1/attempts"):
        status, body = live.get(url)
        assert status == 200, (url, status, body[:800])
    status, body = live.get("/decisions")
    assert "Approve the plan" in body
    live.decide_plan()
    live.ok("/api/run/resume"); live.wait_exit(10)
    status, body = live.get("/decisions")
    assert "Guidance after round 1" in body, body[:3000]
    for url in ("/cycles/toy-1", "/cycles/toy-1/attempts", "/"):
        status, body = live.get(url)
        assert status == 200
