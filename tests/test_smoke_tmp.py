from cockpit_finish_helpers import *

def test_smoke(tmp_path):
    fw = FinishWorld(tmp_path)
    print(fw.get_json("/api/runs/toy-1/holdout"))
    t = fw.page("/cycles/toy-1/report")
    assert "sealed" in t
    print(fw.get_json("/api/runs/toy-1/bundle")["hashes"])
    print(fw.client.get("/api/runs/toy-1/report.md").get_data(as_text=True)[:2000])
    d = fw.ok("/api/holdout/unseal", {"kind": "first", "confirm": True})
    print(d)
    print(fw.get_json("/api/runs/toy-1/holdout"))
    j = fw.ok("/api/jobs/reproduce")["job"]
    print(fw.wait_job(j))
    for url in ["/cycles/toy-1/bundle","/cycles/toy-1/predict","/cycles/toy-1/submission","/cycles/toy-1/gap","/history","/learnings","/fork","/cycles/toy-1/report/print","/cycles/toy-1/rounds/toy-1-R1","/attempts/toy-1-R1-c1"]:
        fw.page(url)
    j = fw.ok("/api/submission/preview")["job"]
    print(fw.wait_job(j))
