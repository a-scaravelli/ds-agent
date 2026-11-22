"""Reproduce, downloads and predict through the page (cases 10, 11, 12, 13, 14, 58)."""

import io
import json
import sqlite3

from cockpit_finish_helpers import FinishWorld

BUNDLE = "toy-1-R0-c1-b1"
GOOD = b"id,x1,x2\n1,1,1\n2,2,2\n3,3,0\n"


def upload(fw, blob: bytes, name="new.csv", **kwargs):
    headers = {"Origin": "http://localhost", "X-CSRF-Token": fw.token}
    return fw.client.post("/api/jobs/predict", data={"file": (io.BytesIO(blob), name)}, headers=headers,
                          content_type="multipart/form-data", **kwargs)


def uploads(fw):
    folder = fw.project / "cockpit" / "uploads"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


# 10
def test_reproduce_runs_as_a_job_and_shows_the_ok_line_with_both_values(tmp_path):
    fw = FinishWorld(tmp_path)
    before = fw.registry_hash()
    job = fw.ok("/api/jobs/reproduce")["job"]
    done = fw.wait_job(job)
    assert done["status"] == "done" and done["result"]["ok"] is True and done["kind"] == "reproduce"
    assert done["result"]["lines"][-1].startswith("OK:") and done["result"]["reproduced"] == done["result"]["reported"]
    assert fw.registry_hash() == before
    assert fw.controller.argvs[-1][1] == "reproduce"
    bundle = fw.get_json("/api/runs/toy-1/bundle")
    assert bundle["reproduce"]["result"]["ok"] is True
    assert "OK:" in fw.page("/cycles/toy-1/bundle")


# 11
def test_reproduce_shows_the_exact_mismatch_text_for_a_changed_metric_and_a_tampered_file(tmp_path):
    fw = FinishWorld(tmp_path)
    db = sqlite3.connect(fw.project / "registry.db")
    record = json.loads(db.execute("SELECT record FROM final_reports").fetchone()[0])
    record["content"]["chosen"]["validation_mean"] += 0.25
    db.execute("UPDATE final_reports SET record = ?", (json.dumps(record),))
    db.commit()
    db.close()
    done = fw.wait_job(fw.ok("/api/jobs/reproduce")["job"])
    assert done["result"]["ok"] is False and done["result"]["mismatch"] == "error: the metrics differ by 0.25"
    assert "the metrics differ by 0.25" in fw.page("/cycles/toy-1/bundle")
    fw = FinishWorld(tmp_path / "second")
    model = fw.project / "out" / "persisted" / BUNDLE / "code" / "model.py"
    model.write_text(model.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    done = fw.wait_job(fw.ok("/api/jobs/reproduce")["job"])
    assert "the persisted code changed: code/model.py" in done["result"]["mismatch"]
    assert "the persisted code changed: code/model.py" in fw.page("/cycles/toy-1/bundle")


# 12
def test_only_the_allowed_bundle_files_download_and_the_code_never_does(tmp_path):
    fw = FinishWorld(tmp_path)
    files = {f["name"]: f["sha256"] for f in fw.get_json("/api/runs/toy-1/bundle")["files"]}
    assert "persisted.json" in files and "report.md" in files and any(n.startswith("model/") for n in files)
    for name, sha in files.items():
        response = fw.client.get(f"/api/runs/toy-1/bundle/files/{name}")
        assert response.status_code == 200 and response.headers["X-Content-SHA256"] == sha, name
        assert "attachment" in response.headers["Content-Disposition"]
    for name in ("code/model.py", "code/features.py", "../registry.db", "..%2Fregistry.db", "model/../../registry.db",
                 "/etc/passwd", "nothing.bin", "model/", "code"):
        assert fw.client.get(f"/api/runs/toy-1/bundle/files/{name}").status_code == 404, name
    assert fw.client.get("/api/runs/toy-9/bundle/files/persisted.json").status_code == 404


# 13
def test_the_upload_is_validated_by_size_name_columns_and_rows_and_nothing_is_kept_after_a_rejection(tmp_path):
    fw = FinishWorld(tmp_path)
    response = upload(fw, b"id,x1\n1,1\n")
    assert response.status_code == 400 and "x2" in response.get_json()["error"]["message"]
    assert "missing columns" in response.get_json()["error"]["message"]
    assert upload(fw, GOOD, name="data.txt").status_code == 400
    assert upload(fw, b"\x00\x01\x02binary", name="data.csv").status_code == 400
    assert upload(fw, b"x" * (21 * 1024 * 1024), name="big.csv").status_code == 413
    many = b"id,x1,x2\n" + b"1,1,1\n" * 100_001
    response = upload(fw, many)
    assert response.status_code == 400 and "100000" in response.get_json()["error"]["message"]
    assert uploads(fw) == []  # nothing is stored after a rejection
    assert fw.client.post("/api/jobs/predict", data={}, headers={"Origin": "http://localhost",
                          "X-CSRF-Token": fw.token}).status_code == 400  # no file at all
    extra = upload(fw, b"id,x1,x2,zzz\n1,1,1,5\n")
    assert extra.status_code == 202 and extra.get_json()["data"]["warnings"] == ["extra column ignored: zzz"]
    fw.wait_job(extra.get_json()["data"]["job"])


# 14
def test_predict_runs_the_orchestrator_command_and_returns_a_prediction_per_row_and_a_csv(tmp_path):
    fw = FinishWorld(tmp_path)
    before = fw.registry_hash()
    response = upload(fw, GOOD)
    assert response.status_code == 202 and response.get_json()["data"]["rows"] == 3
    done = fw.wait_job(response.get_json()["data"]["job"])
    result = done["result"]
    assert done["status"] == "done" and result["ok"] is True and result["rows"] == 3 and len(result["predictions"]) == 3
    assert fw.controller.argvs[-1][1] == "predict"
    csv = fw.client.get(f"/api/jobs/{done['id']}/predictions.csv")
    assert csv.status_code == 200 and "attachment" in csv.headers["Content-Disposition"]
    assert len(csv.get_data(as_text=True).strip().splitlines()) == 4
    assert uploads(fw) == [] and fw.registry_hash() == before


# 58
def test_an_upload_is_stored_under_a_generated_name_inside_the_cap_and_removed_after_the_job(tmp_path):
    fw = FinishWorld(tmp_path)
    seen = []
    original = fw.controller._spawn

    def spy(argv, *args, **kwargs):
        seen.append((list(argv), uploads(fw)))
        return original(argv, *args, **kwargs)

    fw.controller._spawn = spy
    formulas = b'id,x1,x2\n1,"=HYPERLINK(""http://evil"")",1\n2,@SUM(A1),2\n3,+cmd|calc,3\n'
    response = upload(fw, formulas, name="../../../etc/evil.csv")
    assert response.status_code == 202, response.get_data(as_text=True)
    done = fw.wait_job(response.get_json()["data"]["job"])
    argv, stored = seen[-1]
    assert len(stored) == 1 and stored[0].endswith(".csv") and "evil" not in stored[0] and ".." not in stored[0]
    assert all("evil" not in part for part in argv)
    assert uploads(fw) == []  # removed after the job
    assert not (tmp_path / "etc").exists() and done["status"] == "done"
    # a failing job leaves nothing either
    fw.world.kaggle  # the fake world stays usable
    broken = upload(fw, b"id,x1,x2\n1,a,b\n")
    if broken.status_code == 202:
        fw.wait_job(broken.get_json()["data"]["job"])
    assert uploads(fw) == []
    body = fw.client.get("/cycles/toy-1/predict").get_data(as_text=True)
    assert "=HYPERLINK" not in body and "uploads" not in body
