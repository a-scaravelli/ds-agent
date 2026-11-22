"""The reproduce and predict jobs of the orchestrator (cases 10, 11, 14)."""

import hashlib
import json
import sqlite3

from overrides_helpers import FIRST, stopped_world

BUNDLE = "toy-1-R0-c1-b1"


def job(world, tmp_path, name, *args):
    out = tmp_path / f"{name}.json"
    out.unlink(missing_ok=True)
    code, text = world.run(name, "--out", str(out), *args)
    return code, json.loads(out.read_text(encoding="utf-8"))


def registry_hash(world):
    return hashlib.sha256((world.project / "registry.db").read_bytes()).hexdigest()


# 10
def test_reproduce_of_an_intact_bundle_is_ok_with_both_values_and_leaves_the_registry_alone(tmp_path):
    world = stopped_world(tmp_path)
    before = registry_hash(world)
    code, result = job(world, tmp_path, "reproduce")
    assert code == 0 and result["ok"] is True and result["mismatch"] is None and result["bundle"] == BUNDLE
    assert result["reproduced"] == result["reported"] == 0.7291666666666666
    assert result["lines"][-1].startswith("OK:")
    assert registry_hash(world) == before  # the registry is opened read-only


# 11
def test_reproduce_shows_the_exact_text_when_the_reported_metric_changed(tmp_path):
    world = stopped_world(tmp_path)
    db = sqlite3.connect(world.project / "registry.db")
    record = json.loads(db.execute("SELECT record FROM final_reports").fetchone()[0])
    record["content"]["chosen"]["validation_mean"] += 0.25  # a report that no longer matches the stored code
    db.execute("UPDATE final_reports SET record = ?", (json.dumps(record),))
    db.commit()
    db.close()
    code, result = job(world, tmp_path, "reproduce")
    assert code == 0 and result["ok"] is False  # the job itself ran: the mismatch is its answer
    assert result["mismatch"] == "error: the metrics differ by 0.25"
    assert result["lines"][-1] == result["mismatch"]


# 11
def test_reproduce_names_the_persisted_file_that_was_tampered_with(tmp_path):
    world = stopped_world(tmp_path)
    model = world.project / "out" / "persisted" / BUNDLE / "code" / "model.py"
    model.write_text(model.read_text(encoding="utf-8") + "\n# edited by hand\n", encoding="utf-8")
    code, result = job(world, tmp_path, "reproduce")
    assert result["ok"] is False
    assert "the persisted code changed: code/model.py" in result["mismatch"]


# 14
def test_predict_returns_one_prediction_per_row_with_the_id_column_and_checks_the_columns(tmp_path):
    world = stopped_world(tmp_path)
    ok = tmp_path / "ok.csv"
    ok.write_text("id,x1,x2\n1,1,1\n2,2,2\n3,3,0\n", encoding="utf-8")
    code, result = job(world, tmp_path, "predict", "--input", str(ok))
    assert code == 0 and result["ok"] is True and result["rows"] == 3 and len(result["predictions"]) == 3
    assert result["id_column"] == "id" and result["ids"] == [1, 2, 3] and result["warnings"] == []
    extra = tmp_path / "extra.csv"
    extra.write_text("id,x1,x2,zzz\n1,1,1,5\n", encoding="utf-8")
    result = job(world, tmp_path, "predict", "--input", str(extra))[1]
    assert result["ok"] is True and result["warnings"] == ["extra column ignored: zzz"]
    missing = tmp_path / "missing.csv"
    missing.write_text("id,x1\n1,1\n", encoding="utf-8")
    result = job(world, tmp_path, "predict", "--input", str(missing))[1]
    assert result["ok"] is False and result["missing"] == ["x2"] and result["error"] == "missing columns: x2"
    assert FIRST == "toy-1"
