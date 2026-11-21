"""The analytics API: old projects, per-fold, segments, hardest rows, features, compare, history, side by side (cases 7, 14, 22, 23, 30, 34-37, 41)."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from analytics_helpers import (  # noqa: F401  (fixtures)
    ablation_world,
    attempts_of,
    client_for_folder,
    client_of,
    cycle_of,
    failure_world,
    labels_world,
    old_project,
    proba_world,
)
from test_analytics_recording import partition_metrics

from kaggle_agent.web.writes import WRITE_ENDPOINTS

ROOT = Path(__file__).resolve().parent.parent
JS = ROOT / "src" / "kaggle_agent" / "web" / "static" / "js"


def data(client, url, status=200):
    response = client.get(url)
    assert response.status_code == status, (url, response.status_code, response.get_data(as_text=True)[:300])
    body = response.get_json()
    return body["data"] if status == 200 else body["error"]


def attempt_urls(attempt):
    return [f"/api/attempts/{attempt}/{part}" for part in ("analytics", "folds", "panels", "segments", "hardest", "flags", "features")]


def run_urls(run, a, b):
    return [f"/api/runs/{run}/{part}" for part in ("history", "compare", "ablation")] + [
        f"/api/runs/{run}/paired?a={a}&b={b}", f"/api/runs/{run}/side-by-side?ids={a},{b}"]


# 7
def test_an_old_project_opens_everywhere_and_says_not_recorded(labels_world, tmp_path):
    folder = old_project(labels_world, tmp_path / "old")
    client = client_for_folder(folder)
    a, b = attempts_of(labels_world)
    run = cycle_of(labels_world)
    for url in attempt_urls(b) + run_urls(run, a, b):
        assert client.get(url).status_code == 200, url
    for url in (f"/attempts/{b}/analytics", f"/runs/{run}/compare", f"/runs/{run}/paired?a={a}&b={b}",
                f"/runs/{run}/compare?ids={a},{b}"):
        assert client.get(url).status_code == 200, url
    folds = data(client, f"/api/attempts/{b}/folds")
    registry = partition_metrics(labels_world, b)
    assert {p["id"]: p["score"] for p in folds["points"]} == pytest.approx(registry) and folds["source"] == "registry"
    for part in ("panels", "segments", "hardest"):
        out = data(client, f"/api/attempts/{b}/{part}")
        assert out["not_recorded"] is True and out["code"] == "older_project", part
    features = data(client, f"/api/attempts/{b}/features")
    assert features["describe"]["code"] == "older_project" and features["permutation"]["not_recorded"] is True
    page = client.get(f"/attempts/{b}/analytics").get_data(as_text=True)
    assert "Not recorded for this attempt" in page and "before analytics were recorded" in page
    assert not (folder / "analytics").exists() and not (folder / "cockpit").exists()  # a GET creates nothing
    compare = data(client, f"/api/runs/{run}/compare")
    assert compare["count"] == 2 and all(r["family"] is None for r in compare["rows"])


# 14
def test_the_folds_endpoint_returns_every_partition_with_spread_and_worst(proba_world):
    client = client_of(proba_world)
    attempt = attempts_of(proba_world)[-1]
    out = data(client, f"/api/attempts/{attempt}/folds")
    scores = partition_metrics(proba_world, attempt)
    assert len(out["points"]) == 6 and out["spread"] == pytest.approx(np.std(list(scores.values()), ddof=1))
    assert out["direction"] == "minimize" and out["worst"]["score"] == pytest.approx(max(scores.values()))
    assert out["recorded_check"]["matches"] is True
    page = client.get(f"/attempts/{attempt}/analytics").get_data(as_text=True)
    for pid in scores:
        assert f"<td>{pid}</td>" in page  # a row (a dot) per fold
    assert 'data-analytics-section="folds"' in page
    assert data(client, "/api/attempts/nope-1/folds", 404)["status"] == 404
    assert data(client, "/api/attempts/..%2Fx/folds", 404)["status"] == 404


# 22
def test_the_segment_endpoint_filters_by_min_size_without_changing_the_ranking(proba_world):
    client = client_of(proba_world)
    attempt = attempts_of(proba_world)[-1]
    base = f"/api/attempts/{attempt}/segments"
    full = data(client, f"{base}?min_size=1&top=100")
    assert full["segments"] and "lift" in full["segments"][0] and full["overall_error"] > 0
    order = [(s["feature"], s["segment"]) for s in full["segments"]]
    for size in (1, 20, 200):
        shown = data(client, f"{base}?min_size={size}&top=100")
        assert all(s["count"] >= size for s in shown["segments"] + shown["interactions"])
        kept = [(s["feature"], s["segment"]) for s in shown["segments"]]
        assert kept == [k for k in order if k in set(kept)]
    assert data(client, f"{base}?min_size=200")["segments"] == []
    assert data(client, f"{base}?interactions=0")["interactions"] == []
    for bad in ("min_size=0", "min_size=abc", "top=0", "top=101", "interactions=maybe"):
        assert data(client, f"{base}?{bad}", 400)["status"] == 400, bad


# 23
def test_the_hardest_endpoint_sorts_caps_and_stays_in_the_development_rows(proba_world):
    client = client_of(proba_world)
    attempt = attempts_of(proba_world)[-1]
    base = f"/api/attempts/{attempt}/hardest"
    worst = data(client, f"{base}?limit=10")
    errors = [r["error"] for r in worst["rows"]]
    assert errors == sorted(errors, reverse=True) and len(errors) == 10 and len(worst["columns"]) <= 4 + 30
    for sort in worst["columns"]:
        for order in ("asc", "desc"):
            rows = data(client, f"{base}?limit=10&sort={sort}&order={order}")["rows"]
            values = [r[sort] for r in rows if r[sort] is not None]
            assert values == sorted(values, reverse=order == "desc"), (sort, order)
    assert data(client, f"{base}?limit=200")["limit"] == 200
    for bad in ("limit=201", "limit=0", "sort=nope", "order=up", "limit=x"):
        assert data(client, f"{base}?{bad}", 400)["status"] == 400, bad


# 30
def test_native_importance_is_grouped_by_family_and_no_permutation_job_runs_by_itself(proba_world, labels_world):
    with_native = data(client_of(proba_world), f"/api/attempts/{attempts_of(proba_world)[-1]}/features")
    groups = {g["family"]: g for g in with_native["native"]["groups"]}
    assert groups["x1"]["importance"] == pytest.approx(0.8) and {f["feature"] for f in groups["x1"]["features"]} == {"x1", "x1_squared"}
    assert with_native["permutation"]["not_recorded"] is True and with_native["permutation"]["code"] == "no_importance"
    without = data(client_of(labels_world), f"/api/attempts/{attempts_of(labels_world)[-1]}/features")
    assert without["native"] is None and without["describe"]["status"] == "missing"
    for world in (proba_world, labels_world):
        page = client_of(world).get(f"/attempts/{attempts_of(world)[-1]}/analytics").get_data(as_text=True)
        assert 'id="importance-start"' in page and "Compute permutation importance" in page
        assert not (world.project / "cockpit" / "jobs").exists()  # nothing was started
        assert not (world.project / "analytics" / "importance").exists()


# 34
def test_the_ablation_table_groups_by_family_counts_missing_describes_and_starts_nothing(ablation_world):
    client = client_of(ablation_world)
    run = cycle_of(ablation_world)
    a0, a1, a2 = attempts_of(ablation_world)
    out = data(client, f"/api/runs/{run}/ablation")
    assert out["families"]["x1_sq"] == "x1" and out["families"]["x2"] == "x2"  # derived under its source column
    assert out["not_provided"] == [a2]
    row = next(r for r in out["rows"] if r["family"] == "x1")
    assert row["with"]["attempts"] == [a0] and row["without"]["attempts"] == [a1]
    assert "x2" not in [r["family"] for r in out["rows"]]  # every described attempt uses it: nothing to compare
    page = client.get(f"/runs/{run}/compare").get_data(as_text=True)
    assert "Display only" in page and f"Not provided (no describe.json): {a2}" in page
    assert [r.rule for r in client.application.url_map.iter_rules() if "ablation" in r.rule] == ["/api/runs/<path:run_id>/ablation"]
    assert not [w for w in WRITE_ENDPOINTS if "ablation" in w]
    assert client.post(f"/api/runs/{run}/ablation").status_code == 405


# 35
def test_the_compare_table_sorts_filters_and_shows_the_status_of_failed_attempts(ablation_world, failure_world):
    client = client_of(ablation_world)
    run = cycle_of(ablation_world)
    base = f"/api/runs/{run}/compare"
    a0, a1, a2 = attempts_of(ablation_world)
    rows = data(client, base)["rows"]
    assert [r["attempt"] for r in rows] == [a0, a1, a2]
    assert {"score", "spread", "round", "family", "features", "time"} <= set(rows[0])
    assert rows[0]["family"] == "linear" and rows[0]["features"] == 3 and rows[2]["family"] is None
    by_family = data(client, f"{base}?family=boost")["rows"]
    assert [r["attempt"] for r in by_family] == [a1]
    assert [r["attempt"] for r in data(client, f"{base}?round={rows[1]['round']}")["rows"]] == [a1]
    assert [r["attempt"] for r in data(client, f"{base}?q=hypothesis")["rows"]] == [r["attempt"] for r in rows]
    scores = [r["score"] for r in data(client, f"{base}?sort=score&order=desc")["rows"]]
    assert scores == sorted(scores, reverse=True)
    assert data(client, f"{base}?sort=colour", 400)["status"] == 400
    assert data(client, f"{base}?order=sideways", 400)["status"] == 400
    failed = data(client_of(failure_world), f"/api/runs/{cycle_of(failure_world)}/compare")["rows"]
    row = next(r for r in failed if r["status"] == "failed")
    assert row["score"] is None and row["spread"] is None
    ordered = data(client_of(failure_world), f"/api/runs/{cycle_of(failure_world)}/compare?sort=score&order=asc")["rows"]
    assert ordered[-1]["status"] == "failed"  # without a score it always comes last


# 36
def test_the_history_has_points_in_order_best_so_far_baseline_and_direction(ablation_world):
    client = client_of(ablation_world)
    out = data(client, f"/api/runs/{cycle_of(ablation_world)}/history")
    assert out["direction"] == "maximize" and out["metric"] == "accuracy"
    assert [p["attempt"] for p in out["points"]] == attempts_of(ablation_world)
    assert [p["iteration"] for p in out["points"]] == sorted(p["iteration"] for p in out["points"])
    best = None
    for point in out["points"]:
        best = point["score"] if best is None else max(best, point["score"])
        assert point["best"] == best
    assert out["baseline"]["attempt_id"] == attempts_of(ablation_world)[0]
    colour = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(")
    for name in ("analytics.js", "charts.js"):
        assert not colour.search((JS / name).read_text(encoding="utf-8")), name


# 37
def test_the_side_by_side_shows_hypothesis_features_hyperparameters_and_folds_without_code(ablation_world):
    client = client_of(ablation_world)
    run = cycle_of(ablation_world)
    a0, a1, a2 = attempts_of(ablation_world)
    text = client.get(f"/api/runs/{run}/side-by-side?ids={a0},{a1},{a2}").get_data(as_text=True)
    out = json.loads(text)["data"]["attempts"]
    assert [x["attempt"] for x in out] == [a0, a1, a2]
    assert out[0]["features"] == ["x1", "x2", "x1_sq"] and out[0]["hyperparameters"] == {"depth": 2}
    assert out[1]["hyperparameters"] == {"depth": 4, "rate": 0.1}
    assert out[2]["describe"] == "not provided" and out[2]["features"] is None
    assert all(len(x["folds"]) == 3 and x["hypothesis"] for x in out)
    for forbidden in ("def fit", "model.py", "import ", str(ablation_world.base), "workspace"):
        assert forbidden not in text, forbidden
    page = client.get(f"/runs/{run}/compare?ids={a0},{a2}").get_data(as_text=True)
    assert "not provided" in page and "depth = 2" in page
    for ids in (a0, f"{a0},{a1},{a2},{a0}", f"{a0},{a0}", ""):
        assert data(client, f"/api/runs/{run}/side-by-side?ids={ids}", 400)["status"] == 400, ids


# 41
def test_every_new_data_route_is_get_only_and_the_only_new_write_is_the_importance_job(proba_world):
    client = client_of(proba_world)
    app = client.application
    keys = ("/analytics", "/folds", "/panels", "/segments", "/hardest", "/flags", "/features", "/paired", "/history",
            "/compare", "/side-by-side", "/ablation")
    rules = [r for r in app.url_map.iter_rules() if any(r.rule.endswith(k) for k in keys)]
    assert len(rules) >= 15
    attempt, run = attempts_of(proba_world)[-1], cycle_of(proba_world)
    for rule in rules:
        assert rule.methods - {"HEAD", "OPTIONS"} == {"GET"}, rule.rule
        url = rule.rule.replace("<path:aid>", attempt).replace("<path:rid>", run).replace("<path:attempt_id>", attempt).replace("<path:run_id>", run)
        for method in ("post", "put", "delete"):
            assert getattr(client, method)(url).status_code == 405, (method, url)
    assert "/api/jobs/importance" in WRITE_ENDPOINTS
    assert [w for w in WRITE_ENDPOINTS if "analytics" in w or "importance" in w] == ["/api/jobs/importance"]
