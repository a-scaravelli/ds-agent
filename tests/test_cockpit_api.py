"""The JSON API: shape against docs/api.md, filters, errors, old files, direction sources (no network)."""

import json
import re
from pathlib import Path

import pytest
from cockpit_helpers import (  # noqa: F401  (fixtures)
    PROJECT_JSON,
    RUN_A,
    RUN_B,
    all_api_urls,
    api,
    build_cockpit_registry,
    client_for,
    cockpit_client,
    cockpit_db,
    loopback_only,
    make_old_registry,
    page,
)
from registry_helpers import make_attempt, make_cycle, make_execution

from kaggle_agent.proposals import Round
from kaggle_agent.registry import ComparisonRecord, PartitionMetric, SqliteRegistry
from kaggle_agent.web import data as web_data

ROOT = Path(__file__).resolve().parent.parent
API_DOC = ROOT / "docs" / "api.md"


def type_of(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def documented_shapes() -> dict:
    """{endpoint: (container, {key: set of types})} parsed from docs/api.md."""
    text = API_DOC.read_text(encoding="utf-8")
    shapes = {}
    for match in re.finditer(r"^### GET (/api/\S+)\n(.*?)(?=^### |^## |\Z)", text, re.S | re.M):
        block = re.search(r"```shape\n(.*?)```", match.group(2), re.S)
        assert block, match.group(1)
        lines = block.group(1).strip().splitlines()
        container = lines[0].split(": ", 1)[1].strip()
        keys = {}
        for line in lines[1:]:
            key, types = line.split(": ", 1)
            keys[key.strip()] = set(types.strip().split("|"))
        shapes[match.group(1)] = (container, keys)
    return shapes


def concrete(endpoint: str) -> str:
    return (endpoint.replace("<id>", "run-a", 1).replace("<round_id>", "run-a-R1")
            .replace("<n>", "3"))


# ticket 27: the live endpoints are documented in docs/api.md and exercised by the live tests; the shape test
# below covers the read endpoints of ticket 26 and the new GET endpoints that answer without a running project
LIVE_ONLY = {"/api/jobs/<id>"}  # needs a job id; checked by the setup tests
# ticket 29: the analytics endpoints answer with data or with a not_recorded object (a different key set); their shapes
# are checked against real recorded runs by tests/test_analytics_docs.py
ANALYTICS_ONLY = {e for e in documented_shapes() if re.search(
    r"/(analytics|folds|panels|segments|hardest|flags|features|paired|compare|side-by-side|ablation)$", e)
    or e == "/api/runs/<id>/history"}
# ticket 30: these answer for a finished project with a bundle; their shapes are checked by tests/test_cockpit_finish_docs.py
FINISH_ONLY = {"/api/runs/<id>/labels", "/api/runs/<id>/origin", "/api/runs/<id>/bundle",
               "/api/runs/<id>/submissions/preview", "/api/runs/<id>/gap", "/api/runs/<id>/bar-override",
               "/api/history", "/api/learnings", "/api/overrides/pending"}
# downloads are not JSON: documented as "FILE", like the stream is documented as "SSE"
NOT_JSON = {"/api/events/stream", "/api/runs/<id>/report.md", "/api/runs/<id>/bundle/files/<name>",
            "/api/jobs/<id>/predictions.csv"}


def endpoint_url(endpoint: str) -> str:
    if endpoint.startswith("/api/attempts/"):
        return endpoint.replace("<id>", "a1")
    return concrete(endpoint)


def check_shape(endpoint, body):
    container, keys = documented_shapes()[endpoint]
    assert set(body) == {"api_version", "data", "warnings"}, endpoint
    assert body["api_version"] == 1 and isinstance(body["warnings"], list)
    data = body["data"]
    if container == "array<object>":
        assert isinstance(data, list), endpoint
        items = data
    else:
        assert isinstance(data, dict), endpoint
        items = [data]
    for item in items:
        assert set(item) == set(keys), (endpoint, set(item) ^ set(keys))
        for key, value in item.items():
            assert type_of(value) in keys[key], (endpoint, key, type_of(value), keys[key])
    return items


# 8
def test_every_endpoint_has_the_documented_shape(cockpit_client):
    shapes = documented_shapes()
    rules = {
        re.sub(r"<(?:path:|int:)?(\w+)>", lambda m: "<id>" if m.group(1) in
               ("run_id", "attempt_id", "job_id") else f"<{m.group(1)}>", r.rule)
        for r in cockpit_client.application.url_map.iter_rules()
        if r.rule.startswith("/api/") and "GET" in (r.methods or ())
    }
    # the server-sent stream is not JSON: it is documented as "SSE", not "GET"
    assert rules - NOT_JSON == set(shapes), (rules - NOT_JSON) ^ set(shapes)
    nonempty = 0
    for endpoint in shapes:
        if endpoint in LIVE_ONLY | ANALYTICS_ONLY | FINISH_ONLY:
            continue
        body = api(cockpit_client, endpoint_url(endpoint))
        nonempty += bool(check_shape(endpoint, body))
    assert nonempty == len(shapes) - len(LIVE_ONLY | ANALYTICS_ONLY | FINISH_ONLY)  # the fixture has data for every endpoint
    # the other run (running, nothing read) and the attempt without scores have the same shapes
    check_shape("/api/runs/<id>", api(cockpit_client, f"/api/runs/{RUN_B}"))
    check_shape("/api/runs/<id>/holdout", api(cockpit_client, f"/api/runs/{RUN_B}/holdout"))
    check_shape("/api/runs/<id>/final-report", api(cockpit_client, f"/api/runs/{RUN_B}/final-report"))
    check_shape("/api/attempts/<id>", api(cockpit_client, "/api/attempts/a2"))
    check_shape("/api/attempts/<id>", api(cockpit_client, "/api/attempts/a-base"))
    check_shape("/api/runs/<id>/role-traces/<n>", api(cockpit_client, f"/api/runs/{RUN_A}/role-traces/5"))


# 8
def test_the_doc_lists_the_endpoints_bundles_and_no_code_or_diff_endpoint(cockpit_client):
    shapes = documented_shapes()
    assert "/api/runs/<id>/bundles" in shapes
    rows = api(cockpit_client, f"/api/runs/{RUN_A}/bundles")["data"]
    assert [b["id"] for b in rows] == ["a1-b1"]
    assert rows[0]["reference"] == "/abs/path/<b>x</b>/bundle"
    for endpoint in shapes:
        assert not re.search(r"code|diff|files?\b|workspace", endpoint), endpoint
    for rule in cockpit_client.application.url_map.iter_rules():
        # the one file route of ticket 30 is an allowlist of bundle downloads (the code/ folder is never served)
        assert not re.search(r"code|diff|/files?\b|workspace", rule.rule.replace("/bundle/files/<path:name>", "")), rule.rule
    for url in (f"/api/runs/{RUN_A}/code", f"/api/attempts/a1/diff", f"/api/runs/{RUN_A}/files"):
        response = cockpit_client.get(url)
        assert response.status_code == 404 and "error" in response.get_json()


# 8
def test_an_unknown_id_gives_404_with_the_json_error_envelope(cockpit_client):
    for url in ("/api/runs/nope", "/api/runs/nope/chart", "/api/runs/nope/attempts",
                f"/api/runs/{RUN_A}/rounds/nope", "/api/attempts/nope", f"/api/runs/{RUN_A}/role-traces/99",
                "/api/runs/nope/bundles", "/api/nothing"):
        response = cockpit_client.get(url)
        body = response.get_json()
        assert response.status_code == 404, url
        assert body["api_version"] == 1 and set(body["error"]) == {"status", "title", "message"}
        assert body["error"]["status"] == 404 and "data" not in body


# 9
def test_attempt_filters_return_only_matching_attempts_and_intersect(cockpit_client):
    def ids(query):
        body = api(cockpit_client, f"/api/runs/{RUN_A}/attempts{query}")
        return [a["id"] for a in body["data"]]

    assert ids("") == ["a-base", "a1", "a2", "a3", "a4"]
    assert ids("?outcome=failed") == ["a2", "a4"]
    assert ids("?outcome=succeeded") == ["a-base", "a1", "a3"]
    assert ids("?round=run-a-R2") == ["a3", "a4"]
    assert ids("?min_score=0.7") == ["a1", "a3"]
    assert ids("?max_score=0.7") == ["a-base", "a1"]
    assert ids("?min_score=0.65&max_score=0.71") == ["a1"]
    assert ids("?q=TREES") == ["a1"]
    assert ids("?q=useless") == ["a2"]
    assert ids("?outcome=succeeded&round=run-a-R1") == ["a1"]
    assert ids("?outcome=failed&min_score=0.1") == []  # a failed attempt has no score
    assert ids("?outcome=succeeded&q=depth&min_score=0.7") == ["a3"]
    for bad in ("?min_score=abc", "?max_score=1e", "?outcome=banana", "?min_score=nan"):
        response = cockpit_client.get(f"/api/runs/{RUN_A}/attempts{bad}")
        assert response.status_code == 400, bad
        assert response.get_json()["error"]["status"] == 400


def test_llm_and_trace_filters(cockpit_client):
    base = f"/api/runs/{RUN_A}"
    calls = api(cockpit_client, f"{base}/llm-calls")["data"]
    assert [c["role"] for c in calls] == ["modeling", "reviewer", "data"]
    assert [c["index"] for c in calls] == [1, 2, 3]
    assert len(api(cockpit_client, f"{base}/llm-calls?role=reviewer")["data"]) == 1
    assert len(api(cockpit_client, f"{base}/llm-calls?outcome=ok")["data"]) == 2
    assert len(api(cockpit_client, f"{base}/llm-calls?q=bad json")["data"]) == 1
    assert cockpit_client.get(f"{base}/llm-calls?role=banana").status_code == 400
    traces = api(cockpit_client, f"{base}/role-traces")["data"]
    assert [t["n"] for t in traces] == [1, 2, 3, 4, 5]
    assert len(api(cockpit_client, f"{base}/role-traces?role=reviewer")["data"]) == 2
    assert len(api(cockpit_client, f"{base}/role-traces?round=run-a-R2")["data"]) == 1
    assert len(api(cockpit_client, f"{base}/role-traces?candidate=a1")["data"]) == 1
    detail = api(cockpit_client, f"{base}/role-traces/3")["data"]
    assert detail["input"] == {"task": "trees"} and detail["output"]["id"] == "a1"
    assert api(cockpit_client, f"{base}/role-traces/5")["data"]["output"] is None


# 7
def test_an_old_registry_file_gives_200_with_empty_sections_and_warnings(cockpit_db, tmp_path):
    old = make_old_registry(cockpit_db, tmp_path.resolve() / "old.db")
    client = client_for(old, cockpit_db.parent)
    for url in all_api_urls():
        response = client.get(url)
        if url.endswith(("rounds/run-a-R1", "role-traces/3")):
            assert response.status_code == 404, url  # nothing to find in a missing table
            continue
        assert response.status_code == 200, (url, response.get_data(as_text=True)[:300])
        assert isinstance(response.get_json()["warnings"], list)
    runs = api(client, "/api/runs")
    sections = {w["section"] for w in runs["warnings"]}
    assert "rounds" in sections
    assert all("not available in this registry file" in w["reason"] for w in runs["warnings"])
    assert api(client, f"/api/runs/{RUN_A}/rounds")["data"] == []
    assert api(client, f"/api/runs/{RUN_A}/bundles")["data"] == []
    assert {w["section"] for w in api(client, f"/api/runs/{RUN_A}/bundles")["warnings"]} == {"bundles"}
    assert api(client, f"/api/runs/{RUN_A}/reviewer-reports")["data"] == []
    assert api(client, f"/api/runs/{RUN_A}/role-traces")["data"] == []
    assert len(api(client, f"/api/runs/{RUN_A}/attempts")["data"]) == 5
    holdout = api(client, f"/api/runs/{RUN_A}/holdout")["data"]  # the reads table still exists
    assert holdout["status"] == "sealed" and holdout["score"] is None  # sealed: no number before the DS unsealed it
    # the pages render with a note instead of failing
    for url in ("/", f"/cycles/{RUN_A}", f"/cycles/{RUN_A}/rounds", f"/cycles/{RUN_A}/attempts",
                f"/cycles/{RUN_A}/traces", f"/cycles/{RUN_A}/report", f"/cycles/{RUN_A}/guidance",
                "/attempts/a1", "/suggestions"):
        html = page(client, url)
    assert "not available in this registry file" in page(client, f"/cycles/{RUN_A}")


def test_an_old_file_without_the_cycle_columns_gives_a_503_page_not_a_500(cockpit_db, tmp_path):
    import shutil
    import sqlite3

    old = tmp_path.resolve() / "older.db"
    shutil.copy(cockpit_db, old)
    db = sqlite3.connect(old)
    db.execute("ALTER TABLE cycles DROP COLUMN protocol_hash")
    db.commit()
    db.close()
    client = client_for(old)
    runs = client.get("/api/runs")
    assert runs.status_code == 200 and runs.get_json()["data"] == []
    assert runs.get_json()["warnings"][0]["section"] == "runs"
    for url in (f"/api/runs/{RUN_A}", f"/cycles/{RUN_A}"):
        response = client.get(url)
        assert response.status_code == 503 and "Traceback" not in response.get_data(as_text=True)
    assert client.get("/").status_code == 200


# 10
def test_the_overview_html_shows_the_values_of_the_json(cockpit_client):
    body = api(cockpit_client, f"/api/runs/{RUN_A}")["data"]
    html = page(cockpit_client, f"/cycles/{RUN_A}")
    assert f'{body["rounds_done"]} of {body["rounds_planned"]}' in html
    assert f'>{body["best"]["mean"]:.6g}' in html
    assert f'href="/attempts/{body["chosen"]["attempt_id"]}"' in html
    assert body["stop_reason"] in html
    assert body["protocol"]["hash"][:12] in html
    assert body["best"]["mean"] == 0.72 and body["chosen"]["attempt_id"] == "a3"
    assert body["stop_reason"] == "stagnation" and body["rounds_done"] == 2
    # both come from the same module
    source = (ROOT / "src/kaggle_agent/web/pages.py").read_text(encoding="utf-8")
    assert "data.run_summary" in source and "from . import data" in source
    assert "data.run_summary" in (ROOT / "src/kaggle_agent/web/api.py").read_text(encoding="utf-8")


def _summary_for(tmp_path, *, project=None, comparisons=(), values=(0.6, 0.7), traces=None):
    """A small cycle whose protocol must be found without a protocol in the registry."""
    path = tmp_path.resolve() / "p.db"
    reg = SqliteRegistry(path)
    reg.add_cycle(make_cycle("c1", iterations=3))
    ids = []
    for n, value in enumerate(values, start=1):
        aid = f"p{n}"
        reg.add_attempt("c1", make_attempt(aid), f"fp{n}")
        reg.add_execution(make_execution(aid, metrics=[PartitionMetric("validation", value, 0.01)]))
        reg.close_attempt(aid, "ok")
        ids.append(aid)
    reg.add_round("c1", Round("c1-R0", ("base",), (ids[0],)))
    reg.add_round("c1", Round("c1-R1", ("try",), tuple(ids[1:])))
    for c in comparisons:
        reg.add_comparison(ComparisonRecord("c1", *c))
    if traces:
        from registry_helpers import make_trace
        reg.add_role_trace(make_trace("c1", "coordinator", "c1-R1", input=traces))
    reg.close()
    if project is not None:
        (path.parent / "project.json").write_text(json.dumps(project), encoding="utf-8")
    return path


# 17
def test_direction_and_protocol_come_from_the_project_file_when_there_is_one(tmp_path):
    project = {"metric": "rmse", "direction": "minimize",
               "protocol": {"metric": "rmse", "direction": "minimize", "validation_scheme": "holdout",
                            "repetitions": 5, "threshold_k": 2.0, "seed": 9}}
    path = _summary_for(tmp_path, project=project)
    body = api(client_for(path, path.parent), "/api/runs/c1")["data"]
    protocol = body["protocol"]
    assert protocol["direction"] == "min" and protocol["direction_source"] == "project.json"
    assert (protocol["metric"], protocol["scheme"], protocol["repeats"], protocol["threshold_k"],
            protocol["seed"]) == ("rmse", "holdout", 5, 2.0, 9)
    assert protocol["sources"]["scheme"] == "project.json" and protocol["project_file"] is True
    assert body["best"]["attempt_id"] == "p1"  # lower is better
    chart = api(client_for(path, path.parent), "/api/runs/c1/chart")["data"]
    assert chart["direction"] == "min"


def test_direction_is_inferred_from_the_comparisons_without_a_project_file(tmp_path):
    limits = {"rounds": 3, "candidates": 2, "stagnation_k": 2, "fix_retries": 3, "current_round": 1,
              "threshold_k": 1.25, "validation_scheme": "stratified", "repetitions": 4,
              "allowed_libraries": ["numpy"]}
    # a lower raw score with a positive stored difference: the metric is minimised
    path = _summary_for(tmp_path, comparisons=[("p2", "p1", 0.1, 0.02, True)], values=(0.7, 0.6),
                        traces={"run_limits": limits})
    client = client_for(path, path.parent)
    body = api(client, "/api/runs/c1")["data"]
    protocol = body["protocol"]
    assert protocol["direction"] == "min" and protocol["direction_source"] == "inferred from comparisons"
    assert (protocol["scheme"], protocol["repeats"], protocol["threshold_k"]) == ("stratified", 4, 1.25)
    assert protocol["sources"]["scheme"] == "role trace limits" and protocol["project_file"] is False
    chart = api(client, "/api/runs/c1/chart")["data"]
    assert chart["baseline"]["attempt_id"] == "p1"
    assert chart["bar"] == pytest.approx(0.7 - 0.02) and chart["threshold"] == 0.02
    # the opposite case: a higher score with a positive difference means maximise
    (tmp_path / "x").mkdir()
    path2 = _summary_for(tmp_path / "x", comparisons=[("p2", "p1", 0.1, 0.02, True)])
    body2 = api(client_for(path2, path2.parent), "/api/runs/c1")["data"]
    assert body2["protocol"]["direction"] == "max"
    assert body2["protocol"]["direction_source"] == "inferred from comparisons"
    chart2 = api(client_for(path2, path2.parent), "/api/runs/c1/chart")["data"]
    assert chart2["bar"] == pytest.approx(0.6 + 0.02)


def test_direction_falls_back_to_the_best_parameter_and_then_to_max(tmp_path):
    path = _summary_for(tmp_path)
    client = client_for(path, path.parent)
    default = api(client, "/api/runs/c1")["data"]
    assert default["protocol"]["direction"] == "max"
    assert default["protocol"]["direction_source"] == "default (max)"
    assert default["best"]["attempt_id"] == "p2"
    chosen = api(client, "/api/runs/c1?best=min")["data"]
    assert chosen["protocol"]["direction"] == "min" and chosen["protocol"]["direction_source"] == "best parameter"
    assert chosen["best"]["attempt_id"] == "p1"
    assert api(client, "/api/runs/c1?best=sideways")["data"]["protocol"]["direction"] == "max"
    assert default["protocol"]["scheme"] == "v" and default["protocol"]["sources"]["scheme"] == "plan"
    assert default["protocol"]["threshold_k"] is None and default["protocol"]["repeats"] is None


def test_the_project_file_is_read_tolerantly(tmp_path):
    path = _summary_for(tmp_path)
    folder = path.parent
    for content in ("not json", "[1, 2]", '{"protocol": "text"}', "{}"):
        (folder / "project.json").write_text(content, encoding="utf-8")
        body = api(client_for(path, folder), "/api/runs/c1")["data"]
        assert body["protocol"]["direction"] == "max"
    assert web_data.read_project_file(None) is None
    assert web_data.read_project_file(tmp_path / "missing") is None


def test_the_overview_uses_the_fixture_project_file(cockpit_client):
    protocol = api(cockpit_client, f"/api/runs/{RUN_A}")["data"]["protocol"]
    assert protocol["direction"] == "max" and protocol["direction_source"] == "project.json"
    assert protocol["threshold_k"] == PROJECT_JSON["protocol"]["threshold_k"] == 1.5
    assert protocol["hash"] == "protohash-run-a-0123456789" and protocol["sources"]["hash"] == "registry"
    limits = api(cockpit_client, f"/api/runs/{RUN_A}")["data"]["limits"]
    assert limits["rounds"] == 4 and limits["candidates"] == 2 and limits["max_fix_retries"] == 3
