"""One HTML test per page of the cockpit, the redirects, and the untrusted-text rules (no network)."""

import hashlib
import re
from pathlib import Path

import pytest
from cockpit_helpers import (  # noqa: F401  (fixtures)
    RUN_A,
    RUN_B,
    SCRIPT,
    SpyRegistry,
    all_page_urls,
    api,
    client_for,
    cockpit_client,
    cockpit_db,
    html_tags,
    loopback_only,
    page,
    unseal,
)
from registry_helpers import make_attempt, make_cycle, make_execution

from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web import create_app

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "src" / "kaggle_agent" / "web" / "static"
READ_PREFIXES = ("list_", "get_", "metric_series", "cycles_of_suggestion", "count_")


def row_of(html: str, needle: str) -> str:
    """The <tr> that contains ``needle``."""
    for match in re.finditer(r"<tr\b.*?</tr>", html, re.S):
        if needle in match.group(0):
            return match.group(0)
    raise AssertionError(f"no row with {needle!r}")


def text_of(html: str) -> str:
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    return re.sub(r" ([,;])", r"\1", flat)


# 11
def test_the_shell_has_navigation_status_bar_ticker_skip_link_and_landmarks(cockpit_client):
    for url in all_page_urls():
        html = page(cockpit_client, url)
        assert 'class="skip-link" href="#main"' in html, url
        assert '<nav class="side-nav" aria-label="Main">' in html and "<main " in html
        assert '<header class="status-bar"' in html
        assert re.search(r'<div id="live-ticker"[^>]*data-stream-url="/api/events/stream"[^>]*>[^<]*</div>', html), url  # live since ticket 27
        assert f'href="/"' in html and "Runs" in html
    html = page(cockpit_client, f"/cycles/{RUN_A}")
    for label, path in (("Overview", f"/cycles/{RUN_A}"), ("Rounds", f"/cycles/{RUN_A}/rounds"),
                        ("Attempts", f"/cycles/{RUN_A}/attempts"), ("LLM trace", f"/cycles/{RUN_A}/traces"),
                        ("Report and submissions", f"/cycles/{RUN_A}/report"),
                        ("Guidance", f"/cycles/{RUN_A}/guidance")):
        assert re.search(rf'<a href="{path}"[^>]*>{label}', html), label
    bar = html[html.index('<header class="status-bar"'): html.index("</header>")]
    assert "proj-1" in bar and RUN_A in bar and "STOPPED" in bar and "status-succeeded" in bar
    assert 'aria-current="page"' in html
    tokens = (STATIC / "css" / "tokens.css").read_text(encoding="utf-8")
    app_css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    assert 'data-theme="dark"' in tokens
    for status in ("succeeded", "failed", "running", "waiting", "suspended"):
        assert f"--st-{status}-fg" in tokens and f"--st-{status}-bg" in tokens
        assert f".status-{status}" in app_css


# 14
def test_empty_and_error_states(tmp_path):
    empty = tmp_path.resolve() / "empty.db"
    SqliteRegistry(empty).close()
    client = client_for(empty)
    assert "No project yet" in page(client, "/")  # ticket 27: the empty list points to Setup
    assert "No suggestions" in page(client, "/suggestions")
    assert page(client, "/cycles/nope", 404).count('class="error-state"') == 1
    # a cycle without rounds or attempts
    bare = tmp_path.resolve() / "bare.db"
    with SqliteRegistry(bare) as reg:
        reg.add_cycle(make_cycle("bare", state="PLANNING"))
    client = client_for(bare)
    assert "No rounds yet" in page(client, "/cycles/bare/rounds")
    assert "No attempts yet" in page(client, "/cycles/bare/attempts")
    assert "No scored attempt yet" in page(client, "/cycles/bare")
    assert "No final report yet" in page(client, "/cycles/bare/report")
    assert "No guidance" in page(client, "/cycles/bare/guidance")
    assert "No LLM calls" in page(client, "/cycles/bare/traces")
    assert "No role traces" in page(client, "/cycles/bare/traces")


# 14
def test_unknown_ids_and_bad_filters_give_styled_errors_never_500(cockpit_client):
    for url in ("/cycles/nope", "/attempts/nope", f"/cycles/{RUN_A}/rounds/nope", "/cycles/nope/traces",
                f"/cycles/{RUN_A}/traces/99", "/no/such/page"):
        html = page(cockpit_client, url, 404)
        assert 'class="error-state"' in html and "Back to the runs" in html and "Traceback" not in html
    for url in (f"/cycles/{RUN_A}/traces?role=banana", f"/cycles/{RUN_A}/attempts?min_score=x",
                f"/cycles/{RUN_A}/attempts?outcome=banana"):
        html = page(cockpit_client, url, 400)
        assert "Invalid request" in html and "Traceback" not in html
    assert page(cockpit_client, f"/cycles/{RUN_A}/traces?round=nope", 404)


# 15
def test_runs_list_shows_one_row_per_cycle_with_its_figures(cockpit_client):
    html = page(cockpit_client, "/")
    a, b = row_of(html, f'href="/cycles/{RUN_A}"'), row_of(html, f'href="/cycles/{RUN_B}"')
    text_a, text_b = text_of(a), text_of(b)
    assert "proj-1" in text_a and "proj-1" in text_b
    assert "m, v" in text_a  # plan summary: metric and validation, then the models
    assert "2 of 4" in text_a and "0 of 5" in text_b
    assert "0.72" in text_a and "0.5" in text_b  # best score
    assert "a1-b1" in text_a  # chosen model (the bundle)
    assert "status-succeeded" in a and "status-running" in b
    assert "complete, public 0.68" in text_a and "complete" not in text_b
    # ticket 30: the holdout number shows only after the DS unsealed it
    assert "0.69" not in text_a and "sealed" in text_a and "0.69" not in text_b and "sealed" in text_b
    unseal(cockpit_client.application.config["PROJECT_DIR"] / "registry.db")
    unsealed = text_of(row_of(page(cockpit_client, "/"), f'href="/cycles/{RUN_A}"'))
    assert "0.69" in unsealed
    assert re.search(r'<input type="search"[^>]*data-filter-input', html)


# 16
def test_overview_shows_chart_container_fallback_table_protocol_and_limits(cockpit_client):
    chart = api(cockpit_client, f"/api/runs/{RUN_A}/chart")["data"]
    assert chart["baseline"] == {"attempt_id": "a-base", "mean": 0.6, "dispersion": 0.02}
    assert chart["threshold"] == 0.03 and chart["bar"] == pytest.approx(0.63)
    assert chart["chosen_attempt_id"] == "a3" and chart["direction"] == "max"
    by_id = {p["attempt_id"]: p for p in chart["points"]}
    assert set(by_id) == {"a-base", "a1", "a3"}  # failed attempts have no score
    assert (by_id["a1"]["mean"], by_id["a1"]["dispersion"], by_id["a1"]["outcome"]) == (0.7, 0.02, "succeeded")
    assert by_id["a3"]["chosen"] and not by_id["a1"]["chosen"] and by_id["a-base"]["is_baseline"]
    assert by_id["a1"]["is_real"] is True and by_id["a3"]["is_real"] is False
    html = page(cockpit_client, f"/cycles/{RUN_A}")
    assert f'data-chart-url="/api/runs/{RUN_A}/chart"' in html and 'id="score-chart"' in html
    table = html[html.index("Table of the same numbers"):]
    table = table[: table.index("</table>")]
    for aid in ("a-base", "a1", "a3"):
        assert f">{aid}</a>" in table
    assert "0.72" in table and "0.02" in table and "chosen" in table
    text = text_of(html)
    assert "stagnation" in text
    worked = html[html.index('id="worked-h"'): html.index('id="failed-h"')]
    failed = html[html.index('id="failed-h"'): html.index("<h2>Bundles")]
    assert ">a1</a>" in worked and ">a2</a>" in failed and ">a4</a>" in failed
    assert "the feature was useless" in failed
    proto = html[html.index('id="proto-h"'): html.index('id="limits-h"')]
    proto_text = text_of(proto)
    for expected in ("accuracy (project.json)", "kfold (project.json)", "3 (project.json)",
                     "1.5 (project.json)", "42 (project.json)", "protohash-ru"):
        assert expected in proto_text, expected
    limits = text_of(html[html.index('id="limits-h"'): html.index('id="plan-h"')])
    assert "rounds 4" in limits and "stagnation_k 2" in limits and "candidates 2" in limits
    assert "Improvement bar" in html and "0.63" in html and "Baseline" in html
    assert "/abs/path/&lt;b&gt;x&lt;/b&gt;/bundle" in html and "<b>x</b>" not in html


# 18
def test_round_view_shows_tasks_role_outputs_report_verdicts_and_guidance(cockpit_client):
    html = page(cockpit_client, f"/cycles/{RUN_A}/rounds/run-a-R1")
    text = text_of(html)
    for expected in ("explore trees", "check leakage", "no leakage found", "scale", "encode",
                     "focus on trees", "Data findings", "Feature engineering plan", "Modeling plan"):
        assert expected in text, expected
    # the reviewer Markdown is an escaped data node for markdown.js
    node = re.search(r'<div class="md" data-markdown>(.*?)</div>', html, re.S).group(1)
    assert "# Round 1" in node and "- trees **help**" in node
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in node and "<script>alert(1)" not in html
    candidates = html[html.index("<h2>Candidates</h2>"): html.index("Data findings")]
    a1, a2 = row_of(candidates, ">a1</a>"), row_of(candidates, ">a2</a>")
    assert "real improvement" in a1 and "0.07" in text_of(a1) and "0.7" in text_of(a1)
    assert "failed" in a2 and "hypothesis_falsified" in a2
    r2 = text_of(page(cockpit_client, f"/cycles/{RUN_A}/rounds/run-a-R2"))
    assert "not a real improvement" in r2 and "-0.01" in r2 and "focus on trees" not in r2
    data = api(cockpit_client, f"/api/runs/{RUN_A}/rounds/run-a-R1")["data"]
    comparison = next(c for c in data["candidates"] if c["attempt_id"] == "a1")["comparison"]
    assert comparison["vs_bar"] == pytest.approx(0.07) and comparison["is_real"] is True


# 19
def test_attempt_page_shows_every_field_and_no_link_to_files(cockpit_client):
    html = page(cockpit_client, "/attempts/a1")
    text = text_of(html)
    for expected in ("trees help", "add trees", "ev-1", "ev-2", "+0.05", "accuracy", "overfit",
                     "code_error", "a1-crash1", "a1-ok", "code-a1", "data-a1", "numpy 2.5.4",
                     "pandas 3.0.6", "a1-b1", "folderhash-a1", "Retries", "Seed", "3.5", "11"):
        assert expected in text, expected
    assert "0.7" in text and "0.02" in text and "succeeded" in text
    tags = html_tags(html)
    assert any(t == "a" and a.get("href") == "/attempts/a-base" for t, a in tags)  # the parent
    assert any(t == "a" and a.get("href") == f"/cycles/{RUN_A}/rounds/run-a-R1" for t, a in tags)
    assert any(t == "a" and a.get("href", "").startswith(f"/cycles/{RUN_A}/traces?q=a1") for t, a in tags)
    # the bundle reference is escaped text, never an URL
    assert "/abs/path/&lt;b&gt;x&lt;/b&gt;/bundle" in html and "<b>x</b>" not in html
    for name, attrs in tags:
        for key, value in attrs.items():
            if key in ("href", "src", "action", "data-url", "data-chart-url", "hx-get"):
                assert "/abs/path" not in (value or ""), (name, key)
                if value == f"/cycles/{RUN_A}/bundle":  # ticket 30: the page of the persisted bundle (no code, no file path)
                    continue
                assert not re.search(r"bundles?|/code|diff|/files?\b|workspace", value or "", re.I), (key, value)
    assert "because &lt;script&gt;" in html
    other = text_of(page(cockpit_client, "/attempts/a4"))
    assert "code_error" in other and "Traceback: boom" in other


# 20
def test_llm_trace_viewer_groups_calls_per_role_and_opens_stored_traces(cockpit_client):
    html = page(cockpit_client, f"/cycles/{RUN_A}/traces")
    text = text_of(html)
    for expected in ("modeling round 1", "reviewer round 1", "data round 1", "invalid_output", "bad json",
                     "100", "50", "2.5", "m-1"):
        assert expected in text, expected
    digest = hashlib.sha256(b"modeling round 1").hexdigest()[:12]
    assert digest in html
    assert re.search(r'<input type="search" name="q"', html)
    assert [c["role"] for c in api(cockpit_client, f"/api/runs/{RUN_A}/llm-calls")["data"]] == [
        "modeling", "reviewer", "data"]
    filtered = page(cockpit_client, f"/cycles/{RUN_A}/traces?role=modeling")
    assert "modeling round 1" in filtered and "reviewer round 1" not in filtered
    assert "data round 1" not in filtered
    searched = page(cockpit_client, f"/cycles/{RUN_A}/traces?q=bad+json")
    assert "reviewer round 1" in searched and "modeling round 1" not in searched
    # the stored trace opens as a fragment for htmx, and as a full page without the header
    fragment = cockpit_client.get(f"/cycles/{RUN_A}/traces/3", headers={"HX-Request": "true"})
    body = fragment.get_data(as_text=True)
    assert fragment.status_code == 200 and "<html" not in body and "side-nav" not in body
    assert "Input (Round)" in body and "&#34;task&#34;: &#34;trees&#34;" in body
    assert "Output (Attempt)" in body and "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<script>" not in body
    full = page(cockpit_client, f"/cycles/{RUN_A}/traces/3")
    assert "<html" in full and "side-nav" in full and "&#34;task&#34;: &#34;trees&#34;" in full
    nothing = cockpit_client.get(f"/cycles/{RUN_A}/traces/5", headers={"HX-Request": "true"})
    assert "no output" in nothing.get_data(as_text=True)
    # the open links use htmx
    assert 'hx-get="/cycles/run-a/traces/3"' in html


# 21
def test_report_page_with_holdout_submissions_suggestions_and_guidance(cockpit_client):
    html = page(cockpit_client, f"/cycles/{RUN_A}/report")
    text = text_of(html)
    for expected in ("stagnation", "accuracy", "tune depth", "code_error", "try a *linear* blend",
                     "sub-1", "0.68", "complete", "a3", "Other fields", "Final report"):
        assert expected in text or expected in html, expected
    # ticket 30: sealed until the DS unseals it: no score, no row count anywhere on the page
    panel = text_of(html[html.index('id="holdout-h"'): html.index('id="report-h"')])
    assert "sealed" in panel and "0.69" not in text and "25 rows" not in text
    holdout = api(cockpit_client, f"/api/runs/{RUN_A}/holdout")["data"]
    assert holdout["state"] == "ready" and holdout["status"] == "sealed"
    assert holdout["score"] is None and holdout["rows"] is None and holdout["reads"] == []
    unseal(cockpit_client.application.config["PROJECT_DIR"] / "registry.db")
    html = page(cockpit_client, f"/cycles/{RUN_A}/report")
    assert "0.69" in text_of(html) and "25" in text_of(html)
    holdout = api(cockpit_client, f"/api/runs/{RUN_A}/holdout")["data"]
    assert holdout["status"] == "succeeded" and holdout["score"] == 0.69 and len(holdout["reads"]) == 1
    assert holdout["reads"][0]["attempt_id"] == "a3"
    guidance = page(cockpit_client, f"/cycles/{RUN_A}/guidance")
    assert "focus on trees" in guidance and "run-a-R1-g" in guidance
    suggestions = page(cockpit_client, "/suggestions")
    assert "try a linear blend" in suggestions and f'href="/cycles/{RUN_B}"' in suggestions
    # a run without a holdout read
    sealed = page(cockpit_client, f"/cycles/{RUN_B}/report")
    holdout_panel = sealed[sealed.index('id="holdout-h"'): sealed.index('id="report-h"')]
    assert "sealed" in holdout_panel and not re.search(r"\d\.\d", text_of(holdout_panel))
    assert "0.69" not in sealed and "No final report yet" in sealed
    data = api(cockpit_client, f"/api/runs/{RUN_B}/holdout")["data"]
    assert data["status"] == "sealed" and data["score"] is None and data["reads"] == []


# 21
def test_no_page_computes_or_fetches_a_score(cockpit_db):
    with SqliteRegistry(cockpit_db, read_only=True) as inner:
        spy = SpyRegistry(inner)
        client = create_app(spy, project_dir=cockpit_db.parent).test_client()
        for run in (RUN_A, RUN_B):
            page(client, f"/cycles/{run}/report")
            api(client, f"/api/runs/{run}/holdout")
            api(client, f"/api/runs/{run}/final-report")
        assert spy.calls and all(n.startswith(READ_PREFIXES) for n in spy.calls), spy.calls
        assert not {n for n in spy.calls if "holdout" in n and not n.startswith(("get_", "list_"))}


# 22
def test_lists_have_filter_controls_and_rows_carry_data_attributes(cockpit_client):
    runs = page(cockpit_client, "/")
    assert 'data-filter-for="runs-table"' in runs and 'data-filter="status"' in runs
    assert re.search(r'<tr data-status="succeeded" data-text="[^"]*run-a', runs)
    attempts = page(cockpit_client, f"/cycles/{RUN_A}/attempts")
    for control in ('data-filter="text"', 'data-filter="outcome"', 'data-filter="round"',
                    'data-filter="min"', 'data-filter="max"'):
        assert control in attempts, control
    row = row_of(attempts, ">a1</a>")
    assert 'data-outcome="succeeded"' in row and 'data-round="run-a-R1"' in row
    assert 'data-score="0.7"' in row and 'data-text="a1 trees help' in row
    assert 'data-outcome="failed"' in row_of(attempts, ">a2</a>")
    assert 'data-score=""' in row_of(attempts, ">a2</a>")
    traces = page(cockpit_client, f"/cycles/{RUN_A}/traces")
    assert 'data-role="modeling"' in traces and 'data-filter="role"' in traces
    assert re.search(r'<script src="/static/js/app.js"', traces)
    script = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    for token in ("data-filter-for", "data-filter-count", "data-score", "data-text", "aria-sort"):
        assert token in script
    assert cockpit_client.get("/static/js/app.js").status_code == 200
    # server side filters use the same names, with the page rendering only the matches
    only = page(cockpit_client, f"/cycles/{RUN_A}/attempts?outcome=failed")
    assert ">a2</a>" in only and ">a1</a>" not in only


# 28
def test_untrusted_text_is_escaped_and_never_becomes_a_tag_or_a_script_link(cockpit_client):
    urls = [f"/cycles/{RUN_A}/rounds/run-a-R1", f"/cycles/{RUN_A}/traces/4", f"/cycles/{RUN_A}/traces/3",
            "/attempts/a1", f"/cycles/{RUN_A}/report", f"/cycles/{RUN_A}/attempts"]
    for url in urls:
        html = page(cockpit_client, url)
        assert SCRIPT not in html, url
        for name, attrs in html_tags(html):
            assert name != "img" or "onerror" not in attrs, url
            assert not [a for a in attrs if a.startswith("on")], (url, name)
            if name == "script":
                assert "src" in attrs, url
            for key in ("href", "src"):
                assert not (attrs.get(key) or "").lower().startswith("javascript:"), url
    assert "&lt;script&gt;" in page(cockpit_client, "/attempts/a1")
    round_html = page(cockpit_client, f"/cycles/{RUN_A}/rounds/run-a-R1")
    assert "&lt;img src=x onerror=alert(1)&gt;" in round_html
    assert "[bad](javascript:alert(1))" in round_html  # raw Markdown text, rendered by markdown.js
    fragment = cockpit_client.get(f"/cycles/{RUN_A}/traces/4", headers={"HX-Request": "1"}).get_data(as_text=True)
    assert "&lt;script&gt;" in fragment and "<script" not in fragment


# 33
def test_old_metrics_and_reports_urls_redirect_with_301(tmp_path):
    path = tmp_path.resolve() / "slash.db"
    with SqliteRegistry(path) as reg:
        for cycle_id in ("org/c1", "plain"):
            reg.add_cycle(make_cycle(cycle_id))
            reg.add_attempt(cycle_id, make_attempt(f"{cycle_id}-a"), f"fp-{cycle_id}")
            reg.add_execution(make_execution(f"{cycle_id}-a"))
            reg.close_attempt(f"{cycle_id}-a", "done")
    client = client_for(path)
    for cycle_id in ("org/c1", "plain"):
        metrics = client.get(f"/cycles/{cycle_id}/metrics")
        reports = client.get(f"/cycles/{cycle_id}/reports")
        assert metrics.status_code == 301 and metrics.headers["Location"] == f"/cycles/{cycle_id}"
        assert reports.status_code == 301 and reports.headers["Location"] == f"/cycles/{cycle_id}/report"
        overview = client.get(f"/cycles/{cycle_id}/metrics", follow_redirects=True)
        report = client.get(f"/cycles/{cycle_id}/reports", follow_redirects=True)
        assert overview.status_code == 200 and "Score by attempt" in overview.get_data(as_text=True)
        assert report.status_code == 200 and "Final report" in report.get_data(as_text=True)
    for suffix in ("metrics", "reports"):
        gone = client.get(f"/cycles/nope/{suffix}", follow_redirects=True)
        assert gone.status_code == 404 and 'class="error-state"' in gone.get_data(as_text=True)
