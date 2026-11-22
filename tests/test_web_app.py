"""Pages of the cockpit over the small fixture registry of web_helpers (no network)."""

import re
import tomllib
from importlib import resources
from pathlib import Path

import pytest
from registry_helpers import make_attempt, make_cycle, make_execution
from web_helpers import (  # noqa: F401  (fixtures)
    CHILD,
    EMPTY,
    MAIN,
    PIPELINE_URL,
    SCRIPT,
    all_urls,
    db_path,
    no_network,
    page,
    registry,
    web_client,
)

from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web import create_app

ROOT = Path(__file__).resolve().parent.parent


def _after(text, marker):
    return text[text.index(marker):]


def test_runs_list_shows_every_cycle_with_its_data(web_client):
    html = page(web_client, "/")
    for cycle_id in (EMPTY, MAIN, CHILD):
        assert f'href="/cycles/{cycle_id}"' in html
    assert "proj-1" in html and "ITERATING" in html and "PLANNING" in html
    assert "3 of 12" in html and "0 of 10" in html


def test_empty_registry_shows_a_message():
    with SqliteRegistry() as reg:
        html = page(create_app(reg).test_client(), "/")
    assert "No project yet" in html and "Open Setup" in html  # ticket 27: setup mode


def test_overview_lists_the_rounds_in_the_chart_table_and_the_bundles(web_client):
    html = page(web_client, f"/cycles/{MAIN}")
    assert f'data-chart-url="/api/runs/{MAIN}/chart"' in html
    for attempt_id in ("A1", "A2", "B1", "X1"):
        assert f'href="/attempts/{attempt_id}"' in html
    assert "bundle-1" in html and "folderhash1" in html
    rounds = page(web_client, f"/cycles/{MAIN}/rounds")
    positions = [rounds.index(f">R{n}</a>") for n in (1, 2, 3)]
    assert positions == sorted(positions)
    for task in ("explore models", "check leakage", "combine models", "prune"):
        assert task in rounds


def test_round_view_shows_only_its_own_reports_guidance_and_outputs(web_client):
    r1 = page(web_client, f"/cycles/{MAIN}/rounds/R1")
    for text in ("explore models", "report one", "guidance one", "trees", "A3"):
        assert text in r1
    for text in ("report two", "blend", "combine models"):
        assert text not in r1
    r2 = page(web_client, f"/cycles/{MAIN}/rounds/R2")
    for text in ("report two", "blend", "B1", "B2"):
        assert text in r2
    for text in ("report one", "guidance one", "explore models"):
        assert text not in r2


def test_attempt_detail_shows_rationale_hypothesis_and_what_changed(web_client):
    a1 = page(web_client, "/attempts/A1")
    for text in ("because trees", "trees help", "add trees", "ev-1", "ev-2", "A1 concluded"):
        assert text in a1
    b1 = page(web_client, "/attempts/B1")
    assert 'href="/attempts/A2"' in b1 and "bundle-1" in b1 and "folderhash1" in b1
    assert 'href="/cycles/main/rounds/R2"' in b1


def test_attempt_detail_shows_the_runs_in_order(web_client):
    html = page(web_client, "/attempts/A1")
    assert html.index("A1-crash1") < html.index("A1-ok")
    for text in ("fix1", "fixed-code", ">11<", "lr", "cpu", "validation: mean 0.7",
                 "dispersion 0.02", "cv: mean 0.68", "Traceback: boom"):
        assert text in html


@pytest.mark.parametrize(
    "attempt_id, kind, reason",
    [("A3", "code_error", "Traceback: boom"), ("B2", "timeout", "ran out of time"),
     ("C1", "hypothesis_falsified", "the feature was useless")],
)
def test_failures_show_failure_kind_and_reason(web_client, attempt_id, kind, reason):
    html = page(web_client, f"/attempts/{attempt_id}")
    assert kind in html and reason in html
    attempts = page(web_client, f"/cycles/{MAIN}/attempts")
    row = re.search(rf"<tr [^>]*>(?:(?!</tr>).)*?>{attempt_id}</a>.*?</tr>", attempts, re.S).group(0)
    assert kind in row and "failed" in row


def test_pipeline_link_only_when_it_exists(web_client):
    with_link = page(web_client, "/attempts/B1")
    assert f'<a href="{PIPELINE_URL}"' in with_link
    without = page(web_client, "/attempts/A2")
    assert "no pipeline yet" in without and "example.test" not in without


def test_pipeline_reference_never_links_local_files(web_client):
    html = page(web_client, "/attempts/X1")
    assert "/etc/passwd" in html
    assert 'href="/etc/passwd"' not in html and 'href="pipeline' not in html


def test_the_best_follows_the_best_parameter_without_a_project_file(web_client):
    def best(query):
        return web_client.get(f"/api/runs/{MAIN}{query}").get_json()["data"]

    assert best("")["best"]["attempt_id"] == "B1"
    assert best("?best=min")["best"]["attempt_id"] == "X1"
    assert best("?best=min")["protocol"]["direction_source"] == "best parameter"
    assert best("")["protocol"]["direction_source"] == "default (max)"


def test_the_run_without_scores_has_an_empty_chart_state(web_client):
    html = page(web_client, f"/cycles/{EMPTY}")
    assert "No scored attempt yet" in html and "<table" not in html.split("Bundles")[0]


def test_reviewer_reports_and_guidance_appear_in_their_round(web_client):
    html = page(web_client, f"/cycles/{MAIN}/rounds/R1")
    assert "report one" in html and "guidance one: focus on trees" in html


def test_traces_page_shows_calls_and_typed_role_traces(web_client):
    html = page(web_client, f"/cycles/{MAIN}/traces")
    for text in ("coordinator", "modeling", "reviewer", "Goal", "Round", "ReviewRequest",
                 "invalid_output", 'href="/attempts/A1"'):
        assert text in html
    fragment = page(web_client, f"/cycles/{MAIN}/traces/1")
    assert "Input (Goal)" in fragment and "Output (Round)" in fragment


def test_trace_filters(web_client):
    by_role = page(web_client, f"/cycles/{MAIN}/traces?role=modeling")
    assert "ReviewRequest" not in by_role and "Goal" not in by_role
    by_round = page(web_client, f"/cycles/{MAIN}/traces?round=R2")
    assert "ReviewRequest" in by_round and ">R1</a>" not in by_round.split("Role traces")[1]
    bad = web_client.get(f"/cycles/{MAIN}/traces?role=banana")
    text = bad.get_data(as_text=True)
    assert bad.status_code == 400 and "role" in text and "Traceback" not in text
    assert web_client.get(f"/cycles/{MAIN}/traces?round=nope").status_code == 404


def test_guidance_and_suggestions(web_client):
    guidance = page(web_client, f"/cycles/{MAIN}/guidance")
    assert "guidance one: focus on trees" in guidance and "2026-10-30" in guidance
    assert 'href="/cycles/main/rounds/R1"' in guidance
    html = page(web_client, "/suggestions")
    assert "g1" in html and "guidance one: focus on trees" in html
    assert "S2" in html and "second suggestion text" in html
    assert 'href="/cycles/empty"' in html and 'href="/cycles/child"' in html
    assert html.count('href="/cycles/main"') >= 2


def test_every_page_route_responds_and_none_is_left_out(web_client):
    app = web_client.application
    endpoints = {
        r.endpoint for r in app.url_map.iter_rules()
        if r.endpoint != "static" and not r.endpoint.startswith(("api.", "writes."))
    }
    # ticket 27: the cockpit pages are covered by the cockpit tests, not by the registry fixture ids
    endpoints -= {"live_page", "audit_page", "setup", "run_control", "decisions_page", "environment_page",
                  "analytics_attempt", "analytics_compare", "analytics_paired",
                  # ticket 30: covered by the cockpit finish tests (they need a project folder)
                  "report_print", "bundle_page", "predict_page", "submission_page", "gap_page", "history_page",
                  "learnings_page", "fork_page"}
    urls = all_urls()
    assert endpoints == set(urls)
    for name, url in urls.items():
        response = web_client.get(url)
        assert response.status_code in (200, 301), (name, url)


def test_unknown_ids_give_a_styled_404(web_client):
    for url in ("/cycles/nope", "/attempts/nope", f"/cycles/{MAIN}/rounds/nope",
                "/cycles/nope/attempts", "/cycles/nope/report", "/cycles/nope/traces/1", "/elsewhere"):
        response = web_client.get(url)
        text = response.get_data(as_text=True)
        assert response.status_code == 404 and "Traceback" not in text, url
        assert "404" in text and 'class="error-state"' in text


def test_ids_with_a_slash_work_in_links_and_routes():
    with SqliteRegistry() as reg:
        reg.add_cycle(make_cycle("org/c1"))
        reg.add_attempt("org/c1", make_attempt("org/a1"), "fp")
        reg.add_execution(make_execution("org/a1"))
        reg.close_attempt("org/a1", "done")
        client = create_app(reg).test_client()
        assert 'href="/cycles/org/c1"' in page(client, "/")
        assert 'href="/attempts/org/a1"' in page(client, "/cycles/org/c1")
        assert "org/a1" in page(client, "/attempts/org/a1")
        page(client, "/cycles/org/c1/attempts")
        assert client.get("/api/runs/org/c1").get_json()["data"]["id"] == "org/c1"


def test_untrusted_registry_text_is_escaped(web_client):
    for url in ("/attempts/X1", f"/cycles/{MAIN}/traces/6", f"/cycles/{MAIN}/rounds/R1",
                f"/cycles/{MAIN}/attempts", "/attempts/A1", f"/cycles/{MAIN}/report"):
        html = page(web_client, url)
        assert SCRIPT not in html, url
        assert not re.search(r"<script(?![^>]*\ssrc=)", html), url
    assert "&lt;script&gt;" in page(web_client, "/attempts/X1")
    assert "&lt;script&gt;" in page(web_client, f"/cycles/{MAIN}/rounds/R1")
    assert "&lt;script&gt;" in page(web_client, f"/cycles/{MAIN}/traces/6")


def test_views_run_with_the_network_disabled(web_client):
    import socket

    with pytest.raises(AssertionError, match="network"):
        socket.socket().connect(("127.0.0.1", 9))
    for url in all_urls().values():
        assert web_client.get(url).status_code in (200, 301), url


def test_templates_are_package_resources_and_declared_in_package_data():
    templates = resources.files("kaggle_agent.web") / "templates"
    names = {t.name for t in templates.iterdir() if t.name.endswith(".html")}
    referenced = set()
    for module in ("app.py", "pages.py"):
        source = (resources.files("kaggle_agent.web") / module).read_text(encoding="utf-8")
        referenced |= set(re.findall(r'"(\w+\.html)"', source))
    for name in names:
        text = (templates / name).read_text(encoding="utf-8")
        referenced |= set(re.findall(r'(?:extends|from|include|import)\s+"(\w+\.html)"', text))
    assert referenced and referenced <= names
    assert {"base.html", "cycles.html", "cycle.html", "attempts.html", "rounds.html", "round.html",
            "traces.html", "trace_fragment.html", "report.html", "guidance.html", "attempt.html",
            "suggestions.html", "error.html"} <= names
    assert not {"metrics.html", "reports.html"} & names
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = pyproject["tool"]["setuptools"]
    assert setuptools["package-data"] == {"kaggle_agent.web": ["templates/*.html", "static/**"]}
    assert "include-package-data" not in setuptools
    assert setuptools["packages"] == {"find": {"where": ["src"]}}


def _web_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Web interface (`kaggle_agent.web`)")
    return readme[start: readme.index("\n## ", start + 1)]


def test_readme_documents_the_flask_pin_python_versions_and_unpinned_dependencies():
    section = _web_section()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pin = next(d for d in pyproject["project"]["dependencies"] if d.startswith("flask"))
    assert pin.startswith("flask==")
    assert f"Flask {pin.split('==')[1]}" in section
    assert "3.12" in section and "3.14" in section
    assert "not pinned" in section and "werkzeug" in section


def test_readme_has_the_single_start_command_in_a_code_block():
    section = _web_section()
    block = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
    assert block.strip() == "python -m kaggle_agent.web --db PATH"
    assert section.count("python -m kaggle_agent.web --db") == 1


def test_pipeline_reference_convention():
    from kaggle_agent.web.views import pipeline_reference

    assert pipeline_reference(["x", "pipeline:https://a.test/p"]) == {
        "ref": "https://a.test/p", "url": "https://a.test/p"}
    assert pipeline_reference(["pipeline:/etc/passwd"])["url"] is None
    assert pipeline_reference(["pipeline:javascript:alert(1)"])["url"] is None
    assert pipeline_reference(["pipeline:https://"])["url"] is None
    assert pipeline_reference(["other"]) is None and pipeline_reference([]) is None


def test_overview_shows_the_submission_and_its_score():
    from registry_helpers import make_cycle

    from kaggle_agent.registry import SubmissionEntry

    with SqliteRegistry() as reg:
        reg.add_cycle(make_cycle("S1"))
        reg.add_submission(SubmissionEntry("S1", "toy", "sub-7", "a" * 64, "msg", "2026-01-01T12:00:00+00:00",
                                           "pending"))
        reg.update_submission_score("toy", "sub-7", "complete", 0.8125, "2026-01-01T12:05:00+00:00")
        client = create_app(reg).test_client()
        html = page(client, "/cycles/S1")
        assert client.post("/cycles/S1").status_code == 405
    block = _after(html, "<h2>Submissions</h2>")
    assert "sub-7" in block and "complete" in block and "0.8125" in block and "2026-01-01T12:00:00" in block
