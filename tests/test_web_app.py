"""Views of the read-only web interface, rendered from a fixture registry (no network)."""

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


# 1, 2
def test_cycles_list_shows_every_cycle_with_its_data(web_client):
    html = page(web_client, "/")
    for cycle_id in (EMPTY, MAIN, CHILD):
        assert f'href="/cycles/{cycle_id}"' in html
    assert "proj-1" in html and "ITERATING" in html and "PLANNING" in html
    assert ">12<" in html and ">4<" in html


def test_empty_registry_shows_a_message():
    with SqliteRegistry() as reg:
        html = page(create_app(reg).test_client(), "/")
    assert "No cycles" in html


# 3
def test_cycle_view_lists_rounds_candidates_and_the_group_outside_rounds(web_client):
    html = page(web_client, f"/cycles/{MAIN}")
    positions = [html.index(f"Round {n}: R{n}") for n in (1, 2, 3)]
    assert positions == sorted(positions)
    for task in ("explore models", "check leakage", "combine models", "prune"):
        assert task in html
    outside = html.index("Not in a round")
    timeline = html.index("Attempt timeline")
    for attempt_id in ("A1", "A2", "A3", "B1", "B2", "C1"):
        assert html.index(f'href="/attempts/{attempt_id}"') < outside
    assert outside < html.index('href="/attempts/X1"') < timeline
    row_b1 = re.search(r"<tr>(?:(?!</tr>).)*?>B1</a>.*?</tr>", html, re.S).group(0)
    assert 'href="/attempts/A2"' in row_b1 and ">4<" in row_b1
    assert "<td>-</td>" in re.search(r"<tr>(?:(?!</tr>).)*?>A1</a>.*?</tr>", html, re.S).group(0)


# 4
def test_round_view_shows_only_its_own_reports_guidance_and_traces(web_client):
    r1 = page(web_client, f"/cycles/{MAIN}/rounds/R1")
    for text in ("explore models", "report one", "guidance one", "trees", "retry", "A3"):
        assert text in r1
    for text in ("report two", "blend", "combine models"):
        assert text not in r1
    r2 = page(web_client, f"/cycles/{MAIN}/rounds/R2")
    for text in ("report two", "blend", "B1", "B2"):
        assert text in r2
    for text in ("report one", "guidance one", "explore models"):
        assert text not in r2


# 5
def test_candidate_detail_shows_rationale_hypothesis_and_what_changed(web_client):
    a1 = page(web_client, "/attempts/A1")
    for text in ("because trees", "trees help", "add trees", "ev-1", "ev-2", "A1 concluded"):
        assert text in a1
    b1 = page(web_client, "/attempts/B1")
    assert 'href="/attempts/A2"' in b1 and "bundle-1" in b1 and "folderhash1" in b1
    assert 'href="/cycles/main/rounds/R2"' in b1


# 6
def test_candidate_detail_shows_the_runs_in_order(web_client):
    html = page(web_client, "/attempts/A1")
    assert html.index("A1-crash1") < html.index("A1-ok")
    for text in ("fix1", "fixed-code", ">11<", ">1.5<", "lr", "cpu", "validation: mean 0.7",
                 "dispersion 0.02", "cv: mean 0.68", "Traceback: boom"):
        assert text in html
    assert "Artifacts" in page(web_client, "/attempts/B1")


# 7
@pytest.mark.parametrize(
    "attempt_id, kind, reason",
    [("A3", "code_error", "Traceback: boom"), ("B2", "timeout", "ran out of time"),
     ("C1", "hypothesis_falsified", "the feature was useless")],
)
def test_failures_show_failure_kind_and_reason(web_client, attempt_id, kind, reason):
    html = page(web_client, f"/attempts/{attempt_id}")
    assert kind in html and reason in html
    cycle = page(web_client, f"/cycles/{MAIN}")
    row = re.search(rf"<tr>(?:(?!</tr>).)*?>{attempt_id}</a>.*?</tr>", cycle, re.S).group(0)
    assert kind in row and "failed" in row


# 8, 9
def test_pipeline_link_only_when_it_exists(web_client):
    with_link = page(web_client, "/attempts/B1")
    assert f'<a href="{PIPELINE_URL}">' in with_link
    without = page(web_client, "/attempts/A2")
    assert "no pipeline yet" in without and "example.test" not in without


def test_pipeline_reference_never_links_local_files(web_client):
    html = page(web_client, "/attempts/X1")
    assert "/etc/passwd" in html
    assert 'href="/etc/passwd"' not in html and 'href="pipeline' not in html
    section = _after(html, "<h2>Pipeline</h2>")
    assert "<a " not in section[: section.index("<h2>Runs</h2>")]


# 10, 11, 12
def test_metric_series_per_round_and_chart(web_client, registry):
    html = page(web_client, f"/cycles/{MAIN}/metrics?partition=validation")
    assert html.count("<circle") == 3
    series = registry.metric_series_by_round(MAIN, "validation")
    table = _after(html, "<h2>Per round</h2>")
    table = table[: table.index("<h2>Per iteration</h2>")]
    for r in series:
        for p in r.points:
            assert f">{p.attempt_id}</a>" in table and f"{p.mean:.6g}" in table
    for absent in ("A3", "B2", "C1"):
        assert f">{absent}</a>" not in table
    assert "partition=cv" in html
    cv = page(web_client, f"/cycles/{MAIN}/metrics?partition=cv")
    assert cv.count("<circle") == 2


def test_best_is_highlighted_by_the_declared_direction(web_client):
    def best(query):
        html = page(web_client, f"/cycles/{MAIN}/metrics?partition=validation{query}")
        return re.findall(r'class="point best"[^>]*>\s*<title>(\w+):', html), html

    assert best("&best=max")[0] == ["B1"]
    assert best("&best=min")[0] == ["A1"]
    default, html = best("")
    assert default == ["B1"]
    assert "not stored in the registry" in html and "best=min" in html
    empty = page(web_client, f"/cycles/{EMPTY}/metrics")
    assert "No metric data" in empty and "<svg" not in empty


def test_metric_series_per_iteration(web_client, registry):
    html = page(web_client, f"/cycles/{MAIN}/metrics?partition=validation")
    table = _after(html, "<h2>Per iteration</h2>")
    series = registry.metric_series(MAIN, "validation")
    assert [p.iteration for p in series] == [1, 2, 4, 7]
    for p in series:
        assert f">{p.attempt_id}</a>" in table and f"<td>{p.iteration}</td>" in table
        assert f"{p.mean:.6g}" in table and f"{p.dispersion:.4g}" in table


# 13
def test_reviewer_reports_grouped_by_round(web_client):
    html = page(web_client, f"/cycles/{MAIN}/reports")
    assert html.index("report one") < html.index("report two text")
    assert 'href="/cycles/main/rounds/R1"' in html and 'href="/cycles/main/rounds/R2"' in html
    assert html.count("2026-10-30T10:00:00+00:00") == 2


# 14
def test_role_traces_show_typed_input_output_and_outcome(web_client):
    html = page(web_client, f"/cycles/{MAIN}/traces")
    for text in ("coordinator", "modeling", "reviewer", "Input (Goal)", "Output (Round)",
                 "Input (ReviewRequest)", "no output", "invalid_output",
                 "LLM call id</th><td>1</td>", 'href="/attempts/A1"', "start"):
        assert text in html


# 15
def test_trace_filters(web_client):
    by_role = page(web_client, f"/cycles/{MAIN}/traces?role=modeling")
    assert "trees" in by_role and "blend" in by_role and "ReviewRequest" not in by_role
    by_round = page(web_client, f"/cycles/{MAIN}/traces?round=R2")
    assert "blend" in by_round and "ReviewRequest" in by_round and "trees" not in by_round
    both = page(web_client, f"/cycles/{MAIN}/traces?role=modeling&round=R2")
    assert "blend" in both and "retry" not in both
    bad = web_client.get(f"/cycles/{MAIN}/traces?role=banana")
    text = bad.get_data(as_text=True)
    assert bad.status_code == 400 and "role" in text and "Traceback" not in text
    assert web_client.get(f"/cycles/{MAIN}/traces?round=nope").status_code == 404


# 16
def test_candidate_traces_appear_in_the_detail(web_client):
    a1 = page(web_client, "/attempts/A1")
    assert "trees" in a1 and "retry" in a1 and "blend" not in a1
    assert "blend" in page(web_client, "/attempts/B1")


# 17
def test_guidance_and_suggestions(web_client):
    guidance = page(web_client, f"/cycles/{MAIN}/guidance")
    assert "guidance one: focus on trees" in guidance and "2026-10-30" in guidance
    assert 'href="/cycles/main/rounds/R1"' in guidance
    html = page(web_client, "/suggestions")
    assert "g1" in html and "guidance one: focus on trees" in html
    assert "S2" in html and "second suggestion text" in html
    assert 'href="/cycles/empty"' in html and 'href="/cycles/child"' in html
    assert html.count('href="/cycles/main"') >= 2


# 18
def test_every_view_responds_and_none_is_left_out(web_client):
    app = web_client.application
    endpoints = {r.endpoint for r in app.url_map.iter_rules() if r.endpoint != "static"}
    urls = all_urls()
    assert endpoints == set(urls)
    for url in urls.values():
        assert "<h1>" in page(web_client, url)


# 19
def test_unknown_ids_give_a_plain_404(web_client):
    for url in ("/cycles/nope", "/attempts/nope", f"/cycles/{MAIN}/rounds/nope",
                "/cycles/nope/metrics", "/cycles/nope/reports", "/elsewhere"):
        response = web_client.get(url)
        text = response.get_data(as_text=True)
        assert response.status_code == 404 and "Traceback" not in text, url
        assert "<h1>404" in text


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
        page(client, "/cycles/org/c1/metrics")


# 20
def test_untrusted_registry_text_is_escaped(web_client):
    for url in ("/attempts/X1", f"/cycles/{MAIN}/reports", f"/cycles/{MAIN}/traces",
                f"/cycles/{MAIN}/rounds/R1", f"/cycles/{MAIN}/metrics", "/attempts/A1"):
        html = page(web_client, url)
        assert SCRIPT not in html and "<script" not in html, url
    assert "&lt;script&gt;" in page(web_client, "/attempts/X1")
    assert "&lt;script&gt;" in page(web_client, f"/cycles/{MAIN}/reports")
    assert "&lt;script&gt;" in page(web_client, f"/cycles/{MAIN}/traces")


# 30
def test_views_run_with_the_network_disabled(web_client):
    import socket

    with pytest.raises(AssertionError, match="network"):
        socket.socket().connect(("127.0.0.1", 9))
    for url in all_urls().values():
        page(web_client, url)


# 34
def test_templates_are_package_resources_and_declared_in_package_data():
    templates = resources.files("kaggle_agent.web") / "templates"
    names = {t.name for t in templates.iterdir() if t.name.endswith(".html")}
    referenced = set()
    app_source = (resources.files("kaggle_agent.web") / "app.py").read_text(encoding="utf-8")
    referenced |= set(re.findall(r'"(\w+\.html)"', app_source))
    for name in names:
        text = (templates / name).read_text(encoding="utf-8")
        referenced |= set(re.findall(r'(?:extends|from|include|import)\s+"(\w+\.html)"', text))
    assert referenced and referenced <= names
    assert {"base.html", "cycles.html", "cycle.html", "round.html", "metrics.html",
            "reports.html", "traces.html", "guidance.html", "attempt.html",
            "suggestions.html", "error.html"} <= names
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = pyproject["tool"]["setuptools"]
    assert setuptools["package-data"] == {"kaggle_agent.web": ["templates/*.html"]}
    assert "include-package-data" not in setuptools
    assert setuptools["packages"] == {"find": {"where": ["src"]}}
    assert "py-modules" not in setuptools


# 35, 36
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
