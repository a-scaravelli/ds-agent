"""README 'Cockpit', API list, design note, process memory, reversal of ticket 24 and the test map."""

import ast
import json
import re
from pathlib import Path

from cockpit_helpers import cockpit_client, cockpit_db, loopback_only  # noqa: F401  (fixtures)

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PLAN_CASES = 7


def normal(rule: str) -> str:
    names = {"run_id": "<id>", "attempt_id": "<id>", "round_id": "<round>", "n": "<n>", "job_id": "<id>"}
    return re.sub(r"<(?:path:|int:)?(\w+)>", lambda m: names[m.group(1)], rule)


# 50
def test_the_readme_has_the_cockpit_section_with_command_pages_redirects_and_every_api_route(cockpit_client):
    section = README.split("## Cockpit")[1].split("\n## ")[0]
    assert "python -m kaggle_agent.web --db" in section and "--project" in section
    assert "301" in section and "/cycles/<id>/metrics" in section and "/cycles/<id>/reports" in section
    assert "no `--workspace` option" in section and "no code view" in section
    assert "WAL" in section and "Content-Security-Policy" in section and "VENDORED.md" in section
    routes = {normal(r.rule) for r in cockpit_client.application.url_map.iter_rules()
              if r.rule.startswith("/api/")}
    assert len(routes) >= 20
    for route in routes:
        assert f"`{route}`" in section or f"`{route}` " in section, route
    for url in ("/", "/cycles/<id>", "/cycles/<id>/attempts", "/attempts/<id>", "/cycles/<id>/report",
                "/cycles/<id>/traces", "/cycles/<id>/guidance", "/suggestions"):
        assert f"`{url}`" in section, url
    for title in ("Run control", "Single writer", "Files in the project folder", "Live progress", "Decisions",
                  "Closed command list", "Audit, notes and tags"):
        assert title in section, title
    for word in ("cockpit/", "events.jsonl", "audit.jsonl", "decisions.jsonl", "limit-changes.jsonl", "run.lock",
                 "`--supervised`", "SIGTERM", "Last-Event-ID", "`orchestrator_argv`", "`claude_executable`",
                 "`ps -o lstart= -p"):
        assert word in section, word
    running = README.split("## Running a competition")[1].split("\n## ")[0]
    assert "14" in running and "15" in running and "SIGTERM" in running
    web = README.split("## Web interface")[1].split("\n## ")[0]
    assert "threaded server" in web and "one read-only connection per request" in web and "WAL" in web


# 50
def test_the_design_and_the_memory_state_the_reversal_of_ticket_24():
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    sub = design.split("### Web consultation interface")[1].split("\n## ")[0]
    assert "reversal of ticket 24" in sub.lower() and "WAL" in sub and "no code view" in sub
    assert "Planned screens of the next tickets" in sub
    memory = (ROOT / ".asl" / "process-memory.md").read_text(encoding="utf-8")
    entry = re.search(r"(?is)(ticket 26.*)", memory)
    if entry:  # the entry is written by the documentation phase; when present it must say this
        low = entry.group(1).lower()
        assert "decision reversal of ticket #24" in low and "code view" in low


# 55
def test_the_test_map_has_exactly_the_cases_of_the_plan_and_each_test_exists():
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(int(k) for k in cases) == list(range(1, PLAN_CASES + 1))
    for key, entries in cases.items():
        assert entries, key
        for entry in entries:
            tree = ast.parse((ROOT / entry["file"]).read_text(encoding="utf-8"))
            names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            assert entry["name"] in names, (key, entry)
    docs = (ROOT / "tests" / "test_orchestrator_docs.py").read_text(encoding="utf-8")
    assert f"PLAN_CASES = {PLAN_CASES}" in docs


# 50
def test_the_api_document_lists_every_route_and_the_design_and_memory_are_complete():
    api_doc = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    from cockpit_helpers import cockpit_db  # noqa: F401
    from kaggle_agent.web.app import create_app

    app = create_app(lambda: None)
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith("/api/"):
            continue
        route = re.sub(r"<(?:path:|int:)?(\w+)>", lambda m: "<id>" if m.group(1) in ("run_id", "attempt_id", "job_id")
                       else f"<{m.group(1)}>", rule.rule)
        assert f"/{route.lstrip('/')}" in api_doc, route
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    sub = design.split("### Web consultation interface")[1].split("\n## ")[0]
    for word in ("Cockpit 2 of 4", "step 15", "not changed in place", "override", "audited", "subprocess",
                 "web/controller.py"):
        assert word.lower() in sub.lower(), word
    memory = (ROOT / ".asl" / "process-memory.md").read_text(encoding="utf-8")
    entry = re.search(r"(?is)(ticket #?27.*)", memory)
    if entry:  # written by the documentation phase; when present it must name the cockpit
        assert "cockpit" in entry.group(1).lower()
