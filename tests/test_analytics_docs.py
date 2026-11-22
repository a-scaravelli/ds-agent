"""Documentation and bookkeeping of the analytics ticket (cases 46 and 48)."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN_CASES = 48


# 46
def test_the_test_map_holds_exactly_the_cases_of_this_plan_and_every_test_exists():
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(int(k) for k in cases) == list(range(1, PLAN_CASES + 1))
    for key, entries in cases.items():
        assert entries, key
        for entry in entries:
            tree = ast.parse((ROOT / entry["file"]).read_text(encoding="utf-8"))
            names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            assert entry["name"] in names, (key, entry)
    for name in ("test_cockpit_docs.py", "test_orchestrator_docs.py", "test_cockpit_design.py"):
        assert f"PLAN_CASES = {PLAN_CASES}" in (ROOT / "tests" / name).read_text(encoding="utf-8")


# 48
def test_readme_api_doc_design_note_and_memory_describe_the_analytics():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Performance analytics (ticket 29)")[1].split("\n## ")[0]
    for word in ("/attempts/<id>/analytics", "/runs/<id>/compare", "/runs/<id>/paired", "analytics/", "index.jsonl",
                 "describe.json", "RecordingCandidate", "MANUAL_COCKPIT_ANALYTICS.md", "holdout"):
        assert word in section, word
    api_doc = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    for route in ("/api/attempts/<id>/analytics", "/api/attempts/<id>/folds", "/api/attempts/<id>/panels",
                  "/api/attempts/<id>/segments", "/api/attempts/<id>/hardest", "/api/attempts/<id>/flags",
                  "/api/attempts/<id>/features", "/api/runs/<id>/paired", "/api/runs/<id>/history",
                  "/api/runs/<id>/compare", "/api/runs/<id>/side-by-side", "/api/runs/<id>/ablation",
                  "/api/jobs/importance", "not_recorded"):
        assert route in api_doc, route
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    note = design.split("### Performance analytics (ticket 29)")[1].split("\n### ")[0].lower()
    assert "wrapping" in note or "wraps" in note
    assert "holdout" in note and "never" in note
    memory = (ROOT / ".asl" / "process-memory.md").read_text(encoding="utf-8")
    entry = re.search(r"(?is)(ticket #?29.*)", memory)
    if entry:  # written by the documentation phase; when present it must name the analytics
        assert "analytics" in entry.group(1).lower()
