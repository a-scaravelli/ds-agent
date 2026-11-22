"""README, docs/api.md, design note and test map of the closing cockpit ticket (case 55)."""

import json
import re
from pathlib import Path

from cockpit_helpers import cockpit_client, cockpit_db, loopback_only  # noqa: F401  (fixtures)

ROOT = Path(__file__).resolve().parent.parent
PLAN_CASES = 58


def normal(rule: str) -> str:
    names = {"run_id": "<id>", "attempt_id": "<id>", "round_id": "<round>", "n": "<n>", "job_id": "<id>", "name": "<name>"}
    return re.sub(r"<(?:path:|int:)?(\w+)>", lambda m: names.get(m.group(1), "<id>"), rule)


# 55
def test_the_readme_the_api_document_and_the_design_note_cover_the_new_cockpit(cockpit_client):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    for word in ("Unseal and read", "ds-decisions", "holdout-reads.jsonl", "cycles.json", "carry/", "Try this",
                 "chosen by the DS", "bar overridden by the DS", "holdout re-read", "MANUAL_COCKPIT_FINISH.md"):
        assert word in readme, word
    new = [r.rule for r in cockpit_client.application.url_map.iter_rules()
           if r.rule.startswith("/api/") and any(k in r.rule for k in (
               "holdout", "bundle", "gap", "history", "learnings", "overrides", "submission", "choose-model",
               "bar-override", "steering", "try-this", "fork", "reproduce", "predict", "rerun", "finalize", "labels"))]
    assert len(new) >= 15
    for rule in new:
        assert normal(rule) in api or rule in api, rule
    step = [l for l in design.splitlines() if l.startswith("15. **Web interface with input")]
    assert step and "never changed in place" in step[0] and "override" in step[0] and "audit" in step[0]


# 55
def test_the_test_map_holds_exactly_the_cases_of_the_plan_and_every_name_exists():
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(int(k) for k in cases) == list(range(1, PLAN_CASES + 1))
    for key, entries in cases.items():
        assert entries, key
        for entry in entries:
            text = (ROOT / entry["file"]).read_text(encoding="utf-8")
            assert f"def {entry['name']}" in text, (key, entry)
