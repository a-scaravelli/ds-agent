"""Polish of the ticket 1 findings: narrow runs list, short hypotheses, best seen versus chosen."""

import re
from pathlib import Path

from cockpit_helpers import RUN_A, _ok, client_for, page, build_cockpit_registry, cockpit_db, loopback_only  # noqa: F401
from registry_helpers import make_attempt

from kaggle_agent.registry import SqliteRegistry

WEB = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "web"
LONG = " ".join(f"word{n}" for n in range(1, 160))  # about 1000 characters


def extended_registry(path):
    """The cockpit registry plus a better attempt that the program did NOT choose, and a very long hypothesis."""
    reg = SqliteRegistry(path)
    reg.add_attempt(RUN_A, make_attempt("a5", hypothesis=LONG, rationale=LONG, what_changed="long text"), "fp-a5")
    _ok(reg, "a5", 0.75)
    reg.close()
    return path


# 45
def test_the_plan_column_has_a_minimum_width_and_normal_word_breaking_at_narrow_widths(cockpit_db):
    css = (WEB / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert re.search(r"\.plan-col[^{]*\{[^}]*min-width:\s*1\d(\.\d)?rem", css)
    cell = " ".join(re.findall(r"\.plan-cell[^{]*\{([^}]*)\}", css))
    assert "word-break: normal" in cell and "overflow-wrap: normal" in cell
    assert not re.search(r"(break-all|break-word|overflow-wrap:\s*anywhere)", css.split("/* the Plan column")[1])
    narrow = re.search(r"@media \(max-width: (\d+)px\)\s*\{[^@]*?\.plan-col", css.split("/* the Plan column")[1])
    assert narrow and int(narrow.group(1)) >= 800  # covers about 800 px
    html = page(client_for(cockpit_db, cockpit_db.parent), "/")
    assert 'class="plan-col"' in html and 'plan-cell"' in html


# 46
def test_long_hypotheses_are_shortened_with_an_expand_control_and_stay_searchable(cockpit_db):
    extended_registry(cockpit_db)
    client = client_for(cockpit_db, cockpit_db.parent)
    html = page(client, f"/cycles/{RUN_A}/attempts")
    block = html[html.index("long text") - 3000:html.index("long text") + 3000]
    summaries = re.findall(r'<details class="more"><summary>(.*?)</summary><div class="full">(.*?)</div></details>',
                           block, re.S)
    assert summaries
    short, full = summaries[0]
    assert full.strip() == LONG and len(LONG) > 900
    assert 100 <= len(short) <= 130 and (short.endswith("…") or short.endswith("..."))
    assert short.rstrip("….").rsplit(" ", 1)[0] in LONG and not short.rstrip("….").endswith("wor")  # a word boundary
    assert "<details" in html and "word150" in html  # the full text is in the page
    found = client.get(f"/api/runs/{RUN_A}/attempts?q=word150").get_json()["data"]
    assert [a["id"] for a in found] == ["a5"]  # and searchable
    short_hypothesis = page(client, f"/attempts/a1")
    assert "trees help" in short_hypothesis


# 47
def test_best_seen_and_the_chosen_model_are_shown_apart_with_the_rule_and_the_numbers(cockpit_db):
    extended_registry(cockpit_db)
    client = client_for(cockpit_db, cockpit_db.parent)
    run = client.get(f"/api/runs/{RUN_A}").get_json()["data"]
    assert run["best_seen"]["mean"] == 0.75 and run["chosen"]["mean"] == 0.72  # best seen is above the chosen one
    for url in ("/", f"/cycles/{RUN_A}"):
        html = page(client, url)
        assert "Best seen" in html and "Chosen by the program (strict bar)" in html, url
        assert "Best score" not in html
        assert "threshold_k" in html and "baseline noise" in html
    detail = page(client, f"/cycles/{RUN_A}")
    note = detail[detail.index('id="improvement-rule"'):].split("</p>", 1)[0]
    assert "mean paired difference" in note and "bar" in note and "0.75" in detail and "0.72" in detail
    assert re.search(r"threshold_k\s+[\d.]+\s*&times;\s*baseline noise\s+[\d.]+", note)
