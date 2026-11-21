"""The analytics pages: 200 everywhere, design system, narrow widths, plain words and 'not recorded' states (cases 38, 39, 40)."""

import json
import re
import shutil
from html import parser as html_parser
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "src" / "kaggle_agent" / "web"
NEW_TEMPLATES = ("analytics_attempt.html", "analytics_compare.html", "analytics_paired.html")


def pages_for(world):
    a, b = attempts_of(world)[0], attempts_of(world)[-1]
    run = cycle_of(world)
    return [f"/attempts/{b}/analytics", f"/runs/{run}/compare", f"/runs/{run}/compare?sort=score&order=desc&q=hypo",
            f"/runs/{run}/compare?ids={a},{b}", f"/runs/{run}/paired", f"/runs/{run}/paired?a={a}&b={b}"]


# 38
def test_every_new_page_answers_200_for_recorded_unrecorded_and_old_projects(proba_world, labels_world, failure_world, tmp_path):
    clients = [client_of(proba_world), client_of(labels_world), client_of(failure_world)]
    worlds = [proba_world, labels_world, failure_world]
    for client, world in zip(clients, worlds):
        for url in pages_for(world):
            assert client.get(url).status_code == 200, url
        for attempt in attempts_of(world):  # also the attempts that failed
            assert client.get(f"/attempts/{attempt}/analytics").status_code == 200, attempt
    old = client_for_folder(old_project(labels_world, tmp_path / "old"))
    for url in pages_for(labels_world):
        assert old.get(url).status_code == 200, url
    assert client_of(proba_world).get("/attempts/nope/analytics").status_code == 404
    assert client_of(proba_world).get("/runs/nope/compare").status_code == 404
    assert client_of(proba_world).get(f"/runs/{cycle_of(proba_world)}/paired?a=../x&b=y").status_code == 404


# 38
def test_the_new_templates_use_tokens_only_no_inline_script_or_style_and_the_theme_toggle(proba_world):
    for name in NEW_TEMPLATES:
        text = (WEB / "templates" / name).read_text(encoding="utf-8")
        assert not re.search(r"\sstyle\s*=|<style\b|\sonclick\s*=|<script>[^<]", text), name
        assert not re.search(r"#[0-9a-fA-F]{6}\b|\brgba?\(", text), name
    client = client_of(proba_world)
    for url in pages_for(proba_world):
        html = client.get(url).get_data(as_text=True)
        assert html.count('id="theme-toggle"') == 1 and '<meta name="color-scheme" content="light">' in html
        assert "Content-Security-Policy" in client.get(url).headers
    page = client.get(pages_for(proba_world)[0]).get_data(as_text=True)
    assert 'class="panel"' in page and 'class="table-wrap"' in page and "analytics.js" in page
    assert client.get(pages_for(proba_world)[1]).get_data(as_text=True).count("aria-current") >= 1


# 39
def test_new_components_sit_in_the_scroll_container_and_collapse_under_the_breakpoint():
    css = (WEB / "static" / "css" / "app.css").read_text(encoding="utf-8")
    block = css[css.index("performance analytics (ticket 29)"):]
    assert "@media (max-width: 900px)" in block  # the existing breakpoint
    for value, unit in re.findall(r"(?<!-)(?:min-|max-)?width:\s*([\d.]+)(px|rem)", block):
        assert float(value) * (1 if unit == "px" else 16) <= 1280, (value, unit)
    for name in NEW_TEMPLATES:
        text = (WEB / "templates" / name).read_text(encoding="utf-8")
        assert text.count("<table") <= text.count('class="table-wrap"'), name
        assert not re.search(r"width\s*=\s*\"?\d{4,}", text), name
    js = (WEB / "static" / "js" / "analytics.js").read_text(encoding="utf-8")
    assert js.count("<table") == 0 and '"table-wrap"' in js  # tables built by the script sit in the same container
    manual = (ROOT / "tests" / "MANUAL_COCKPIT_ANALYTICS.md").read_text(encoding="utf-8")
    assert "1280" in manual and "narrow" in manual.lower()


class Collect(html_parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections, self.explain, self.not_recorded, self.text = 0, 0, [], []
        self._stack = []

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class") or ""
        if tag == "section" and "panel" in classes.split():
            self.sections += 1
        if tag == "p" and "explain" in classes.split():
            self.explain += 1
        if tag == "div" and "not-recorded" in classes.split():
            self.not_recorded.append("")
            self._stack.append(len(self.not_recorded) - 1)
        elif self._stack and tag == "div":
            self._stack.append(None)

    def handle_endtag(self, tag):
        if tag == "div" and self._stack:
            self._stack.pop()

    def handle_data(self, data):
        for index in self._stack:
            if index is not None:
                self.not_recorded[index] += data


def parsed(client, url):
    collect = Collect()
    html = client.get(url).get_data(as_text=True)
    collect.feed(html)
    return collect, html


# 40
def test_every_panel_has_an_explanation_and_each_missing_input_a_distinct_reason(proba_world, labels_world, failure_world, tmp_path):
    for world in (proba_world, labels_world):
        collect, _ = parsed(client_of(world), pages_for(world)[0])
        assert collect.sections >= 6 and collect.explain >= collect.sections - 0
    compare, _ = parsed(client_of(proba_world), pages_for(proba_world)[1])
    assert compare.explain >= compare.sections
    reasons = {}
    # run failed
    failed = attempts_of(failure_world)[1]
    reasons["run_failed"] = parsed(client_of(failure_world), f"/attempts/{failed}/analytics")[0].not_recorded
    # older project
    old = client_for_folder(old_project(labels_world, tmp_path / "old"))
    reasons["older_project"] = parsed(old, pages_for(labels_world)[0])[0].not_recorded
    # describe not provided and label-only (labels world)
    reasons["label_only"] = parsed(client_of(labels_world), pages_for(labels_world)[0])[0].not_recorded
    # too large and invalid describe (an index written by a recorder that hit the limits)
    folder = old_project(labels_world, tmp_path / "limits")
    bundle = f"{attempts_of(labels_world)[-1]}-b1"
    (folder / "analytics").mkdir()
    lines = [{"v": 1, "kind": "oof", "bundle_id": bundle, "status": "too_large", "rows": 9},
             {"v": 1, "kind": "describe", "bundle_id": bundle, "status": "invalid", "reason": "the file is not valid JSON"},
             {"v": 1, "kind": "development", "status": "too_large"}]
    (folder / "analytics" / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    limits = client_for_folder(folder)
    page_limits, html = parsed(limits, pages_for(labels_world)[0])
    reasons["too_large"] = page_limits.not_recorded
    texts = {k: " ".join(" ".join(v).split()) for k, v in reasons.items()}
    assert "run of this attempt failed" in texts["run_failed"]
    assert "before analytics were recorded" in texts["older_project"]
    assert "labels only" in texts["label_only"]
    assert "over the size limit" in texts["too_large"] and "provided but invalid: the file is not valid JSON" in " ".join(page_limits.not_recorded)
    assert all(t.count("Not recorded for this attempt") >= 1 for t in texts.values())
    assert len({re.sub(r"Not recorded for this attempt", "", t)[:60] for t in texts.values()}) == len(texts)  # distinct reasons
    provided, _ = parsed(client_of(labels_world), pages_for(labels_world)[0])
    assert "describe.json" in " ".join(provided.not_recorded + [_]) or "Not provided" in _
