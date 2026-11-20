"""Design pass of the cockpit: light default, opt-in dark, radius tokens, contrast, colour guard, markup contract."""

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from cockpit_helpers import (  # noqa: F401  (fixtures)
    all_page_urls,
    client_for,
    cockpit_client,
    cockpit_db,
    html_tags,
    loopback_only,
)
from markup_contract import FIXTURE, current
from test_cockpit_assets import LARGE_PAIRS, TEXT_PAIRS, _colours, contrast

from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web.security import CSP

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "src" / "kaggle_agent" / "web"
STATIC = WEB / "static"
TEMPLATES = WEB / "templates"
TOKENS = (STATIC / "css" / "tokens.css").read_text(encoding="utf-8")
APP_CSS = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
APP_RULE_LIST = [(m.group(1).strip(), m.group(2))
                 for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", APP_CSS, flags=re.S))]
APP_RULES = dict(APP_RULE_LIST)
OWN_JS = [STATIC / "js" / n for n in ("app.js", "charts.js", "cockpit.js", "live.js", "markdown.js", "notify.js", "theme.js")]
STATUSES = ("succeeded", "failed", "running", "waiting", "suspended", "unknown")


def _dark_block() -> str:
    start = TOKENS.index(':root[data-theme="dark"]')
    return TOKENS[start:]


def _tokens(block: str) -> dict:
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", block))


def _hue(colour: str) -> float:
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return 1.0 * b - max(r, g) if max(r, g, b) else 0  # positive when blue dominates


# 1
def test_light_tokens_are_white_and_the_dark_variant_is_an_optional_blue_block():
    light, dark = _colours()
    light_block = TOKENS[: TOKENS.index(':root[data-theme="dark"]')]
    assert light["--bg"] == "#ffffff" and light["--surface"] == "#ffffff"
    assert "--bg: #ffffff" in light_block and "--accent:" in light_block
    dark_only = _tokens(_dark_block())
    assert "--bg" in dark_only and "--accent" in dark_only and set(dark_only) <= set(_tokens(light_block))
    assert dark_only["--bg"] != "#000000" and dark_only["--bg"] != "#ffffff"
    assert _hue(dark_only["--accent"]) > 0 and _hue(light["--accent"]) > 0  # blue accent in both
    assert _hue(dark_only["--bg"]) > 0  # slate blue, not grey or black
    assert not [name for name in dark_only if name.startswith(("--radius", "--sp-", "--fs-"))]


# 2
def test_light_is_the_default_even_on_a_dark_os_and_dark_is_an_explicit_opt_in():
    for text in [TOKENS, APP_CSS, *[p.read_text(encoding="utf-8") for p in OWN_JS]]:
        assert "prefers-color-scheme" not in text
    assert re.search(r"html\s*\{\s*color-scheme:\s*light;\s*\}", APP_CSS)
    assert 'html[data-theme="dark"] { color-scheme: dark; }' in APP_CSS
    assert len(re.findall(r"color-scheme:\s*dark", APP_CSS)) == 1
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    head = base[: base.index("</head>")]
    tag = re.search(r"<script[^>]*js/theme\.js[^>]*>", head)
    assert tag and "defer" not in tag.group(0) and "async" not in tag.group(0)
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", base)
    source = (STATIC / "js" / "theme.js").read_text(encoding="utf-8")
    assert 'stored() === "dark"' in source and "try" in source and "removeItem" in source and "themechange" in source


THEME_RUNNER = """
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
function run(value, throws, click) {
  const listeners = {}; const calls = [];
  const attrs = {};
  const button = {setAttribute(k, v) { attrs[k] = v; }, addEventListener(t, f) { listeners.click = f; }};
  const store = {};
  if (value !== null) store['cockpit-theme'] = value;
  const events = [];
  const doc = {documentElement: {dataset: {}}, getElementById: () => button,
    addEventListener(t, f) { listeners[t] = f; }, dispatchEvent(e) { events.push(e.type); }};
  const win = {matchMedia: () => ({matches: true}),
    localStorage: {getItem(k) { if (throws) throw new Error('blocked'); return k in store ? store[k] : null; },
      setItem(k, v) { if (throws) throw new Error('blocked'); store[k] = v; calls.push('set'); },
      removeItem(k) { if (throws) throw new Error('blocked'); delete store[k]; calls.push('remove'); }}};
  const ctx = {document: doc, window: win, CustomEvent: function (type) { this.type = type; }};
  vm.createContext(ctx); vm.runInContext(source, ctx);
  const before = doc.documentElement.dataset.theme || null;
  listeners.DOMContentLoaded();
  if (click) { listeners.click(); listeners.click(); }
  const mid = doc.documentElement.dataset.theme || null;
  return {before, pressed: attrs['aria-pressed'], after: mid, store: Object.assign({}, store), events, calls};
}
const out = {
  none: run(null, false, false), light: run('light', false, false), garbage: run('x<y', false, false),
  throwing: run(null, true, false), dark: run('dark', false, false),
  flip: run(null, false, true), throwingClick: run(null, true, true),
};
const one = (() => { // a single click from light
  const r = run(null, false, false); return r; })();
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_theme_js_sets_dark_only_for_the_exact_stored_value_when_run_by_node(tmp_path):
    runner = tmp_path / "run.js"
    runner.write_text(THEME_RUNNER, encoding="utf-8")
    done = subprocess.run(["node", str(runner), str(STATIC / "js" / "theme.js")],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    for case in ("none", "light", "garbage", "throwing"):  # the fake OS reports dark everywhere
        assert out[case]["before"] is None and out[case]["pressed"] == "false", case
    assert out["dark"]["before"] == "dark" and out["dark"]["pressed"] == "true"
    # two clicks from light: dark then light again, the key written then removed, events dispatched
    assert out["flip"]["calls"] == ["set", "remove"] and out["flip"]["events"] == ["themechange"] * 2
    assert out["flip"]["after"] is None and out["flip"]["store"] == {}
    assert out["throwingClick"]["after"] is None and out["throwingClick"]["events"] == ["themechange"] * 2


# 3
def test_every_border_radius_uses_a_token_and_the_main_components_are_rounded():
    tokens = _tokens(TOKENS)
    assert (tokens["--radius-sm"], tokens["--radius-md"], tokens["--radius-lg"]) == ("6px", "10px", "14px")
    assert "--radius-pill" in tokens and tokens["--radius"] == "var(--radius-sm)"
    for text in (TOKENS, APP_CSS):
        for value in re.findall(r"border-radius:\s*([^;}]+)", text):
            assert re.fullmatch(r"var\(--radius[\w-]*\)|0|50%", value.strip()), value
    assert "999px" not in APP_CSS
    for selector in (".side-nav a", ".table-wrap", "dialog", "input, select, button", ".panel", ".kpi", ".badge",
                     ".tag", ".empty"):
        match = [body for sel, body in APP_RULE_LIST if sel == selector]
        assert any("border-radius: var(--radius" in body for body in match), selector
    assert "html[data-theme" not in " ".join(sel for sel, body in APP_RULES.items() if "border-radius" in body)


# 4
def test_text_and_accent_pairs_meet_wcag_aa_in_light_and_dark():
    light, dark = _colours()
    required = {("--text", "--bg"), ("--text", "--surface"), ("--text", "--surface-alt"),
                ("--text-muted", "--surface"), ("--link", "--surface"), ("--link", "--accent-soft"),
                ("--accent-fg", "--accent"), ("--nav-active-fg", "--nav-active-bg"),
                ("--text-muted", "--table-head-bg"), ("--danger", "--danger-bg")}
    assert required <= set(TEXT_PAIRS)
    for mode, colours in (("light", light), ("dark", dark)):
        for foreground, background in required:
            assert contrast(colours[foreground], colours[background]) >= 4.5, (mode, foreground, background)


# 5
def test_status_colours_are_readable_and_every_status_has_a_glyph_and_a_label(cockpit_client):
    light, dark = _colours()
    for colours in (light, dark):
        for status in STATUSES:
            assert contrast(colours[f"--st-{status}-fg"], colours[f"--st-{status}-bg"]) >= 4.5, status
    glyphs = {}
    for status in STATUSES:
        match = re.search(rf"\.badge\.status-{status}(?:::before|,[^{{]*)*[^{{]*::before\s*\{{\s*content:\s*\"([^\"]+)\"", APP_CSS)
        assert match, status
        glyphs[status] = match.group(1)
    assert len(set(glyphs.values())) == len(glyphs)  # each status has its own glyph
    html = cockpit_client.get("/").get_data(as_text=True)
    assert re.search(r'<span class="badge status-\w+">\s*[A-Za-z]', html)  # the text label stays


# 6
def test_focus_ring_borders_progress_and_chart_series_keep_3_to_1():
    light, dark = _colours()
    for colours in (light, dark):
        for name in ("--focus", "--border-strong", "--progress-fill", "--chart-line", "--chart-chosen",
                     "--chart-bar", "--chart-baseline", "--chart-error"):
            for background in ("--surface", "--bg"):
                assert contrast(colours[name], colours[background]) >= 3.0, (name, background)
        assert contrast(colours["--chart-line"], colours["--chart-bg"]) >= 3.0
    assert {("--focus", "--surface"), ("--progress-fill", "--surface"), ("--chart-line", "--surface")} <= set(LARGE_PAIRS)
    assert ":focus-visible" in APP_CSS and "--shadow-focus" in APP_CSS


# 7
HEX = re.compile(r"(?<![&\w])#[0-9a-fA-F]{3,8}\b")
FUNCTIONAL = re.compile(r"\b(?:rgba?|hsla?|oklch|color-mix)\(")
NAMES = "black|white|red|green|blue|gray|grey|orange|yellow|purple|navy|teal|pink|brown|silver|gold|cyan|magenta"
CSS_NAMED = re.compile(
    rf"(?<![\w-])(?:color|background(?:-color)?|border(?:-[\w]+)?-color|outline-color|fill|stroke)\s*:\s*[^;}}]*?\b(?:{NAMES})\b")
JS_NAMED = re.compile(rf"""["'](?:{NAMES})["']""")
JS_STYLE = re.compile(r"(?:fillStyle|strokeStyle)\s*=\s*([^;]+);|\b(?:stroke|fill)\s*:\s*([^,}\n]+)")


def colour_violations(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    found = [m.group(0) for m in HEX.finditer(text)] + [m.group(0) for m in FUNCTIONAL.finditer(text)]
    if path.suffix == ".css":
        found += [m.group(0) for m in CSS_NAMED.finditer(text)]
    elif path.suffix == ".js":
        found += JS_NAMED.findall(text)
        for match in JS_STYLE.finditer(text):
            value = (match.group(1) or match.group(2)).strip()
            if re.match(r"""["'#]|rgb|hsl""", value):
                found.append(value)
    return found


def test_no_hard_coded_colour_outside_the_tokens_file():
    files = [*sorted(TEMPLATES.glob("*.html")), STATIC / "css" / "app.css", *OWN_JS]
    assert len(OWN_JS) == 7 and all(p.is_file() for p in files)
    for path in files:
        assert colour_violations(path) == [], path.name
    assert "#b42318" not in APP_CSS
    # the guard itself catches what it must and lets the allowed words through
    probe = ROOT / "does-not-exist.css"
    for bad in ("a { color: #fff; }", "a { background: red; }", "a { border-top-color: rgba(0,0,0,.5); }"):
        tmp = probe.with_name("probe.css")
        try:
            tmp.write_text(bad, encoding="utf-8")
            assert colour_violations(tmp), bad
        finally:
            tmp.unlink(missing_ok=True)
    assert not CSS_NAMED.search("a { background: transparent; color: currentColor; border-color: inherit; }")
    assert not HEX.search("&#10003; required data-green")


# 8
def test_the_chart_reads_tokens_keeps_its_markers_rebuilds_on_theme_change_and_has_rounded_overlays():
    source = (STATIC / "js" / "charts.js").read_text(encoding="utf-8")
    for token in ("--chart-line", "--chart-error", "--chart-baseline", "--chart-bar", "--chart-chosen",
                  "--chart-grid", "--chart-axis", "--chart-bg"):
        assert f'css("{token}")' in source, token
    for part in ("dash: [6, 4]", "dash: [2, 3]", "ctx.arc(", "p.dispersion", "themechange", "improvement bar"):
        assert part in source, part
    rules = APP_RULES
    for selector in (".chart .u-legend", ".chart .u-tooltip", ".chart"):
        assert "border-radius: var(--radius" in rules[selector], selector
    assert re.search(r"\.chart \.u-select", APP_CSS) and ".chart .u-hz .u-cursor-x" in APP_CSS
    light, dark = _colours()
    for colours in (light, dark):
        assert contrast(colours["--chart-line"], colours["--surface"]) >= 3.0
        assert contrast(colours["--chart-axis"], colours["--chart-bg"]) >= 4.5
    assert light["--chart-bg"] == "#ffffff"


# 9
def test_every_page_the_error_page_and_the_empty_registry_answer(cockpit_client, tmp_path):
    for url in all_page_urls():
        assert cockpit_client.get(url).status_code == 200, url
    missing = cockpit_client.get("/cycles/nope")
    assert missing.status_code == 404 and 'class="error-state"' in missing.get_data(as_text=True)
    empty = tmp_path.resolve() / "empty.db"
    SqliteRegistry(empty).close()
    client = client_for(empty)
    assert "No project yet" in client.get("/").get_data(as_text=True)
    assert "No suggestions" in client.get("/suggestions").get_data(as_text=True)


# 10
def test_no_functional_markup_was_removed_by_the_restyle():
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    now = current()
    assert set(recorded) <= set(now)
    for name, expected in recorded.items():
        for key in ("attributes", "classes", "controls"):
            missing = set(expected[key]) - set(now[name][key])
            assert not missing, (name, key, sorted(missing))
    for path in TEMPLATES.glob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\sstyle\s*=|<style\b|\sonclick\s*=", text), path.name
    assert {"id=theme-toggle", "type=button"} <= set(now["base.html"]["attributes"])


# 11
def test_csp_dependencies_and_vendored_files_are_unchanged(cockpit_client):
    assert "script-src 'self'" in CSP and "style-src 'self'" in CSP
    assert "unsafe-inline" not in CSP and "unsafe-eval" not in CSP and "http" not in CSP
    for url in ("/", "/cycles/nope", "/static/js/theme.js"):
        assert cockpit_client.get(url).headers["Content-Security-Policy"] == CSP
    html = cockpit_client.get("/").get_data(as_text=True)
    for name, attrs in html_tags(html):
        if name == "script" and attrs.get("src"):
            assert attrs["src"].startswith("/static/")
    assert "/static/js/theme.js" in html
    for path in (STATIC / "css" / "app.css", STATIC / "css" / "tokens.css", *OWN_JS):
        text = path.read_text(encoding="utf-8")
        assert "url(" not in text and "@import" not in text and not re.search(r"https?://", text), path.name
    record = (STATIC / "VENDORED.md").read_text(encoding="utf-8")
    for path in (STATIC / "vendor").iterdir():
        assert hashlib.sha256(path.read_bytes()).hexdigest() in record, path.name
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert not [d for d in deps if re.match(r"(?i)(node|npm|sass|tailwind|bootstrap)", d)]


# 12
def test_the_narrow_width_fixes_are_kept_and_no_own_width_is_wider_than_1280px():
    block = APP_CSS[APP_CSS.index("@media (max-width: 900px) {"):]
    assert "grid-template-areas: \"nav\" \"top\" \"main\"" in block
    assert re.search(r"dl\.facts\s*\{\s*grid-template-columns:\s*minmax\(0, 1fr\)", block)
    assert ".side-nav { position: static" in block
    assert re.search(r"\.plan-col, \.plan-cell \{ min-width: 12rem", APP_CSS)
    assert re.search(r"\.table-wrap \{ overflow-x: auto", APP_CSS)
    assert re.search(r"\.table-wrap \{[^}]*overflow: auto", APP_CSS)
    for value, unit in re.findall(r"(?<!-)(?:min-)?width:\s*([\d.]+)(px|rem)", APP_CSS):
        assert float(value) * (1 if unit == "px" else 16) <= 1280, (value, unit)


# 13
def test_the_readme_has_the_design_section_with_the_manual_checklist():
    section = (ROOT / "README.md").read_text(encoding="utf-8").split("### Cockpit design")[1].split("\n##")[0]
    for word in ("800", "1280", "1440", "light", "dark", "docs/screenshots/", "tokens.css", "prefers-color-scheme",
                 "cockpit-theme", "localStorage.setItem", "localStorage.removeItem", "toggle"):
        assert word in section, word
    for page in ("runs list", "run overview", "rounds", "attempts", "trace", "report", "suggestions", "guidance",
                 "live", "run control", "decisions", "setup", "environment", "audit log", "error", "empty", "dialogs"):
        assert page in section.lower(), page


# 14
def test_the_theme_toggle_is_one_accessible_native_button_on_every_page(cockpit_client):
    for url in [*all_page_urls(), "/cycles/nope"]:
        html = cockpit_client.get(url).get_data(as_text=True)
        tags = html_tags(html)
        buttons = [attrs for name, attrs in tags if name == "button" and attrs.get("id") == "theme-toggle"]
        assert len(buttons) == 1, url
        assert buttons[0].get("type") == "button" and "aria-pressed" in buttons[0] and "tabindex" not in buttons[0]
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert "<form" not in base and "fetch(" not in base
    routes = {r.rule for r in cockpit_client.application.url_map.iter_rules()}
    assert not [r for r in routes if "theme" in r]


# 15
def test_the_test_map_holds_exactly_the_cases_of_this_ticket():
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(int(k) for k in cases) == list(range(1, 16))
    for key, entries in cases.items():
        assert entries, key
        for entry in entries:
            tree = ast.parse((ROOT / entry["file"]).read_text(encoding="utf-8"))
            assert entry["name"] in {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}, (key, entry)
    for name in ("test_cockpit_docs.py", "test_orchestrator_docs.py"):
        assert "PLAN_CASES = 15" in (ROOT / "tests" / name).read_text(encoding="utf-8")
