"""Static assets and design system: local files only, no inline code, widths, contrast (no network)."""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from cockpit_helpers import (  # noqa: F401  (fixtures)
    all_page_urls,
    cockpit_client,
    cockpit_db,
    html_tags,
    loopback_only,
)

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "src" / "kaggle_agent" / "web"
STATIC = WEB / "static"
TEMPLATES = WEB / "templates"
OWN_CSS = [STATIC / "css" / "tokens.css", STATIC / "css" / "app.css"]
OWN_JS = [STATIC / "js" / name for name in ("app.js", "charts.js", "markdown.js", "theme.js")]
MAX_MIN_WIDTH_PX = 1280
MIN_WIDTH_ALLOWLIST: set[str] = set()  # documented exceptions to the min-width rule: none


# 12
def test_css_has_sticky_headers_focus_rings_narrow_rule_and_no_fixed_wide_widths():
    app = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    assert re.search(r"thead th\s*\{[^}]*position:\s*sticky", app, re.S)
    assert ":focus-visible" in app
    assert re.search(r"@media \(max-width:\s*900px\)", app)
    for path in OWN_CSS:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"width:\s*\d{4,}px", text), path.name
        for value in re.findall(r"min-width:\s*([\d.]+)(px|rem|em)", text):
            pixels = float(value[0]) * (1 if value[1] == "px" else 16)
            assert pixels <= MAX_MIN_WIDTH_PX or f"{value[0]}{value[1]}" in MIN_WIDTH_ALLOWLIST, (path.name, value)
    assert "vendor" not in " ".join(p.name for p in OWN_CSS)  # vendored files are not scanned


# 12
def test_interactive_elements_are_native_and_keyboard_reachable():
    for path in sorted(TEMPLATES.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\sonclick\s*=|\sonkey\w+\s*=", text), path.name
        assert not re.search(r'role="button"', text), path.name
        assert not re.search(r"<a\s(?![^>]*\bhref=)", text), path.name  # every link has an href
        assert not re.search(r'tabindex="[1-9]', text), path.name
    shell = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert 'class="skip-link"' in shell and 'lang="en"' in shell and "<main" in shell
    assert 'th button.sort' in (STATIC / "css" / "app.css").read_text(encoding="utf-8")


# 13
def _colours() -> tuple[dict, dict]:
    text = (STATIC / "css" / "tokens.css").read_text(encoding="utf-8")
    dark_start = text.index(':root[data-theme="dark"]')
    light_block, dark_block = text[:dark_start], text[dark_start:]
    pattern = re.compile(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\s*;")
    light = dict(pattern.findall(light_block))
    dark = {**light, **dict(pattern.findall(dark_block))}
    return light, dark


def _luminance(colour: str) -> float:
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground: str, background: str) -> float:
    a, b = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (a + 0.05) / (b + 0.05)


TEXT_PAIRS = [
    ("--text", "--bg"), ("--text", "--surface"), ("--text", "--surface-alt"),
    ("--text-muted", "--bg"), ("--text-muted", "--surface"), ("--text-muted", "--surface-alt"),
    ("--link", "--surface"), ("--link", "--bg"), ("--accent-fg", "--accent"),
    ("--text", "--accent-soft"), ("--link", "--accent-soft"), ("--nav-active-fg", "--nav-active-bg"),
    ("--text-muted", "--table-head-bg"), ("--text", "--table-head-bg"), ("--danger", "--danger-bg"),
    ("--danger", "--surface"),
    *[(f"--st-{s}-fg", f"--st-{s}-bg") for s in
      ("succeeded", "failed", "running", "waiting", "suspended", "unknown")],
]
LARGE_PAIRS = [
    ("--border-strong", "--surface"), ("--border-strong", "--bg"), ("--focus", "--surface"), ("--focus", "--bg"),
    ("--progress-fill", "--surface"), ("--progress-fill", "--bg"), ("--chart-line", "--surface"),
    ("--chart-line", "--chart-bg"),
]


def test_the_contrast_of_the_documented_token_pairs_is_accessible_in_both_modes():
    light, dark = _colours()
    assert contrast("#000000", "#ffffff") == pytest.approx(21)
    for mode, colours in (("light", light), ("dark", dark)):
        for foreground, background in TEXT_PAIRS:
            ratio = contrast(colours[foreground], colours[background])
            assert ratio >= 4.5, (mode, foreground, background, round(ratio, 2))
        for foreground, background in LARGE_PAIRS:
            ratio = contrast(colours[foreground], colours[background])
            assert ratio >= 3.0, (mode, foreground, background, round(ratio, 2))
    assert light["--bg"] != dark["--bg"]
    assert light["--bg"] == "#ffffff" and dark["--bg"] != "#000000"  # white page by default, dark is not black


# 27
def _static_references(html: str) -> list[str]:
    found = []
    for name, attrs in html_tags(html):
        if name == "script" and attrs.get("src"):
            found.append(attrs["src"])
        if name == "link" and attrs.get("href"):
            found.append(attrs["href"])
        if name == "img" and attrs.get("src"):
            found.append(attrs["src"])
    return found


def test_every_script_stylesheet_and_image_of_every_page_is_served_locally(cockpit_client):
    checked = set()
    for url in all_page_urls() + ["/cycles/nope"]:
        html = cockpit_client.get(url).get_data(as_text=True)
        for reference in _static_references(html):
            assert reference.startswith("/static/"), (url, reference)
            if reference in checked:
                continue
            checked.add(reference)
            response = cockpit_client.get(reference)
            assert response.status_code == 200, reference
            kind = response.headers["Content-Type"]
            expected = "css" if reference.endswith(".css") else "javascript"
            assert expected in kind, (reference, kind)
    assert {"/static/css/tokens.css", "/static/css/app.css", "/static/vendor/htmx.min.js",
            "/static/vendor/marked.min.js", "/static/js/markdown.js", "/static/js/app.js",
            "/static/vendor/uplot.iife.min.js", "/static/vendor/uplot.min.css",
            "/static/js/charts.js", "/static/js/theme.js"} <= checked


def test_no_own_file_mentions_an_external_url_or_inline_code():
    own = [*TEMPLATES.glob("*.html"), *OWN_CSS, *OWN_JS]
    for path in own:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"https?://", text, re.I), path.name
        assert "@import" not in text and "url(" not in text, path.name
        if path.suffix == ".html":
            for match in re.finditer(r"<script\b([^>]*)>", text):
                assert "src=" in match.group(1) or "application/json" in match.group(1), path.name
            assert not re.search(r"\sstyle\s*=", text), path.name
            assert not re.search(r"<style\b", text), path.name
    for page in WEB.glob("*.py"):
        assert "cdn" not in page.read_text(encoding="utf-8").lower(), page.name


def test_vended_files_are_recorded_with_name_version_and_licence():
    record = (STATIC / "VENDORED.md").read_text(encoding="utf-8")
    files = sorted(p.name for p in (STATIC / "vendor").iterdir() if p.is_file())
    assert files == ["htmx.min.js", "marked.min.js", "uplot.iife.min.js", "uplot.min.css"]
    for name in files:
        assert f"vendor/{name}" in record, name
    for library, licence in (("htmx", "0BSD"), ("uPlot", "MIT"), ("marked", "MIT")):
        assert re.search(rf"\| {library} \| \d+\.\d+\.\d+ \| {licence}", record), library
    assert "highlight" not in record.lower().replace("no syntax highlighter is vendored", "")
    import hashlib

    for name in files:
        digest = hashlib.sha256((STATIC / "vendor" / name).read_bytes()).hexdigest()
        assert digest in record, name


def test_package_data_includes_the_static_folder():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "static/**" in pyproject["tool"]["setuptools"]["package-data"]["kaggle_agent.web"]
    assert pyproject["project"]["dependencies"] and not any(
        "htmx" in d or "marked" in d for d in pyproject["project"]["dependencies"])


# 28
def test_markdown_js_escapes_raw_html_and_filters_link_schemes():
    source = (STATIC / "js" / "markdown.js").read_text(encoding="utf-8")
    assert "marked.use(" in source
    assert re.search(r"renderer:\s*\{\s*html:\s*function\s*\(token\)\s*\{\s*return escapeHtml\(token\.text\)", source)
    assert "walkTokens" in source and '"link"' in source and '"image"' in source
    assert 'scheme === "http" || scheme === "https"' in source
    assert "sanitize" not in source and "innerHTML" in source
    record = (STATIC / "VENDORED.md").read_text(encoding="utf-8")
    version = re.search(r"\| marked \| (\d+)\.\d+\.\d+ \|", record)
    assert version and int(version.group(1)) >= 5
    marked = (STATIC / "vendor" / "marked.min.js").read_text(encoding="utf-8")
    assert f"marked v{version.group(0).split('|')[2].strip()}" in marked


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_markdown_js_behaves_as_documented_when_run_by_node(tmp_path):
    script = tmp_path / "run.js"
    script.write_text(
        "const fs = require('fs'), vm = require('vm');\n"
        "const ctx = {window: {}, document: {addEventListener() {}, querySelectorAll() { return []; }}};\n"
        "vm.createContext(ctx);\n"
        "for (const file of process.argv.slice(2)) vm.runInContext(fs.readFileSync(file, 'utf8'), ctx);\n"
        "const html = vm.runInContext('marked.parse(\"<script>alert(1)</script> <img src=x onerror=alert(1)>"
        "\\\\n\\\\n[a](javascript:alert(1)) [b](https://x.y) [c](/rel)\", {async: false})', ctx);\n"
        "console.log(html);\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        ["node", str(script), str(STATIC / "vendor" / "marked.min.js"), str(STATIC / "js" / "markdown.js")],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "<script" not in out and "<img" not in out and "&lt;script&gt;" in out
    assert 'href="javascript' not in out.lower() and 'href="https://x.y"' in out and 'href="/rel"' in out
