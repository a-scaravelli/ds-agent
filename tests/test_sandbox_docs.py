"""The sandbox documentation says what is enforced and what is not. No network."""

from pathlib import Path

from test_design_doc import parse_sections

from kaggle_agent.sandbox import Sandbox, SandboxResult

ROOT = Path(__file__).resolve().parent.parent


def _readme_section() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Light sandbox")
    return text[start : text.index("\n## ", start + 1)]


def test_never_retry_rule_is_in_the_result_docstring_and_the_readme():
    for text in (SandboxResult.__doc__, _readme_section()):
        low = " ".join(text.lower().split())
        assert "protected_modified" in low and "diff_rejected" in low
        assert "never be retried" in low
        assert "status" in low


def test_readme_states_enforced_convention_and_limits():
    low = _readme_section().lower()
    for phrase in ("enforced", "by convention", "does **not** guarantee", "same os user",
                   "network isolation", "filesystem isolation", "absolute path", "root", "partial", "mac"):
        assert phrase in low, phrase
    assert "python -i" in low and "explicitly" in low and "sandbox_protected_dir" in low
    doc = " ".join(Sandbox.__doc__.lower().split())
    assert "python -i" in doc and "explicitly" in doc


def test_design_mentions_the_light_sandbox_in_sections_8_15_16():
    sections = {n: body.lower() for n, _, body in parse_sections((ROOT / "docs" / "design.md").read_text(encoding="utf-8"))}
    for n in (8, 15, 16):
        assert "light sandbox" in sections[n], n
    assert "never retried" in sections[8]
