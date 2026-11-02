"""README, design note and docstrings of the writer, including the explicit cost limit."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "src" / "kaggle_agent" / "writer"
MANUAL_VARIABLE = "KAGGLE_AGENT_RUN_CLAUDE"


def readme_section():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Bundle writer")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def test_readme_has_a_bundle_writer_section_with_the_required_facts():
    section = readme_section()
    low = section.lower()
    for word in ("read", "write", "edit", "glob", "no bash", "max_repairs", "invalidbundleerror",
                 "bundleerror", "registry first", "sibling folder", "isolation limits",
                 "best effort", "bundle_model_dir", MANUAL_VARIABLE.lower(), "manual"):
        assert word in low or word.replace(" ", "") in low.replace(" ", ""), word
    assert "looked up in the registry first" in section
    assert "dollar cost of the writer calls is NOT recorded" in section
    assert "tokens, model and prompt hash are recorded" in section
    assert "follow-up decision" in section


def test_the_design_step_12_note_says_part_1_is_done_and_lists_what_remains():
    text = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("12. **Orchestrator"))
    assert "Part 1 done" in line and "Remaining" in line
    for word in ("orchestrator", "command line", "real lead input", "real submission"):
        assert word in line
    assert "- `writer`:" in text


def test_the_claude_writer_docstring_states_the_cost_limit():
    doc = ast.get_docstring(ast.parse((WRITER / "claude_writer.py").read_text(encoding="utf-8")))
    assert "cost of the writer calls is NOT recorded" in doc
    assert "tokens, model and prompt hash" in doc


def test_every_module_of_the_writer_has_a_docstring():
    for path in sorted(WRITER.glob("*.py")):
        assert ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))), path
