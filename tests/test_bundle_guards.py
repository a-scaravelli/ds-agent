"""Guards on the bundle package: sandbox only, no new library, nothing of later steps (cases 12, 22 to 24)."""

import ast
import json
import re
import tomllib
from pathlib import Path

from bundle_helpers import BUNDLE_DIR

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def sources():
    return sorted(BUNDLE_DIR.glob("*.py"))


def imported(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield "." * node.level + (node.module or "")


def test_no_other_subprocess_mechanism_in_the_bundle_package():
    forbidden = {"subprocess", "multiprocessing", "concurrent", "threading", "pty", "socket"}
    for path in sources():
        tops = {name.split(".")[0] for name in imported(path)}
        assert not tops & forbidden, path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "fork", "execv", "spawnv"}:
                raise AssertionError(f"{path}: {node.attr}")
    assert "sandbox" in (BUNDLE_DIR / "runner.py").read_text(encoding="utf-8")


def test_child_script_does_not_import_the_agent_and_other_modules_stay_in_bounds():
    for name in imported(BUNDLE_DIR / "child.py"):
        assert not name.startswith(("kaggle_agent", ".")), name
        assert name.split(".")[0] in {"importlib", "json", "os", "random", "sys", "pathlib", "numpy", "pandas"}
    for path in sources():
        for name in imported(path):
            assert name.split(".")[0] not in {"sklearn", "flask", "kaggle"}, (path, name)
            if name.startswith("kaggle_agent."):
                assert name.split(".")[1] in {"bundle", "sandbox", "registry", "evaluation", "proposals"}, name


def test_dependencies_unchanged_and_nothing_of_later_steps_exists():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(d.split("==")[0] for d in pyproject["project"]["dependencies"])
    assert names == ["flask", "kaggle", "numpy", "pandas", "scikit-learn"]
    stems = {p.stem for p in SRC.rglob("*.py")}
    assert not stems & {"pipeline", "report", "orchestrator"}
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources()).lower()
    for word in ("virtualenv", "venv", "docker", "pip install", "immutable"):
        assert word not in text.replace("(immutable bundles, virtual environments, containers)", ""), word
    for package in ("loop", "review", "budget"):  # filled by step 9, each with a documented package
        init = SRC / package / "__init__.py"
        assert init.is_file() and ast.get_docstring(ast.parse(init.read_text(encoding="utf-8")))


def test_readme_documents_the_bundle_contract_and_known_limits():
    assert "## Bundle contract and runner" in README
    section = README.split("## Bundle contract and runner")[1].split("\n## ")[0]
    for word in ("bundle.json", "fit(train)", "predict(bundle_id, X)", "protected_modified", "diff_rejected",
                 "expected_hashes", "never retried"):
        assert word in section, word
    limits = README.split("## Known limits")[1]
    assert re.search(r"pandas.*importable.*sys\.executable.*user site.*python -I", limits, re.S)
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    assert "`bundle`" in design


def test_test_map_holds_the_plan_cases():
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(int(k) for k in cases) == list(range(1, len(cases) + 1))


def test_bundle_tests_avoid_linux_only_assumptions():
    this = Path(__file__).name
    for path in sorted(Path(__file__).parent.glob("*bundle*.py")):
        if path.name == this:
            continue
        text = path.read_text(encoding="utf-8")
        assert "/proc" not in text and "/tmp" not in text, path
