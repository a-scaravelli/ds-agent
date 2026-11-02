"""Structural guards of loop, budget and review, and their documentation (cases 30, 31, 34)."""

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

from kaggle_agent.evaluation import evaluation_code_hash

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"
README = (ROOT / "README.md").read_text(encoding="utf-8")
STDLIB = set(sys.stdlib_module_names)
NETWORK = {"socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp", "asyncio",
           "subprocess", "multiprocessing", "threading", "ftplib", "smtplib"}
THIRD_PARTY = {"pandas", "numpy", "sklearn", "flask", "kaggle"}
# the hash of the frozen evaluation sources: this ticket must not change them
EVALUATION_HASH = "c267079fdf0cd9fefa3501b7c4552e91ea72893df06e5641eedec112a6cb90ce"


def imports(path):
    """(top-level name, dotted internal package or module) of every import of a module."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield "<relative>", "." * node.level + (node.module or "")
            else:
                yield (node.module or "").split(".")[0], node.module or ""


def internal(path):
    """Packages of ``kaggle_agent`` a module imports (``loop`` for ``kaggle_agent.loop.x``)."""
    found = set()
    for top, full in imports(path):
        if top == "kaggle_agent":
            parts = full.split(".")
            if len(parts) > 1:
                found.add(parts[1])
        elif top == "<relative>" and full.startswith(".."):
            found.add(full.lstrip(".").split(".")[0])
    return found


def modules(package):
    return sorted((SRC / package).glob("*.py"))


@pytest.mark.parametrize("package", ["budget", "review"])
def test_budget_and_review_import_only_the_standard_library_and_proposals(package):
    assert len(modules(package)) >= 2
    for path in modules(package):
        for top, full in imports(path):
            assert top in STDLIB | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
            assert top not in NETWORK, f"{path}: {full}"
        assert internal(path) <= {"proposals"}, path


def test_loop_imports_only_the_allowed_internal_packages():
    allowed = {"agents", "bundle", "evaluation", "registry", "proposals", "budget", "review", "llm"}
    assert {p.name for p in modules("loop")} >= {"machine.py", "round.py", "candidate.py", "derive.py"}
    for path in modules("loop"):
        for top, full in imports(path):
            assert top in STDLIB | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
            assert top not in NETWORK | THIRD_PARTY, f"{path}: {full}"
            assert "kaggle_agent.llm.real" not in full and "kaggle_agent.web" not in full, path
        assert internal(path) <= allowed, (path, internal(path) - allowed)
    # a sandbox is reached only through the bundle runner passed in as a factory
    text = "\n".join(p.read_text(encoding="utf-8") for p in modules("loop"))
    assert "subprocess" not in text and "Popen" not in text


def test_no_other_package_imports_loop_budget_or_review():
    for path in sorted(SRC.rglob("*.py")):
        package = path.relative_to(SRC).parts[0]
        if package == "loop":
            continue
        used = internal(path)
        if package == "pipeline":  # step 11 drives the loop; test_pipeline_guards limits what it takes
            assert "review" not in used, path
            continue
        assert "loop" not in used, path
        assert not used & {"budget", "review"}, path
    # budget and review do not import each other or the loop
    for package in ("budget", "review"):
        for path in modules(package):
            assert internal(path) <= {"proposals"}, path


def test_frozen_sources_are_unchanged_and_no_later_step_is_present():
    assert evaluation_code_hash() == EVALUATION_HASH
    stems = {p.stem for p in SRC.rglob("*.py") if p.relative_to(SRC).parts[0] != "pipeline"}
    assert not stems & {"pipeline", "report", "orchestrator", "holdout", "export"}
    assert "orchestrator" not in {p.stem for p in SRC.rglob("*.py")}  # step 12 is absent everywhere
    # agents never import the loop: roles are called by the pipeline, not the other way round
    for path in sorted((SRC / "agents").glob("*.py")):
        assert "loop" not in internal(path), path


def test_the_loop_has_no_modeling_logic_of_its_own():
    forbidden = ("sklearn", "RandomForest", "LogisticRegression", "fit_transform")
    for path in modules("loop"):
        text = path.read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in text, (path.name, word)


@pytest.mark.parametrize("name", ["loop", "budget", "review"])
def test_public_names_have_docstrings(name):
    package = importlib.import_module(f"kaggle_agent.{name}")
    assert package.__doc__ and len(package.__doc__.split()) > 20
    for public in package.__all__:
        item = getattr(package, public)
        if callable(item) and not isinstance(item, (str, int)):
            assert item.__doc__, f"{name}.{public} has no docstring"
    for path in modules(name):
        assert ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))), path


def test_readme_documents_the_pipeline_the_open_interfaces_and_the_limits():
    assert "## Round pipeline and guidance" in README
    section = README.split("## Round pipeline and guidance")[1].split("\n## ")[0]
    for word in (
        "CycleMachine", "LeadInput", "BundleWriter", "FakeBundleWriter", "`<cycle>-R0`",
        "provisional", "Cycle.parameters", "placeholder", "new plan approval", "AWAITING_GUIDANCE",
        "step 12", "open question for the lead", "between `add_attempt`", "never retried",
    ):
        assert word in section, word
    low = " ".join(section.split()).lower()
    assert "claude-driven code writer" in low and "real command or file input" in low
    assert "not built" in low


def test_the_test_map_holds_the_cases_of_the_plan():
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(int(k) for k in cases) == list(range(1, 36))
