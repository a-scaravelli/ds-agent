"""Only proposals, registry and evaluation have code; the rest is docstrings and stdlib."""

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
PROPOSALS = SRC / "proposals"
REGISTRY = SRC / "registry"
EVALUATION = SRC / "evaluation"
THIRD_PARTY = {"pandas", "numpy", "sklearn"}
LATER_TICKETS = {"registry", "evaluation", "kaggle", "llm"}
FUTURE_PACKAGES = {"kaggle", "llm", "sandbox", "loop", "review", "budget"}
NETWORK_MODULES = {
    "socket", "ssl", "http", "urllib", "ftplib", "smtplib", "xmlrpc",
    "requests", "httpx", "aiohttp", "asyncio", "telnetlib", "poplib", "imaplib",
}


def _modules():
    return sorted(SRC.rglob("*.py"))


def _imports(path):
    """Yield (top-level name, full dotted name) for every import in a module."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                module = node.module or ""
                yield module.split(".")[0], module
            else:
                # relative import inside proposals: `from .x import y` stays local
                yield "<relative>", "." * node.level + (node.module or "")


def test_third_party_imports_only_in_evaluation():
    stdlib = set(sys.stdlib_module_names)
    for path in _modules():
        allowed = stdlib | {"kaggle_agent", "<relative>"}
        if EVALUATION in path.parents:
            allowed |= THIRD_PARTY
        for top, _ in _imports(path):
            assert top in allowed, f"{path}: {top}"


def test_no_functions_or_classes_outside_code_packages():
    for path in _modules():
        if any(d in path.parents for d in (PROPOSALS, REGISTRY, EVALUATION)):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ), path
        for node in tree.body:
            ok = isinstance(node, (ast.Import, ast.ImportFrom)) or (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            ) or (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant))
            assert ok, f"{path}: logic not allowed"


def test_proposals_imports_nothing_from_later_tickets():
    for path in sorted(PROPOSALS.glob("*.py")):
        for top, full in _imports(path):
            assert top not in LATER_TICKETS, f"{path}: {full}"
            if top == "kaggle_agent":
                sub = full.split(".")[1:2]
                assert not sub or sub[0] not in LATER_TICKETS | {
                    "loop", "sandbox", "review", "budget",
                }, f"{path}: {full}"
            if top == "<relative>":
                assert full.lstrip(".").split(".")[0] not in LATER_TICKETS, f"{path}: {full}"


def test_proposals_uses_no_network_modules():
    for path in sorted(PROPOSALS.glob("*.py")):
        for top, full in _imports(path):
            assert top not in NETWORK_MODULES, f"{path}: {full}"


def test_registry_imports_only_stdlib_proposals_and_itself():
    stdlib = set(sys.stdlib_module_names)
    forbidden = LATER_TICKETS | {"loop", "sandbox", "review", "budget"}
    for path in sorted(REGISTRY.glob("*.py")):
        for top, full in _imports(path):
            assert top in stdlib or top in {"kaggle_agent", "<relative>"}, f"{path}: {full}"
            assert top not in NETWORK_MODULES, f"{path}: {full}"
            if top == "kaggle_agent":
                parts = full.split(".")
                assert len(parts) > 1 and parts[1] in {"proposals", "registry"}, f"{path}: {full}"
            if top == "<relative>":
                assert full.lstrip(".").split(".")[0] not in forbidden - {"registry"}, f"{path}: {full}"


def test_runtime_dependencies_are_exactly_the_three_pinned_libraries():
    import tomllib

    pyproject = tomllib.loads((SRC.parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    names = sorted(d.split("==")[0] for d in deps)
    assert names == ["numpy", "pandas", "scikit-learn"]
    assert all("==" in d and d.split("==")[1][:1].isdigit() for d in deps)
    everything = " ".join(
        deps + [d for group in pyproject["project"].get("optional-dependencies", {}).values() for d in group]
    ).lower()
    assert "lightgbm" not in everything and "mlflow" not in everything


def test_evaluation_imports_only_allowed_modules():
    stdlib = set(sys.stdlib_module_names)
    for path in sorted(EVALUATION.glob("*.py")):
        for top, full in _imports(path):
            assert top in stdlib | THIRD_PARTY | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
            assert top not in NETWORK_MODULES, f"{path}: {full}"
            if top == "kaggle_agent":
                parts = full.split(".")
                assert len(parts) > 1 and parts[1] in {"proposals", "registry", "evaluation"}, (
                    f"{path}: {full}"
                )
            if top == "<relative>":
                assert full.lstrip(".").split(".")[0] not in FUTURE_PACKAGES, f"{path}: {full}"


def test_registry_and_proposals_stay_stdlib_only():
    for package in (PROPOSALS, REGISTRY):
        for path in sorted(package.glob("*.py")):
            for top, full in _imports(path):
                assert top not in THIRD_PARTY, f"{path}: {full}"
