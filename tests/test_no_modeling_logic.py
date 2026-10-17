"""Only the proposals data model has code; the rest is docstrings and stdlib."""

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
PROPOSALS = SRC / "proposals"
LATER_TICKETS = {"registry", "evaluation", "kaggle", "llm"}
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


def test_no_third_party_imports():
    stdlib = set(sys.stdlib_module_names)
    for path in _modules():
        for top, _ in _imports(path):
            assert top in stdlib or top in {"kaggle_agent", "<relative>"}, f"{path}: {top}"


def test_no_functions_or_classes_outside_proposals():
    for path in _modules():
        if PROPOSALS in path.parents:
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
