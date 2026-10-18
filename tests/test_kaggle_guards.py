"""Guards on the Kaggle client: nothing submits by itself, scope of the ticket (no network)."""

import ast
import sys
from pathlib import Path

from test_no_modeling_logic import KAGGLE, NETWORK_MODULES, REAL_ADAPTER, SRC, _imports

TESTS = Path(__file__).resolve().parent
NEW_TESTS = sorted(TESTS.glob("test_kaggle_*.py")) + [TESTS / "kaggle_helpers.py"]


def _calls_of(tree, attribute):
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", None) == attribute or getattr(node.func, "id", None) == attribute)
    ]


def test_nothing_in_the_source_calls_submit():
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not _calls_of(tree, "submit"), path
        defined = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "submit"]
        assert not defined or path == KAGGLE / "interface.py", path


def test_no_module_outside_kaggle_imports_the_client():
    for path in SRC.rglob("*.py"):
        if KAGGLE in path.parents:
            continue
        for _, full in _imports(path):
            assert "kaggle" not in full.split(".")[1:2] or full == "kaggle_agent", f"{path}: {full}"


def test_only_the_template_method_reaches_the_send_hook():
    for path in KAGGLE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != "submit":
                assert not _calls_of(node, "_send"), (path, node.name)
    interface = ast.parse((KAGGLE / "interface.py").read_text(encoding="utf-8"))
    submit = next(n for n in ast.walk(interface) if isinstance(n, ast.FunctionDef) and n.name == "submit")
    called = {getattr(c.func, "id", getattr(c.func, "attr", None)) for c in ast.walk(submit) if isinstance(c, ast.Call)}
    assert {"validate_submission", "daily_limit", "_send"} <= called


def test_new_tests_import_neither_network_modules_nor_the_library():
    allowed_third_party = {"pytest"}
    stdlib = set(sys.stdlib_module_names)
    for path in NEW_TESTS:
        for top, full in _imports(path):
            assert top not in NETWORK_MODULES | {"kaggle"}, f"{path}: {full}"
            local = top in {"kaggle_agent", "<relative>"} or (TESTS / f"{top}.py").exists()
            assert top in stdlib | allowed_third_party or local, f"{path}: {full}"


def test_no_key_scanner_in_the_kaggle_package():
    # the key-like-string check belongs to the tickets that save artifacts
    for path in KAGGLE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert not any(w in node.name.lower() for w in ("scan", "keylike", "key_like")), node.name


def test_real_adapter_is_the_only_module_with_a_library_name():
    for path in KAGGLE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path != REAL_ADAPTER:
            assert "kaggle.api" not in text, path
