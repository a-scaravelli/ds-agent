"""Structural guards of the writer package and of the ticket constraints.

This file and ``test_writer_docs.py`` only name the manual-test environment variable; no other
writer test may read the environment.
"""

import ast
import json
import tomllib
from pathlib import Path

from kaggle_agent.sandbox.protected import hash_path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"
WRITER = SRC / "writer"
TESTS = ROOT / "tests"
PLAN_CASES = 31

# content hashes of the two packages this ticket must not touch, taken at the start of the ticket
LOOP_HASH = "65394233b94a92e7d41e87bb7c33bc374a4257a8daf6e05f92af053ee8719ae8"
EVALUATION_SOURCES_HASH = "c267079fdf0cd9fefa3501b7c4552e91ea72893df06e5641eedec112a6cb90ce"
MANUAL_VARIABLE = "KAGGLE_AGENT_RUN_CLAUDE"


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def imported(path):
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


def writer_modules():
    return sorted(WRITER.glob("*.py"))


def test_the_writer_imports_nothing_from_evaluation_or_pipeline_and_no_heavy_library():
    assert {p.name for p in writer_modules()} >= {"claude_writer.py", "workspace.py", "checks.py", "prompt.py"}
    for path in writer_modules():
        for name in imported(path):
            parts = name.lstrip(".").split(".")
            assert parts[0] not in {"pandas", "numpy", "subprocess", "sklearn", "socket", "urllib"}, (path, name)
            if parts[0] == "kaggle_agent":
                assert parts[1] not in {"evaluation", "pipeline", "web", "kaggle"}, (path, name)


def test_the_writer_never_calls_the_engine_the_holdout_or_the_evaluation():
    forbidden = ("holdout", "evaluate", "freeze", "engine", "session", "verify")
    for path in writer_modules():
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                assert not any(word in name.lower() for word in forbidden), (path, name)


def test_the_loop_and_evaluation_packages_are_unchanged_and_dependencies_are_the_pinned_ones():
    assert hash_path(SRC / "loop") == LOOP_HASH
    assert hash_path(SRC / "evaluation") == EVALUATION_SOURCES_HASH
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(d.split("==")[0] for d in pyproject["project"]["dependencies"])
    assert names == ["flask", "kaggle", "numpy", "pandas", "scikit-learn"]


def test_scope_limits_no_orchestrator_command_or_tools_for_other_roles():
    stems = {p.stem for p in SRC.rglob("*.py")}
    assert "orchestrator" not in stems
    assert not (WRITER / "__main__.py").exists()
    writers = [p for p in SRC.rglob("*.py") if "BundleWriter)" in p.read_text(encoding="utf-8")]
    assert {p.name for p in writers} == {"fake.py", "claude_writer.py"}
    for path in sorted((SRC / "agents").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "WRITER_TOOLS" not in text and "writer" not in {n for n in imported(path)}, path
        assert '"Write"' not in text and '"Bash"' not in text, path


def environment_scanned_files():
    skipped = {"test_writer_manual_claude.py", "test_writer_guards.py", "test_writer_docs.py"}
    files = [p for p in sorted(TESTS.glob("test_writer_*.py")) if p.name not in skipped]
    return [*files, TESTS / "writer_helpers.py"]


def test_writer_tests_read_no_environment_and_assume_no_proc_or_tmp():
    files = environment_scanned_files()
    assert len(files) >= 9
    for path in files:
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"environ", "getenv"}, (path.name, node.lineno)
            elif isinstance(node, ast.Name):
                assert node.id not in {"environ", "getenv"}, (path.name, node.lineno)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "/proc" not in node.value and not node.value.startswith("/tmp"), (path.name, node.lineno)
            assert not (isinstance(node, ast.Constant) and node.value == MANUAL_VARIABLE), path.name


def test_only_the_manual_test_reads_the_manual_variable():
    readers = []
    for path in sorted([*TESTS.glob("test_writer_*.py"), TESTS / "writer_helpers.py"]):
        if path.name in {"test_writer_guards.py", "test_writer_docs.py"}:
            continue
        reads = any(isinstance(n, ast.Attribute) and n.attr in {"environ", "getenv"} for n in ast.walk(parse(path)))
        if reads:
            readers.append(path.name)
    assert readers == ["test_writer_manual_claude.py"]


def test_the_test_map_holds_exactly_the_plan_cases():
    cases = json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))["cases"]
    keys = sorted(int(k) for k in cases)
    assert keys == list(range(1, PLAN_CASES + 1))
