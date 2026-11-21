"""Structural guards of the orchestrator ticket: untouched packages, dependencies, scope (cases 36, 37, 39)."""

import ast
import json
import tomllib
from pathlib import Path

from kaggle_agent.orchestrator import codes
from kaggle_agent.sandbox.protected import hash_path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"
ORCHESTRATOR = SRC / "orchestrator"
TESTS = ROOT / "tests"

# content hashes of the packages this ticket must not touch (no git: the bytes themselves are pinned)
PINNED = {
    "evaluation": "c267079fdf0cd9fefa3501b7c4552e91ea72893df06e5641eedec112a6cb90ce",
    # ticket 27: updated deliberately for ONE line in loop/machine.py (CycleMachine.run re-resolves the
    # limits every turn so a limit extension reaches a running cycle); nothing else in loop/ changed
    "loop": "83835939181e704198356b794c2e9c4e2254ce583b12ef87ae381bc9c6ef851d",
    "pipeline": "612f737d5af5dd10e3ddd0c4071928abc4d2ebc77e5a9bc3b876ff941ee9543c",
    "writer": "2a40997bd9258bccd7d41da8ca0091ab39a404978867e9e021cf3fd5281286e4",
    "llm": "acbe904742217e7fc6062554462a0f9fb1b43a2a33a92d1a7adf9b741c25464a",
    "kaggle": "844b183fa141703d045b676cf886a7449655fad56ad375d72a1c4758dd8e594a",
    "bundle": "d082d3d2c7191c7a0d8ab7d0b4fea23a7bcce706f874b6fe490832705ebb7985",
    "registry": "04350ec5249dcffe22929c187ec50b13beb028406cd58a15dbeb15992890ef23",
}
NETWORK = {"socket", "urllib", "http", "requests", "httpx", "ftplib", "smtplib", "subprocess", "asyncio"}
REAL_ADAPTERS = {"RealKaggleClient", "RealClaudeClient", "SubprocessSandbox"}


def imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


# 37
def test_the_frozen_and_untouched_packages_keep_their_pinned_hashes():
    for name, expected in PINNED.items():
        assert hash_path(SRC / name) == expected, name


# 4 (ticket 26: the registry hash was updated deliberately for WAL and the busy timeout; the web
# pin is gone because the whole web package was rewritten, the structural web guard of
# test_no_modeling_logic.py replaces it)
def test_the_registry_pin_was_updated_deliberately_and_the_web_pin_is_gone():
    assert hash_path(SRC / "registry") == PINNED["registry"]
    for name in ("evaluation", "loop", "pipeline", "writer", "llm", "kaggle", "bundle"):
        assert name in PINNED
    text = (TESTS / "test_orchestrator_guards.py").read_text(encoding="utf-8")
    for word in ("WEB_" + "REST_HASH", "WEB_" + "EDITED", "web_" + "rest_hash"):
        assert word not in text  # the words are split above, so this file does not contain them


def test_dependencies_are_unchanged():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(d.split("==")[0] for d in pyproject["project"]["dependencies"])
    assert names == ["catboost", "flask", "kaggle", "lightgbm", "numpy", "pandas", "scikit-learn", "six", "xgboost"]


def test_the_orchestrator_name_is_used_only_by_its_own_package():
    for root in (SRC, TESTS):
        assert "orchestrator" not in {p.stem for p in root.rglob("*.py")}
    directories = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("orchestrator")
                   if p.is_dir() and ".venv" not in p.parts and "__pycache__" not in p.parts
                   and ".git" not in p.parts and "egg-info" not in str(p)}
    assert directories == {"src/kaggle_agent/orchestrator"}


def test_no_other_package_imports_the_orchestrator():
    for path in SRC.rglob("*.py"):
        if ORCHESTRATOR in path.parents:
            continue
        for name in imports(path):
            assert "orchestrator" not in name.split("."), (path, name)


def test_the_orchestrator_imports_no_network_module_and_builds_real_adapters_only_lazily():
    assert len(list(ORCHESTRATOR.glob("*.py"))) >= 10
    for path in sorted(ORCHESTRATOR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in imports(path):
            assert name.split(".")[0] not in NETWORK, (path, name)
            assert not name.endswith(("kaggle.real", "llm.real")) or path.name == "build.py", (path, name)
        # ticket 29: real_sandbox builds the subprocess sandbox alone (the importance job needs no Kaggle or Claude client)
        allowed = {n for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name in ("real_dependencies", "real_sandbox")
                   for n in ast.walk(f)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name in REAL_ADAPTERS:
                    assert node in allowed, (path.name, node.lineno)
        if path.name == "build.py":  # the real adapters are imported inside the function only
            top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            for node in top:
                assert not any(a.name in REAL_ADAPTERS for a in getattr(node, "names", [])), node.lineno


def test_the_lead_file_knows_only_the_plan_guidance_and_review_tables():
    from kaggle_agent.orchestrator import lead_file

    assert set(lead_file._TABLES) == {"plan", "guidance", "review"}
    assert "protocol_change" not in lead_file._TABLES["guidance"]


# 36
def test_exit_codes_are_distinct_and_documented():
    values = [getattr(codes, name) for name in ("FINISHED", "ERROR", "USAGE", "WAITING", "SUSPENDED",
                                                "LEAD_INVALID", "CRASHED", "NOT_ACCEPTED", "INVALID_FILE",
                                                "DAILY_LIMIT", "SCORE_PENDING", "MARKER", "PAUSED",
                                                "INTERRUPTED")]
    assert len(set(values)) == len(values) == len(codes.ALL)
    assert [codes.NOT_ACCEPTED, codes.INVALID_FILE, codes.DAILY_LIMIT, codes.SCORE_PENDING, codes.MARKER] == [
        20, 21, 22, 23, 24]
    assert (codes.WAITING, codes.SUSPENDED, codes.LEAD_INVALID, codes.CRASHED) == (10, 11, 12, 13)
    assert (codes.PAUSED, codes.INTERRUPTED) == (14, 15)


def test_the_tests_of_this_ticket_use_no_hard_coded_tmp_or_proc():
    # ticket 27: also the cockpit tests and their helpers (the existing ticket 26 files included)
    paths = set(TESTS.glob("*orchestrator*.py")) | set(TESTS.glob("test_cockpit*.py")) | {
        TESTS / "cockpit_live_helpers.py", TESTS / "cockpit_helpers.py", TESTS / "fake_run_main.py"}
    assert len(paths) > 10
    for path in sorted(paths):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "/proc" not in node.value and not node.value.startswith("/tmp"), (path.name, node.lineno)


# 39, 51
def test_only_the_manual_test_names_the_manual_variable():
    named, cockpit_named, cockpit_uses_real = [], [], []
    for path in sorted(TESTS.glob("*.py")):
        if path.name in {Path(__file__).name, "test_orchestrator_docs.py", "test_cockpit_docs.py"}:
            continue  # these only name the variables to check the others
        tree = ast.parse(path.read_text(encoding="utf-8"))
        values = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if "KAGGLE_AGENT_RUN_REAL" in values and "orchestrator" in path.name:
            named.append(path.name)
        if "KAGGLE_AGENT_RUN_REAL" in values and path.name.startswith("test_cockpit"):
            cockpit_uses_real.append(path.name)
        if "KAGGLE_AGENT_RUN_COCKPIT_DIR" in values:
            cockpit_named.append(path.name)
    assert named == ["test_orchestrator_manual_real.py"]
    assert cockpit_uses_real == []  # no cockpit test names the real-run variable
    assert cockpit_named == ["test_cockpit_manual.py"]  # ticket 27: the one cockpit manual variable
    manual = (TESTS / "test_orchestrator_manual_real.py").read_text(encoding="utf-8")
    assert manual.count("pytest.mark.skipif") == 2 and "KAGGLE_AGENT_RUN_REAL=1" in manual
    assert (TESTS / "MANUAL_COCKPIT_RUN.md").exists()
    assert "skipif" in (TESTS / "test_cockpit_manual.py").read_text(encoding="utf-8")


def test_the_test_map_is_valid_json():
    assert "cases" in json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))
