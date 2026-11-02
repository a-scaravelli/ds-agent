"""Structural guards of the orchestrator ticket: untouched packages, dependencies, scope (cases 36, 37, 39)."""

import ast
import hashlib
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
    "loop": "65394233b94a92e7d41e87bb7c33bc374a4257a8daf6e05f92af053ee8719ae8",
    "pipeline": "612f737d5af5dd10e3ddd0c4071928abc4d2ebc77e5a9bc3b876ff941ee9543c",
    "writer": "7c55b7ad8aef4c3e0e2a63d07f1fbec4b5b0262add989c6179392899c28a970e",
    "llm": "3a218b7783b9fddd051703819e47b94a7f09ffe10c4f678fb992c1138e20dddc",
    "kaggle": "1de2fec26529fbc423dcdea2757f02223e88b80a15010e3b9204e28308ac70d4",
    "bundle": "d082d3d2c7191c7a0d8ab7d0b4fea23a7bcce706f874b6fe490832705ebb7985",
    "registry": "ab5bc75eebc24452575e9fe4a383246745a0808d5252b77cbbf41c43ac607cef",
}
WEB_EDITED = {"views.py", "cycle.html"}
WEB_REST_HASH = "da5a0886fa9377bf9080fee7ea6e800cd63533235f5bbdcb04ba8c81676fa86f"
NETWORK = {"socket", "urllib", "http", "requests", "httpx", "ftplib", "smtplib", "subprocess", "asyncio"}
REAL_ADAPTERS = {"RealKaggleClient", "RealClaudeClient", "SubprocessSandbox"}


def web_rest_hash() -> str:
    web = SRC / "web"
    digest = hashlib.sha256()
    for path in sorted(web.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name not in WEB_EDITED:
            digest.update(path.relative_to(web).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


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
    assert web_rest_hash() == WEB_REST_HASH


def test_dependencies_are_unchanged():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(d.split("==")[0] for d in pyproject["project"]["dependencies"])
    assert names == ["flask", "kaggle", "numpy", "pandas", "scikit-learn"]


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
        allowed = {n for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "real_dependencies"
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
                                                "DAILY_LIMIT", "SCORE_PENDING", "MARKER")]
    assert len(set(values)) == len(values) == len(codes.ALL)
    assert [codes.NOT_ACCEPTED, codes.INVALID_FILE, codes.DAILY_LIMIT, codes.SCORE_PENDING, codes.MARKER] == [
        20, 21, 22, 23, 24]
    assert (codes.WAITING, codes.SUSPENDED, codes.LEAD_INVALID, codes.CRASHED) == (10, 11, 12, 13)


def test_the_tests_of_this_ticket_use_no_hard_coded_tmp_or_proc():
    for path in TESTS.glob("*orchestrator*.py"):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "/proc" not in node.value and not node.value.startswith("/tmp"), (path.name, node.lineno)


# 39
def test_only_the_manual_test_names_the_manual_variable():
    named = []
    for path in sorted(TESTS.glob("*orchestrator*.py")):
        if path.name in {Path(__file__).name, "test_orchestrator_docs.py"}:
            continue  # these two only name the variable to check the others
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(n, ast.Constant) and n.value == "KAGGLE_AGENT_RUN_REAL" for n in ast.walk(tree)):
            named.append(path.name)
    assert named == ["test_orchestrator_manual_real.py"]
    manual = (TESTS / "test_orchestrator_manual_real.py").read_text(encoding="utf-8")
    assert manual.count("pytest.mark.skipif") == 2 and "KAGGLE_AGENT_RUN_REAL=1" in manual


def test_the_test_map_is_valid_json():
    assert "cases" in json.loads((ROOT / ".asl" / "test-map.json").read_text(encoding="utf-8"))
