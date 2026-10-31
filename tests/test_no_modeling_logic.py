"""Only proposals, registry, evaluation, kaggle and llm have code; the rest is docstrings and stdlib."""

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
PROPOSALS = SRC / "proposals"
REGISTRY = SRC / "registry"
EVALUATION = SRC / "evaluation"
KAGGLE = SRC / "kaggle"
LLM = SRC / "llm"
LLM_REAL = LLM / "real.py"
TESTS = Path(__file__).resolve().parent
REAL_ADAPTER = KAGGLE / "real.py"
KAGGLE_LIBRARY = {"kaggle"}
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


def test_third_party_imports_only_in_evaluation_and_kaggle_adapter():
    stdlib = set(sys.stdlib_module_names)
    for path in _modules():
        allowed = stdlib | {"kaggle_agent", "<relative>"}
        if EVALUATION in path.parents:
            allowed |= THIRD_PARTY
        if path == REAL_ADAPTER:
            allowed |= KAGGLE_LIBRARY
        for top, _ in _imports(path):
            assert top in allowed, f"{path}: {top}"


def test_no_functions_or_classes_outside_code_packages_proposals_registry_evaluation_kaggle_llm():
    for path in _modules():
        if any(d in path.parents for d in (PROPOSALS, REGISTRY, EVALUATION, KAGGLE, LLM)):
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


def test_runtime_dependencies_are_the_pinned_libraries():
    import tomllib

    pyproject = tomllib.loads((SRC.parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    names = sorted(d.split("==")[0] for d in deps)
    assert names == ["kaggle", "numpy", "pandas", "scikit-learn"]
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


def test_registry_never_imports_credentials_or_kaggle():
    for path in sorted(REGISTRY.glob("*.py")):
        for top, full in _imports(path):
            assert "credentials" not in full and "kaggle" not in full.split("."), f"{path}: {full}"


def _kaggle_modules():
    return sorted(KAGGLE.glob("*.py"))


def test_only_kaggle_real_adapter_imports_network_or_library():
    for path in _kaggle_modules():
        for top, full in _imports(path):
            if path == REAL_ADAPTER:
                continue
            assert top not in NETWORK_MODULES | KAGGLE_LIBRARY | THIRD_PARTY, f"{path}: {full}"
            assert top in set(sys.stdlib_module_names) | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
    for top, full in _imports(REAL_ADAPTER):
        assert top in set(sys.stdlib_module_names) | {"kaggle_agent", "<relative>"}, full


def test_kaggle_imports_no_later_ticket_package_and_only_recording_imports_registry():
    forbidden = {"llm", "sandbox", "loop", "review", "budget", "evaluation"}
    for path in _kaggle_modules():
        for top, full in _imports(path):
            parts = full.lstrip(".").split(".")
            if top == "kaggle_agent":
                assert parts[1] not in forbidden, f"{path}: {full}"
                if parts[1] == "registry":
                    assert path.name == "recording.py", f"{path}: {full}"
                else:
                    assert parts[1] in {"kaggle"}, f"{path}: {full}"
            if top == "<relative>":
                assert parts[0] not in forbidden | {"registry"}, f"{path}: {full}"


def test_only_kaggle_modules_import_credentials():
    for path in _modules():
        if KAGGLE in path.parents:
            continue
        for _, full in _imports(path):
            assert "credentials" not in full.split("."), f"{path}: {full}"


def _llm_modules():
    return sorted(LLM.glob("*.py"))


def test_only_llm_real_imports_subprocess_and_llm_uses_no_network_or_third_party():
    stdlib = set(sys.stdlib_module_names)
    for path in _llm_modules():
        for top, full in _imports(path):
            assert top not in NETWORK_MODULES | THIRD_PARTY | KAGGLE_LIBRARY, f"{path}: {full}"
            assert top in stdlib | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
            if top == "subprocess":
                assert path == LLM_REAL, f"{path}: {full}"
    for path in _modules():
        if LLM not in path.parents:
            assert "subprocess" not in {top for top, _ in _imports(path)}, path
    for path in sorted(TESTS.glob("*.py")):
        if path.name != Path(__file__).name:
            assert "subprocess" not in {top for top, _ in _imports(path)} or path.name in {
                "llm_helpers.py", "test_llm_real.py",
            }, path


def test_importing_the_llm_package_does_not_import_subprocess_real_module():
    import kaggle_agent.llm as llm

    assert not hasattr(llm, "RealClaudeClient")
    imported = {full for _, full in _imports(Path(llm.__file__))}
    assert not {name for name in imported if "real" in name or "subprocess" in name}


def test_llm_imports_no_later_ticket_package_and_only_recording_imports_registry():
    forbidden = {"loop", "sandbox", "budget", "review", "evaluation", "kaggle"}
    for path in _llm_modules():
        for top, full in _imports(path):
            parts = full.lstrip(".").split(".")
            assert "credentials" not in parts, f"{path}: {full}"
            if top == "kaggle_agent":
                assert parts[1] not in forbidden, f"{path}: {full}"
                if parts[1] == "registry":
                    assert path.name == "recording.py", f"{path}: {full}"
                else:
                    assert parts[1] in {"llm", "proposals"}, f"{path}: {full}"
            if top == "<relative>":
                assert parts[0] not in forbidden | {"registry"}, f"{path}: {full}"


def test_llm_has_no_budget_enforcement_or_per_phase_model_policy():
    text = "\n".join(p.read_text(encoding="utf-8") for p in _llm_modules()).lower()
    for word in ("max_budget", "budget_exceeded", "token_limit", "max_tokens", "phase_model"):
        assert word not in text, word


def test_no_test_builds_a_real_client_without_a_runner():
    for path in sorted(TESTS.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name == "RealClaudeClient":
                    keys = {k.arg for k in node.keywords}
                    assert "runner" in keys, f"{path.name}:{node.lineno}"


def test_evaluation_imports_neither_tests_nor_llm_and_no_unpinned_library_anywhere():
    pinned = {"kaggle", "pandas", "numpy", "sklearn"}
    stdlib = set(sys.stdlib_module_names)
    for path in _modules():
        for top, full in _imports(path):
            assert top in stdlib | pinned | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
    for path in sorted(EVALUATION.glob("*.py")):
        for top, full in _imports(path):
            assert top not in {"tests", "evaluation_helpers", "registry_helpers", "pytest"}, path
            assert not full.startswith("kaggle_agent.llm"), f"{path}: {full}"
            assert "llm" not in full.lstrip(".").split("."), f"{path}: {full}"


def test_nothing_of_tickets_6_to_16_exists():
    for package in ("sandbox", "loop", "review", "budget"):
        files = [p for p in (SRC / package).rglob("*.py")]
        assert [p.name for p in files] == ["__init__.py"], package
    names = {p.stem for p in SRC.rglob("*.py")}
    assert not names & {"runner", "bundle", "agents", "pipeline", "web", "app", "server"}
    assert not [p for p in SRC.iterdir() if p.is_dir() and p.name in {"web", "agents", "runner"}]
