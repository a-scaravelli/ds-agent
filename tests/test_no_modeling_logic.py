"""Only proposals, registry, evaluation, kaggle, llm and sandbox have code; the rest is docstrings and stdlib."""

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
PROPOSALS = SRC / "proposals"
REGISTRY = SRC / "registry"
EVALUATION = SRC / "evaluation"
KAGGLE = SRC / "kaggle"
LLM = SRC / "llm"
WEB = SRC / "web"
LLM_REAL = LLM / "real.py"
AGENTS = SRC / "agents"
SANDBOX = SRC / "sandbox"
BUNDLE = SRC / "bundle"
LOOP = SRC / "loop"
BUDGET = SRC / "budget"
REVIEW = SRC / "review"
PIPELINE = SRC / "pipeline"
WRITER = SRC / "writer"
ORCHESTRATOR = SRC / "orchestrator"
SANDBOX_RUNNER = SANDBOX / "subprocess_sandbox.py"
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
        if BUNDLE in path.parents:
            allowed |= {"pandas", "numpy"}
        if PIPELINE in path.parents or ORCHESTRATOR in path.parents:
            allowed |= {"pandas"}
        if path == REAL_ADAPTER:
            allowed |= KAGGLE_LIBRARY
        if WEB in path.parents:
            allowed |= {"flask", "werkzeug"}  # werkzeug comes with Flask: web/__main__.py serves with it
        for top, _ in _imports(path):
            assert top in allowed, f"{path}: {top}"


def test_no_functions_or_classes_outside_code_packages_proposals_registry_evaluation_kaggle_llm_sandbox():
    for path in _modules():
        if any(d in path.parents for d in (PROPOSALS, REGISTRY, EVALUATION, KAGGLE, LLM, AGENTS, SANDBOX, WEB, BUNDLE,
                                       LOOP, BUDGET, REVIEW, PIPELINE, WRITER, ORCHESTRATOR)):
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
    assert names == ["catboost", "flask", "kaggle", "lightgbm", "numpy", "pandas", "scikit-learn", "six", "xgboost"]
    assert all("==" in d and d.split("==")[1][:1].isdigit() for d in deps)
    everything = " ".join(
        deps + [d for group in pyproject["project"].get("optional-dependencies", {}).values() for d in group]
    ).lower()
    # ticket 25: xgboost, lightgbm and catboost are runtime dependencies (bundles import them in the
    # sandbox child); no source module does. Only mlflow stays out.
    assert "mlflow" not in everything


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
        if LLM not in path.parents and path != SANDBOX_RUNNER:
            assert "subprocess" not in {top for top, _ in _imports(path)}, path
    for path in sorted(TESTS.glob("*.py")):
        if path.name != Path(__file__).name:
            assert "subprocess" not in {top for top, _ in _imports(path)} or path.name in {
                "llm_helpers.py", "test_llm_real.py", "test_pipeline_command.py", "test_pipeline_flow.py",
                "test_writer_manual_claude.py", "test_agents_limits.py",  # ticket 25: fresh-interpreter import check
            } or "sandbox" in path.name, path


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
    pinned = {"kaggle", "pandas", "numpy", "sklearn", "flask", "werkzeug"}  # werkzeug: part of Flask
    stdlib = set(sys.stdlib_module_names)
    for path in _modules():
        for top, full in _imports(path):
            assert top in stdlib | pinned | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
    for path in sorted(EVALUATION.glob("*.py")):
        for top, full in _imports(path):
            assert top not in {"tests", "evaluation_helpers", "registry_helpers", "pytest"}, path
            assert not full.startswith("kaggle_agent.llm"), f"{path}: {full}"
            assert "llm" not in full.lstrip(".").split("."), f"{path}: {full}"


def test_nothing_of_tickets_10_to_16_exists():
    # `loop`, `budget` and `review` are filled by step 9; the later steps are still absent
    for package in ("loop", "review", "budget"):
        assert (SRC / package / "__init__.py").is_file()
    # `runner`, `bundle` and `pipeline` modules exist only inside the bundle package (step 7)
    names = {p.stem for p in SRC.rglob("*.py") if BUNDLE not in p.parents and PIPELINE not in p.parents}
    assert not names & {"runner", "bundle", "pipeline"}
    assert not {p.stem for p in BUNDLE.rglob("*.py")} & {"pipeline", "report", "orchestrator"}
    # step 11 (report, holdout, pipeline) lives in the `pipeline` package only; no file has the stem
    # `orchestrator` (the step 12 package has that name, its modules do not)
    assert not {p.stem for p in SRC.rglob("*.py")} & {"orchestrator"}
    assert (PIPELINE / "__init__.py").is_file()
    later = {p.stem for p in LOOP.rglob("*.py")} | {p.stem for p in BUDGET.rglob("*.py")}
    later |= {p.stem for p in REVIEW.rglob("*.py")}
    assert not later & {"report", "orchestrator", "holdout", "pipeline", "runner", "bundle"}
    # step 12 part 2 adds the orchestrator package, and only at src/kaggle_agent/orchestrator
    assert not [p for p in SRC.iterdir() if p.is_dir() and p.name in {"runner"}]


# --- web package (step 10) ---


def _web_modules():
    return sorted(WEB.glob("*.py"))


def test_web_imports_only_stdlib_flask_proposals_registry_and_itself():
    stdlib = set(sys.stdlib_module_names)
    forbidden = {"evaluation", "llm", "kaggle", "sandbox", "loop", "review", "budget",
                 "orchestrator", "pipeline", "bundle"}
    assert _web_modules()
    for path in _web_modules():
        for top, full in _imports(path):
            assert top in stdlib | {"flask", "werkzeug", "kaggle_agent", "<relative>"}, f"{path}: {full}"
            assert top not in NETWORK_MODULES, f"{path}: {full}"
            if top == "kaggle_agent":
                parts = full.split(".")
                assert len(parts) > 1 and parts[1] in {"proposals", "registry", "web"}, full
                assert parts[1] not in forbidden, full
            if top == "<relative>":
                assert full.lstrip(".").split(".")[0] not in forbidden, f"{path}: {full}"


def test_only_the_web_start_module_imports_the_sqlite_registry():
    for path in _web_modules():
        sqlite_import = any(full.endswith("registry.sqlite") for _, full in _imports(path))
        assert sqlite_import == (path.name == "__main__.py"), path
    for path in _modules():
        if WEB in path.parents:
            continue
        for top, _ in _imports(path):
            assert top != "flask", path


# 32 (ticket 26): the structural web guard. The web package never reads a bundle folder: the only
# function that opens a file is data.read_project_file (project.json), and no path call gets a
# string about the workspace or the bundles folder. Written on the code, not on the words.

FILE_CALLS = {"open", "read_text", "read_bytes", "iterdir", "glob", "rglob", "listdir", "scandir",
              "walk", "write_text", "write_bytes", "unlink"}
PATH_CALLS = {"open", "Path", "PurePath", "PosixPath", "joinpath", "with_name", "relative_to"}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _mentions_bundle_folder(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            text = sub.value.lower()
            if "workspace" in text or "bundles/" in text or "/bundles" in text or text == "bundles":
                return True
    return False


def _enclosing_functions(tree: ast.AST):
    """Map every node to the name of the innermost function that contains it."""
    owner = {}

    def visit(node, current):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
            owner[child] = name
            visit(child, name)

    visit(tree, None)
    return owner


def test_web_opens_files_only_in_read_project_file():
    assert (WEB / "data.py").is_file()
    found = []
    for path in _web_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in FILE_CALLS:
                found.append((path.name, owner.get(node)))
    assert found, "the guard must see the one file read of the web package"
    assert set(found) == {("data.py", "read_project_file")}, found


def test_web_passes_no_workspace_or_bundles_string_to_a_file_or_path_call():
    for path in _web_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = _call_name(node)
                root = _root_name(node.func.value) if isinstance(node.func, ast.Attribute) else None
                if name in PATH_CALLS | FILE_CALLS or root == "os":
                    for arg in [*node.args, *(k.value for k in node.keywords)]:
                        assert not _mentions_bundle_folder(arg), (path.name, node.lineno)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):  # Path / "workspace"
                assert not _mentions_bundle_folder(node), (path.name, node.lineno)


def test_web_start_command_has_no_workspace_option():
    tree = ast.parse((WEB / "__main__.py").read_text(encoding="utf-8"))
    options = [
        arg.value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "add_argument"
        for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]
    assert "--db" in options and "--project" in options
    assert not [o for o in options if "workspace" in o]


def test_the_structural_guard_detects_what_it_guards():
    bad = ast.parse("def f(p):\n    return open(p)\n\nx = Path('a') / 'workspace'\ny = os.listdir('workspace/bundles/b1')")
    calls = [n for n in ast.walk(bad) if isinstance(n, ast.Call) and _call_name(n) in FILE_CALLS]
    assert len(calls) == 2
    assert any(isinstance(n, ast.BinOp) and _mentions_bundle_folder(n) for n in ast.walk(bad))


def test_sandbox_imports_only_stdlib_and_itself_and_subprocess_only_in_its_runner():
    stdlib = set(sys.stdlib_module_names)
    forbidden = {"registry", "evaluation", "kaggle", "llm", "loop", "review", "budget", "proposals"}
    for path in sorted(SANDBOX.glob("*.py")):
        for top, full in _imports(path):
            parts = full.lstrip(".").split(".")
            assert top in stdlib | {"kaggle_agent", "<relative>"}, f"{path}: {full}"
            assert top not in NETWORK_MODULES | THIRD_PARTY | KAGGLE_LIBRARY, f"{path}: {full}"
            assert "credentials" not in parts, f"{path}: {full}"
            if top == "kaggle_agent":
                assert parts[1] == "sandbox", f"{path}: {full}"
            if top == "<relative>":
                assert parts[0] not in forbidden, f"{path}: {full}"
            if top == "subprocess":
                assert path == SANDBOX_RUNNER, f"{path}: {full}"


def test_sandbox_defines_no_bundle_contract():
    text = "\n".join(p.read_text(encoding="utf-8") for p in SANDBOX.glob("*.py"))
    for word in ("def fit", "def predict", "class Bundle", "class Runner"):
        assert word not in text, word


# --- bundle package (step 7) ---


def test_bundle_imports_only_stdlib_pandas_numpy_and_allowed_packages():
    stdlib = set(sys.stdlib_module_names)
    allowed_packages = {"bundle", "sandbox", "registry", "evaluation", "proposals"}
    forbidden = {"llm", "kaggle", "web", "agents", "loop", "review", "budget"}
    assert list(BUNDLE.glob("*.py"))
    for path in sorted(BUNDLE.glob("*.py")):
        for top, full in _imports(path):
            parts = full.lstrip(".").split(".")
            assert top in stdlib | {"pandas", "numpy", "kaggle_agent", "<relative>"}, f"{path}: {full}"
            assert top not in NETWORK_MODULES | KAGGLE_LIBRARY | {"sklearn", "flask"}, f"{path}: {full}"
            assert "credentials" not in parts, f"{path}: {full}"
            if top == "kaggle_agent":
                assert len(parts) == 1 or parts[1] in allowed_packages, f"{path}: {full}"
            if top == "<relative>":
                assert parts[0] not in forbidden, f"{path}: {full}"
            if path.name == "child.py":
                assert top in stdlib | {"pandas", "numpy"}, f"{path}: {full}"
