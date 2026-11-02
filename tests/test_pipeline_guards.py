"""Structural guards of the pipeline package and of the ticket (cases 10, 20, 32, 33, 34)."""

import ast
import json
import sys
import tomllib
from pathlib import Path

from test_loop_guards import EVALUATION_HASH, NETWORK, imports, internal

from kaggle_agent.evaluation import evaluation_code_hash

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"
TESTS = ROOT / "tests"
PIPELINE = SRC / "pipeline"
STDLIB = set(sys.stdlib_module_names)
NEW_TESTS = sorted([*TESTS.glob("test_pipeline_*.py"), TESTS / "test_bundle_persistence.py",
                    TESTS / "pipeline_helpers.py"])


def pipeline_modules():
    return sorted(PIPELINE.glob("*.py"))


def test_pipeline_imports_only_the_allowed_packages_and_no_network_or_web():
    assert {p.name for p in pipeline_modules()} >= {"flow.py", "report.py", "holdout.py", "persisted.py",
                                                    "suggestions.py", "command.py", "__main__.py"}
    allowed = {"bundle", "budget", "evaluation", "loop", "proposals", "registry", "sandbox", "pipeline"}
    for path in pipeline_modules():
        for top, full in imports(path):
            assert top in STDLIB | {"kaggle_agent", "<relative>", "pandas"}, f"{path}: {full}"
            assert top not in NETWORK - {"subprocess"}, f"{path}: {full}"
            assert top not in {"kaggle", "flask", "numpy", "sklearn"}, f"{path}: {full}"
            assert "kaggle_agent.llm.real" not in full and "kaggle_agent.web" not in full, f"{path}: {full}"
            assert "subprocess" not in full, path
        assert internal(path) <= allowed | {""}, (path, internal(path) - allowed)
        assert "review" not in internal(path) and "web" not in internal(path)


def test_pipeline_takes_only_the_public_driving_names_from_the_loop():
    taken = set()
    for path in pipeline_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[:2] == ["kaggle_agent", "loop"]:
                assert node.module == "kaggle_agent.loop" or node.module == "kaggle_agent.loop.states", path
                taken |= {a.name for a in node.names}
    assert taken <= {"CycleContext", "CycleMachine", "CycleOutcome", "CycleView", "Hooks", "PLANNING"}, taken
    assert {"Hooks", "CycleContext", "CycleMachine", "CycleView"} <= taken


def test_no_pickle_marshal_or_shelve_anywhere_and_predictions_are_read_as_json():
    banned = {"pickle", "marshal", "shelve", "cPickle", "dill", "joblib", "cloudpickle"}
    for path in sorted(SRC.rglob("*.py")):
        names = {top for top, _ in imports(path)}
        assert not names & banned, path
        text = path.read_text(encoding="utf-8")
        assert "pickle.load" not in text and "np.load(" not in text, path
    runner = (SRC / "bundle" / "runner.py").read_text(encoding="utf-8")
    assert "json.loads" in runner and "read_pickle" not in runner
    for path in pipeline_modules():
        assert "read_pickle" not in path.read_text(encoding="utf-8"), path


def test_evaluation_sources_and_the_frozen_hash_are_unchanged():
    assert evaluation_code_hash() == EVALUATION_HASH
    # nothing in the pipeline reaches into evaluation internals other than the public names
    for path in pipeline_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("kaggle_agent.evaluation."):
                raise AssertionError(f"{path}: import from the evaluation internals {node.module}")


def test_new_tests_use_no_proc_no_fixed_tmp_and_run_children_isolated():
    proc, tmp = "/" + "proc", "/" + "tmp"
    for path in NEW_TESTS:
        text = path.read_text(encoding="utf-8")
        assert proc not in text and tmp not in text, path
    command = (TESTS / "test_pipeline_command.py").read_text(encoding="utf-8")
    assert '"-I"' in command  # the module test starts its child in isolated mode
    for path in pipeline_modules():
        assert proc not in path.read_text(encoding="utf-8"), path
    helpers = (TESTS / "pipeline_helpers.py").read_text(encoding="utf-8")
    assert "tmp_path" in helpers and "timeout_seconds=120" in helpers  # generous timeouts


def test_persisted_paths_in_the_pipeline_are_resolved():
    flow = (PIPELINE / "flow.py").read_text(encoding="utf-8")
    persisted = (PIPELINE / "persisted.py").read_text(encoding="utf-8")
    assert ".resolve()" in flow and ".resolve()" in persisted  # /var -> /private/var on macOS



def test_out_of_scope_stays_out_and_dependencies_are_unchanged():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(d.split("==")[0] for d in pyproject["project"]["dependencies"])
    assert names == ["flask", "kaggle", "numpy", "pandas", "scikit-learn"]
    stems = {p.stem for p in SRC.rglob("*.py")}
    assert "orchestrator" not in stems
    for path in sorted(SRC.rglob("*.py")):
        assert "orchestrator" not in {full.split(".")[-1] for _, full in imports(path)}, path
    # the only bundle writers are the loop fake, the interface module and the Claude-driven writer
    writers = [p for p in SRC.rglob("*.py") if "BundleWriter)" in p.read_text(encoding="utf-8")]
    assert {p.name for p in writers} <= {"fake.py", "writer.py", "claude_writer.py"}
    assert all("pipeline" not in p.parts for p in writers)


def test_readme_and_design_document_the_new_behaviour_and_public_names_have_docstrings():
    import importlib

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Final report and pipeline" in readme
    section = readme.split("## Final report and pipeline")[1].split("\n## ")[0]
    for word in ("python -m kaggle_agent.pipeline reproduce", "python -m kaggle_agent.pipeline predict",
                 "BUNDLE_MODEL_DIR", "pipeline:<persisted dir>", "multiclass `roc_auc`", "neither command re-reads it",
                 "does not carry over the best model or the history", "What it proves", "HoldoutOutcome",
                 "no usable attempt", "<cycle>-sg-<kind>", "the holdout is read once per cycle"):
        assert word.lower() in section.lower() or word.lower() in readme.lower(), word  # the section or the bundle part
    bundle = readme.split("## Bundle contract and runner")[1].split("\n## ")[0]
    for word in ("BUNDLE_MODEL_DIR", "two different child processes", "multiclass", "scoring_error", "JSON"):
        assert word in bundle or word.lower() in bundle.lower(), word
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    assert "Step 11 (done, level 1)" in design and "`HoldoutOutcome`" in design
    assert "`add_execution_artifact`" in design and "`BUNDLE_MODEL_DIR`" in design
    package = importlib.import_module("kaggle_agent.pipeline")
    assert package.__doc__ and len(package.__doc__.split()) > 20
    for name in package.__all__:
        item = getattr(package, name)
        if callable(item):
            assert item.__doc__, f"pipeline.{name} has no docstring"
    for path in pipeline_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), path
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                assert ast.get_docstring(node), f"{path.name}: {node.name}"
