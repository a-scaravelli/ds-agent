"""Structural guards of the analytics ticket: holdout, frozen packages, single writer, scope (cases 1, 4, 6, 20, 42, 45)."""

import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest
from analytics_helpers import (  # noqa: F401  (fixtures)
    attempts_of,
    client_of,
    cycle_of,
    index_lines,
    proba_world,
)
from test_orchestrator_guards import PINNED

from kaggle_agent.evaluation import EvaluationEngine, Protocol
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.sandbox.protected import hash_path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"
WEB = SRC / "web"
ANALYTICS_FILES = (sorted((SRC / "analytics").glob("*.py")) + sorted(WEB.glob("analytics_*.py"))
                   + [SRC / "orchestrator" / "analytics_record.py", SRC / "orchestrator" / "importance.py"])
HOLDOUT_NAMES = {"HoldoutAccessor", "holdout_session", "evaluate_holdout", "_holdout_data", "HoldoutEvaluator",
                 "open_holdout_session", "close_holdout_session", "add_holdout_read", "list_holdout_reads",
                 "get_holdout_outcome", "add_holdout_outcome", "holdout", "HoldoutRead", "HoldoutOutcome"}
ALLOWED_SPLIT_ATTRIBUTES = {"data", "development_data", "partitions", "development_rows"}
WRITE_PREFIXES = ("add_", "set_", "close_", "open_", "record_", "update_", "delete_", "start_", "create_", "save_")


def names_in(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id, node
        elif isinstance(node, ast.Attribute):
            yield node.attr, node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                yield alias.name.split(".")[-1], node
            if isinstance(node, ast.ImportFrom) and node.module:
                yield node.module.split(".")[-1], node
        elif isinstance(node, ast.arg):
            yield node.arg, node


# 1
def test_no_analytics_code_imports_or_calls_a_holdout_name_or_evaluation_split():
    assert len(ANALYTICS_FILES) >= 20
    for path in ANALYTICS_FILES:
        for name, node in names_in(path):
            assert name not in HOLDOUT_NAMES, (path.name, name, node.lineno)
            assert name != "split" or not isinstance(node, (ast.Import, ast.ImportFrom)), (path.name, node.lineno)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.endswith("evaluation.split"), (path.name, node.lineno)
                assert not node.module.endswith("registry.sqlite") or path.name == "importance.py", path.name
            if isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)):
                base = node.value.id if isinstance(node.value, ast.Name) else node.value.attr
                if base in ("split", "_split"):  # the only data access: FrozenSplit.data and friends
                    assert node.attr in ALLOWED_SPLIT_ATTRIBUTES, (path.name, node.attr, node.lineno)
    # string scan: no holdout row, marker or accessor named in code strings (docstrings aside)
    for path in ANALYTICS_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert "holdout" not in node.value.lower(), (path.name, node.lineno)


# 4
def test_frozen_packages_keep_their_pins_and_the_engine_verifies_after_a_recorded_run(proba_world):
    for name in ("evaluation", "loop", "registry", "pipeline", "llm", "kaggle", "bundle", "writer"):
        assert hash_path(SRC / name) == PINNED[name], name
    protocol = Protocol.from_dict(json.loads((proba_world.project / "project.json").read_text(encoding="utf-8"))["protocol"])
    registry = SqliteRegistry(str(proba_world.project / "registry.db"), read_only=True)
    try:
        EvaluationEngine(registry, cycle_of(proba_world), protocol).verify()
    finally:
        registry.close()
    assert index_lines(proba_world)  # recorded, and still verified


# 6
def test_the_registry_is_not_written_by_pages_apis_or_by_the_analytics_code(proba_world):
    db = proba_world.project / "registry.db"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    sizes = {p.name: p.stat().st_size for p in db.parent.glob("registry.db-*")}
    client = client_of(proba_world)
    attempt, run = attempts_of(proba_world)[-1], cycle_of(proba_world)
    urls = [f"/api/attempts/{attempt}/{p}" for p in ("analytics", "folds", "panels", "segments", "hardest", "flags", "features")]
    urls += [f"/api/runs/{run}/{p}" for p in ("history", "compare", "ablation")]
    urls += [f"/attempts/{attempt}/analytics", f"/runs/{run}/compare", f"/runs/{run}/paired?a={attempts_of(proba_world)[0]}&b={attempt}"]
    for url in urls:
        assert client.get(url).status_code == 200, url
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert {p.name: p.stat().st_size for p in db.parent.glob("registry.db-*")} == sizes
    for path in ANALYTICS_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = ast.unparse(node.func.value)
                if "registry" in owner.lower():
                    assert not node.func.attr.startswith(WRITE_PREFIXES), (path.name, node.func.attr, node.lineno)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "SqliteRegistry":
                assert path.name == "importance.py" and any(
                    k.arg == "read_only" and getattr(k.value, "value", None) is True for k in node.keywords), (path.name, node.lineno)
    assert not [p for p in ANALYTICS_FILES if p.parent == WEB and "SqliteRegistry" in p.read_text(encoding="utf-8")]


# 42
def test_no_new_dependency_no_new_vendored_file_and_no_secret_or_code_in_the_answers(proba_world):
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(d.split("==")[0] for d in pyproject["project"]["dependencies"])
    assert names == ["catboost", "flask", "kaggle", "lightgbm", "numpy", "pandas", "scikit-learn", "six", "xgboost"]
    static = WEB / "static"
    assert sorted(p.name for p in (static / "vendor").iterdir()) == ["htmx.min.js", "marked.min.js", "uplot.iife.min.js", "uplot.min.css"]
    record = (static / "VENDORED.md").read_text(encoding="utf-8")
    for p in (static / "vendor").iterdir():
        assert hashlib.sha256(p.read_bytes()).hexdigest() in record, p.name
    client = client_of(proba_world)
    attempt, run = attempts_of(proba_world)[-1], cycle_of(proba_world)
    key_like = re.compile(r"sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{8,}|BEGIN [A-Z ]*PRIVATE KEY|[0-9a-f]{40,}")
    urls = [f"/api/attempts/{attempt}/{p}" for p in ("analytics", "folds", "panels", "segments", "hardest", "flags", "features")]
    urls += [f"/api/runs/{run}/side-by-side?ids={attempts_of(proba_world)[0]},{attempt}", f"/api/runs/{run}/history"]
    for url in urls:
        text = client.get(url).get_data(as_text=True)
        assert not key_like.search(text.replace('"sha256"', "")) or "sha256" in text, url
        for forbidden in ("def fit", "def predict", "import os", str(proba_world.base), "BUNDLE_MODEL_DIR", "workspace/"):
            assert forbidden not in text, (url, forbidden)
    for line in index_lines(proba_world):
        assert not key_like.search(json.dumps({k: v for k, v in line.items() if k != "sha256"})), line


# 45
def test_no_ticket_4_item_is_built_and_the_files_keep_a_schema_version(proba_world):
    client = client_of(proba_world)
    banned = re.compile(r"unseal|submit|submission|export|reproduce|memory|override|learning[ -]curve|code[ -]view|diff\b", re.I)
    rules = [r.rule for r in client.application.url_map.iter_rules()
             if re.search(r"analytics|folds|panels|segments|hardest|flags|features|paired|history|compare|side-by-side|ablation|importance", r.rule)]
    assert rules and not [r for r in rules if banned.search(r)]
    for name in ("analytics_attempt.html", "analytics_compare.html", "analytics_paired.html"):
        text = (WEB / "templates" / name).read_text(encoding="utf-8")
        # a form button is type="submit": HTML, not the ticket 4 submission
        assert not banned.search(re.sub(r"\{[{%#].*?[}%#]\}", "", text).replace('type="submit"', "")), name
        assert not re.search(r"<button[^>]*>[^<]*(ablation|holdout|submit)", text, re.I), name
    js = (WEB / "static" / "js" / "analytics.js").read_text(encoding="utf-8")
    assert not banned.search(js) and "holdout" not in js.lower()
    controller = (WEB / "controller.py").read_text(encoding="utf-8")
    # ticket 30 extended the closed list deliberately (bundle jobs and the submission jobs); importance is unchanged
    assert 'JOB_COMMANDS = ("validate-config", "profile", "preview", "importance", *BUNDLE_JOBS, *SUBMIT_JOBS)' in controller
    assert all(line["v"] == 1 for line in index_lines(proba_world))
    stored = [p for p in (proba_world.project / "analytics" / "describe").glob("*.json")]
    assert stored and json.loads(stored[0].read_text(encoding="utf-8"))["v"] == 1
