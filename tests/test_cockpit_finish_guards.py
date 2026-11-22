"""Structural guards of ticket 30: frozen packages, single writer, dependencies, design, controller (50, 51, 53, 56, 57)."""

import ast
import re
import sqlite3
import subprocess
import tomllib
from pathlib import Path

import pytest
from cockpit_finish_helpers import FinishWorld, iterating_world
from test_cockpit_design import colour_violations
from test_orchestrator_guards import PINNED

from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.sandbox.protected import hash_path
from kaggle_agent.web import controller as controller_module
from kaggle_agent.web.controller import RunController
from kaggle_agent.web.writes import WRITE_ENDPOINTS

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"
WEB = SRC / "web"
FROZEN = ("evaluation", "loop", "registry", "pipeline", "llm", "kaggle", "sandbox", "bundle", "writer", "analytics")
# ticket 30 pins the two packages the older guard did not list: they are byte-identical to origin/main
PINNED_TOO = {"sandbox": "497755bec46127ccbe0fdfd204fb4032213fb318d0b79ad1542ac8f3d69fcda1",
              "analytics": "10872be60fcbc9007d799fdcf4f588ad29644019d0094552e262542c43560c80"}
SCHEMA = {
    "attempts": ["id", "cycle_id", "iteration", "fingerprint", "closed", "conclusion", "proposal", "round_id"],
    "bundles": ["id", "cycle_id", "folder_hash", "reference", "schema_json"],
    "comparisons": ["cycle_id", "candidate_attempt_id", "reference_attempt_id", "difference", "threshold", "is_real"],
    "cycles": ["id", "plan", "iterations", "stagnation_k", "max_fix_retries", "state", "protocol_hash", "project_id",
               "parameters"],
    "executions": ["id", "attempt_id", "sequence", "outcome", "failure_kind", "record"],
    "final_reports": ["cycle_id", "record"], "holdout_outcomes": ["cycle_id", "record"],
    "holdout_reads": ["cycle_id", "attempt_id", "rows", "read_at"],
    "holdout_sessions": ["cycle_id", "opened_at", "closed_at"],
    "lead_guidance": ["id", "cycle_id", "round_id", "text", "created_at"],
    "llm_calls": ["cycle_id", "purpose", "outcome", "started_at", "duration_seconds", "input_tokens", "output_tokens",
                  "attempt", "model", "rejection_reason", "role"],
    "reviewer_reports": ["id", "round_id", "text", "created_at"],
    "role_traces": ["cycle_id", "role", "round_id", "candidate_id", "input_type", "input", "output_type", "output",
                    "outcome", "llm_call_id", "created_at"],
    "rounds": ["id", "cycle_id", "position", "tasks"],
    "submissions": ["cycle_id", "competition", "submission_id", "file_sha256", "message", "submitted_at", "status",
                    "public_score", "completed_at"],
    "suggestion_cycles": ["suggestion_id", "cycle_id", "position"],
    "suggestions": ["id", "text", "origin_cycle"],
}


# 50
def test_the_frozen_packages_are_byte_identical_and_the_registry_schema_is_unchanged(tmp_path):
    for name, expected in {**PINNED, **PINNED_TOO}.items():
        assert hash_path(SRC / name) == expected, name
    assert set(PINNED) | set(PINNED_TOO) == set(FROZEN)
    SqliteRegistry(str(tmp_path / "r.db")).close()
    db = sqlite3.connect(tmp_path / "r.db")
    found = {t: [r[1] for r in db.execute(f"pragma table_info({t})")]
             for (t,) in db.execute("select name from sqlite_master where type = 'table' and name not like 'sqlite_%'")}
    db.close()
    assert found == SCHEMA
    git = ROOT / ".git"
    if git.exists():  # in a checkout: the same statement, from the repository itself
        try:
            out = subprocess.run(["git", "diff", "--stat", "origin/main", "--", *[f"src/kaggle_agent/{n}" for n in FROZEN]],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return
        if out.returncode == 0:
            assert out.stdout.strip() == ""


# 51
def test_the_registry_is_not_written_by_any_web_call_and_the_web_has_no_writable_registry():
    for path in WEB.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "SqliteRegistry":
                keywords = {k.arg: k.value for k in node.keywords}
                assert isinstance(keywords.get("read_only"), ast.Constant) and keywords["read_only"].value is True, \
                    f"{path.name}:{node.lineno}"
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "connect" and \
                    getattr(getattr(node.func, "value", None), "id", "") == "sqlite3":
                assert path.name == "__main__.py", path.name  # the read-only URI check of the server start


def test_a_full_flow_of_gets_and_posts_leaves_the_registry_file_unchanged(tmp_path):
    fw = iterating_world(tmp_path)
    before = fw.registry_hash()
    wal = fw.project / "registry.db-wal"
    wal_before = wal.read_bytes() if wal.exists() else None
    for rule in sorted(WRITE_ENDPOINTS):
        if rule in ("/api/run/start", "/api/run/resume"):
            continue
        fw.post(rule, {"attempt": "toy-1-R1-c1", "confirm": True, "kind": "first", "reason": "x", "from_run": "toy-1"})
    for url in ("/", "/history", "/learnings", "/fork", "/cycles/toy-1/report", "/cycles/toy-1/bundle",
                "/cycles/toy-1/submission", "/cycles/toy-1/gap", "/api/history", "/api/overrides/pending",
                "/api/runs/toy-1/holdout", "/api/runs/toy-1/bar-override"):
        fw.client.get(url)
    # the orchestrator is the only writer: the commands the web started are the ones that may have changed it
    started = [a[1] for a in fw.controller.argvs]
    if started:
        assert set(started) <= {"reproduce", "predict", "finalize", "rerun", "submit", "fork", "resume", "start", "preview",
                                  "profile", "importance", "validate-config"}
    else:
        assert fw.registry_hash() == before and (wal.read_bytes() if wal.exists() else None) == wal_before


# 53
def test_no_new_dependency_and_the_new_modules_import_only_the_standard_library_and_pinned_packages():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = sorted(d.split("==")[0] for d in project["dependencies"])
    assert names == ["catboost", "flask", "kaggle", "lightgbm", "numpy", "pandas", "scikit-learn", "six", "xgboost"]
    import sys

    stdlib = sys.stdlib_module_names
    for path in (SRC / "overrides").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
            for module in modules:
                top = module.split(".")[0]
                assert top in stdlib or module.startswith("kaggle_agent.proposals") or top == "kaggle_agent", (path.name, module)
                if top == "kaggle_agent":
                    assert module.split(".")[1] in ("proposals", "overrides"), (path.name, module)


# 56
def test_the_new_templates_and_stylesheets_use_tokens_only_and_the_dialogs_are_rounded():
    web_static = WEB / "static"
    new = ("report_print", "holdout_panel", "bundle", "predict", "submission", "gap", "history", "learnings", "fork",
           "override_panels")
    files = [*[WEB / "templates" / f"{n}.html" for n in new], web_static / "css" / "app.css",
             web_static / "css" / "print.css", web_static / "js" / "overrides.js"]
    for path in files:
        assert colour_violations(path) == [], path.name
    print_css = (web_static / "css" / "print.css").read_text(encoding="utf-8")
    assert "@media print" in print_css and not re.search(r"border-radius:\s*[0-9]", print_css)
    for template in ("fork.html", "submission.html", "override_panels.html", "holdout_panel.html"):
        text = (WEB / "templates" / template).read_text(encoding="utf-8")
        assert "style=" not in text and "<style" not in text, template  # the CSP allows no inline style
        assert not re.search(r"https?://", text), template
    app_css = (web_static / "css" / "app.css").read_text(encoding="utf-8")
    assert re.search(r"dialog[^{]*\{[^}]*border-radius: var\(--radius", app_css)
    for body in re.findall(r"border-radius:\s*([^;}]+)", app_css):
        assert re.fullmatch(r"var\(--radius[\w-]*\)|0|50%", body.strip()), body


# 57
EVIL = ["; rm -rf /", "../../etc/passwd", "a b", "x\nrm", "$(id)", "`id`", "a" * 200, ""]


@pytest.mark.parametrize("bad", EVIL)
def test_a_bad_id_or_hash_is_refused_before_anything_is_spawned(tmp_path, bad):
    fw = FinishWorld(tmp_path)
    spawned = []
    fw.controller._spawn = lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError("spawned"))
    for kind, params in (("rerun", {"attempt": bad, "seed_offset": 1, "decision": "a" * 32}), ("submit-confirm", {"sha256": bad}),
                         ("submit-confirm", {"sha256": "0" * 64, "adopt_id": bad or "x/y"}),
                         ("predict", {"upload": bad}), ("reproduce", {"bundle": bad})):
        attempt = params.pop("attempt", None)
        with pytest.raises((ValueError, TypeError)):
            fw.controller.start_job(kind, attempt=attempt, **params)
    with pytest.raises((ValueError, TypeError)):
        fw.controller.launch("fork", fork={"from_run": "toy-1", "from_attempt": bad, "request_id": "request-0001"})
    assert spawned == []


def test_the_controller_keeps_a_closed_list_and_unknown_commands_never_spawn(tmp_path):
    fw = FinishWorld(tmp_path)
    assert controller_module.JOB_COMMANDS == ("validate-config", "profile", "preview", "importance", "reproduce",
                                              "predict", "finalize", "rerun", "submit-preview", "submit-confirm")
    spawned = []
    fw.controller._spawn = lambda *a, **k: spawned.append(a)
    for command in ("shell", "rm", "submit", "fork --x", "../x"):
        with pytest.raises(ValueError):
            fw.controller.start_job(command)
        with pytest.raises(ValueError):
            fw.controller.launch(command)
    assert spawned == [] and issubclass(RunController, object)
    source = (WEB / "controller.py").read_text(encoding="utf-8")
    assert "shell=True" not in source and "os.system" not in source and "os.environ" not in source
