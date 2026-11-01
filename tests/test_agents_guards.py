"""Structural guards of the agents subpackage (cases 16-19, 22, 23)."""

import ast
import dataclasses
import sys
import types
import typing
from pathlib import Path

import pytest

from kaggle_agent.agents import (
    CoordinatorInput,
    DataInput,
    FeatureEngineeringInput,
    ModelingInput,
    ReviewerInput,
    ReviewerNotes,
)
from kaggle_agent.agents import summaries
from kaggle_agent.proposals import ReviewerReport

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"
AGENTS = SRC / "agents"
ROLE_MODULES = ("coordinator", "data", "feature_engineering", "modeling", "reviewer")
ROLE_RUNNERS = {f"run_{name}" for name in ROLE_MODULES}
SHARED_MODULES = {"call", "prompting", "summaries", "errors"}
INPUTS = (CoordinatorInput, DataInput, FeatureEngineeringInput, ModelingInput, ReviewerInput)
DECISION_WORDS = (
    "accept", "reject", "approve", "decision", "verdict", "gate", "pass", "winner", "best",
    "score",
)
FORBIDDEN_FIELD_PARTS = ("label", "holdout", "credential", "path", "dataframe")
FORBIDDEN_TYPE_NAMES = {"Registry", "DataFrame", "Credentials", "Path", "PurePath"}
NETWORK = {"socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp", "asyncio", "subprocess"}
FORBIDDEN_PACKAGES = {"evaluation", "kaggle", "sandbox", "web", "loop", "review", "budget"}


def _imports(path):
    """(top-level module, dotted module, imported names, relative level) per import."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], a.name, (), 0
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            yield module.split(".")[0], module, tuple(a.name for a in node.names), node.level


def _agents_files():
    return sorted(AGENTS.glob("*.py"))


def _dataclass_types(annotation, seen):
    """Every dataclass type mentioned in an annotation, recursively."""
    if dataclasses.is_dataclass(annotation):
        yield annotation
    for arg in typing.get_args(annotation):
        yield from _dataclass_types(arg, seen)


def _all_types(root):
    """The dataclass ``root`` and every dataclass reachable through its field annotations."""
    seen, stack = [], [root]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.append(cls)
        for annotation in typing.get_type_hints(cls).values():
            stack.extend(_dataclass_types(annotation, seen))
    return seen


def _names_in(annotation):
    if isinstance(annotation, type):
        yield annotation.__name__
    for arg in typing.get_args(annotation):
        yield from _names_in(arg)


def test_reviewer_output_has_no_field_that_decides():
    for cls in (ReviewerNotes, ReviewerReport):
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            assert hints[field.name] is str, (cls.__name__, field.name)
            assert not any(w in field.name.lower() for w in DECISION_WORDS), field.name
    assert [f.name for f in dataclasses.fields(ReviewerNotes)] == ["text"]
    for field in dataclasses.fields(ReviewerInput):
        assert not any(w in field.name.lower() for w in DECISION_WORDS), field.name
    from kaggle_agent.agents import build_reviewer_prompt
    from agents_helpers import reviewer_input

    assert "does not decide" in build_reviewer_prompt(reviewer_input())


def test_roles_do_not_import_or_call_each_other():
    allowed_relative = SHARED_MODULES
    for name in ROLE_MODULES:
        path = AGENTS / f"{name}.py"
        for top, module, names, level in _imports(path):
            if level:
                target = module.split(".")[0] if module else None
                assert target in allowed_relative, f"{path.name}: {module}"
            elif top == "kaggle_agent":
                parts = module.split(".")
                assert parts[1] in {"llm", "proposals"}, f"{path.name}: {module}"
                assert len(parts) == 2 or parts[1] == "proposals", f"{path.name}: {module}"
        own = f"run_{name}"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                assert called not in ROLE_RUNNERS - {own}, f"{path.name}: {called}"


def test_modeling_receives_the_previous_attempt_only_as_an_argument(tmp_path):
    from kaggle_agent.agents import run_modeling
    from kaggle_agent.llm import FakeLlmClient, Reply
    from kaggle_agent.llm.fake import valid_attempt_json
    from agents_helpers import attempt, context, fresh_registry, modeling_input, permissions

    previous = attempt("FE-5", what_changed="unique-change-text")
    client = FakeLlmClient([Reply(valid_attempt_json())])
    run_modeling(
        client, fresh_registry(), context(), modeling_input(previous), permissions(tmp_path)
    )
    assert "unique-change-text" in client.calls[0].prompt


@pytest.mark.parametrize("root", INPUTS + (summaries.DataSummary, summaries.RegistrySummary))
def test_inputs_are_typed_summaries_without_sensitive_fields(root):
    for cls in _all_types(root):
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            lowered = field.name.lower()
            assert not any(p in lowered for p in FORBIDDEN_FIELD_PARTS), (cls.__name__, field.name)
            names = set(_names_in(hints[field.name]))
            assert not names & FORBIDDEN_TYPE_NAMES, (cls.__name__, field.name)
            assert not any(
                p in n.lower() for n in names for p in ("label", "holdout", "credential")
            ), (cls.__name__, field.name)


def test_agents_imports_only_allowed_modules():
    stdlib = set(sys.stdlib_module_names)
    for path in _agents_files():
        for top, module, _, level in _imports(path):
            if level:
                assert module.split(".")[0] in SHARED_MODULES | set(ROLE_MODULES) | {""}, module
                continue
            assert top in stdlib | {"kaggle_agent"}, f"{path.name}: {module}"
            assert top not in NETWORK, f"{path.name}: {module}"
            if top == "kaggle_agent":
                parts = module.split(".")
                assert len(parts) > 1 and parts[1] in {"llm", "proposals", "registry"}, module
                assert parts[1] not in FORBIDDEN_PACKAGES, module


def test_only_call_imports_from_registry():
    for path in _agents_files():
        for top, module, names, level in _imports(path):
            from_registry = (top == "kaggle_agent" and module.split(".")[1:2] == ["registry"]) or (
                level and module.split(".")[0] == "registry"
            )
            assert not from_registry or path.name == "call.py", f"{path.name}: {module}"


def test_existing_packages_do_not_import_agents():
    for package in ("proposals", "registry", "evaluation", "kaggle", "llm", "sandbox", "web"):
        for path in sorted((SRC / package).rglob("*.py")):
            for top, module, names, level in _imports(path):
                parts = module.split(".")
                assert "agents" not in parts, f"{path}: {module}"
                assert "agents" not in names or level == 0 and top != "kaggle_agent", path


def test_nothing_chains_roles_and_no_other_step_is_present():
    for package in ("loop", "review", "budget"):
        assert [p.name for p in (SRC / package).rglob("*.py")] == ["__init__.py"], package
    stems = {p.stem for p in SRC.rglob("*.py")}
    assert not stems & {"runner", "bundle", "pipeline"}
    for path in _agents_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("run_"):
                called = {
                    n.func.id for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                } & ROLE_RUNNERS
                assert not called, f"{path.name}: {node.name}"


def test_no_agents_test_builds_a_real_client():
    for path in sorted(Path(__file__).parent.glob("*agents*.py")):
        if path != Path(__file__).resolve():
            assert "RealClaudeClient" not in path.read_text(encoding="utf-8"), path.name


def test_readme_has_the_agents_section_and_its_example_runs():
    import re

    readme = (SRC.parent.parent / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Role agents (`kaggle_agent.agents`)")
    section = readme[start : readme.index("\n## ", start + 1)]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 1
    exec(compile(blocks[0], "README agents example", "exec"), {})
    for word in ("RoleConfigurationError", "record_trace", "Known limits"):
        assert word in section
