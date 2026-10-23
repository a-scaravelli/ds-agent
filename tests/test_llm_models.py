"""ToolPermissions, Usage and consumption totals (cases 7, 9)."""

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from kaggle_agent.llm import ToolPermissions
from kaggle_agent.llm.models import CallRecord, Usage, summarize

LLM_DIR = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "llm"


def test_default_permissions_allow_no_tool(tmp_path):
    assert ToolPermissions(working_dir=tmp_path).allowed_tools == ()


def test_invalid_permissions_are_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError):
        ToolPermissions(working_dir=tmp_path, allowed_tools=("Read", ""))
    with pytest.raises(ValueError):
        ToolPermissions(working_dir=tmp_path, allowed_tools=("  ",))
    with pytest.raises(ValueError):
        ToolPermissions(working_dir=tmp_path, allowed_tools="Read")
    with pytest.raises(ValueError):
        ToolPermissions(working_dir="relative/dir")
    with pytest.raises(ValueError):
        ToolPermissions(working_dir=tmp_path / "missing")


def test_permissions_are_immutable_and_tools_become_a_tuple(tmp_path):
    perms = ToolPermissions(working_dir=tmp_path, allowed_tools=["Read"])
    assert perms.allowed_tools == ("Read",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        perms.allowed_tools = ("Bash",)


def test_summarize_adds_known_tokens_only():
    def call(outcome, usage):
        return CallRecord("p", outcome, "t", 2.0, usage, 1)

    total = summarize([call("ok", Usage(10, 5)), call("rate_limited", Usage())])
    assert (total.calls, total.input_tokens, total.output_tokens) == (2, 10, 5)
    assert total.duration_seconds == 4.0 and total.rate_limit_events == 1
    assert summarize([call("ok", Usage())]).input_tokens is None


def test_no_model_name_is_written_in_the_llm_code():
    pattern = re.compile(r"claude-|opus|sonnet|haiku|fable", re.IGNORECASE)
    for path in sorted(LLM_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ast.get_docstring(tree) == node.value:
                    continue
                if len(node.value) < 200:
                    assert not pattern.search(node.value), f"{path.name}: {node.value}"


def test_model_defaults_to_none_in_client_and_ask():
    import inspect

    from kaggle_agent.llm import LlmClient
    from kaggle_agent.llm.real import RealClaudeClient

    assert inspect.signature(LlmClient.__init__).parameters["model"].default is None
    assert inspect.signature(RealClaudeClient.__init__).parameters["model"].default is None
    assert inspect.signature(LlmClient.ask).parameters["model"].default is None
