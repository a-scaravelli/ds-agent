"""The real writer in the unchanged loop: Claude errors suspend, invalid bundles follow the old rules."""

import re

import pytest
from loop_helpers import CYCLE, crash, make_setup
from registry_helpers import make_cycle
from writer_helpers import ScriptedClaude, bundle_files, make_request, put, task

from kaggle_agent.llm import CliFailureError, QuotaExhaustedError, RateLimitError
from kaggle_agent.loop import is_placeholder
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.writer import ClaudeBundleWriter


def build(tmp_path, steps, **setup_args):
    registry = SqliteRegistry()
    claude = ScriptedClaude(steps)
    writer = ClaudeBundleWriter(
        claude, registry, CYCLE, tmp_path.resolve() / "writer-ws", task(), max_repairs=1, protected=(),
    )
    setup = make_setup(tmp_path, registry=registry, writer=writer, **setup_args)
    return setup, claude, writer


def adaptive(claude, fix_is_valid, always_wrong=False):
    """A step that writes a valid bundle with the id found in the prompt, or a broken one."""
    def step(working_dir):
        prompt = claude.prompts[-1]
        bundle_id = re.search(r"Bundle id: (\S+)", prompt).group(1)
        good = not always_wrong and (fix_is_valid or "## Fix request" not in prompt)
        put(working_dir, bundle_files(bundle_id, manifest_id=None if good else "WRONG"))
        return "{}"
    return step


@pytest.mark.parametrize("error", [
    QuotaExhaustedError("quota exhausted (simulated)"),
    RateLimitError("rate limited (simulated)"),
    CliFailureError("cli failed (simulated)"),
])
def test_claude_errors_reach_the_caller_unchanged(tmp_path, error):
    claude = ScriptedClaude([error])
    registry = SqliteRegistry()
    registry.add_cycle(make_cycle(CYCLE))
    writer = ClaudeBundleWriter(claude, registry, CYCLE, tmp_path.resolve() / "ws", task(), protected=())
    with pytest.raises(type(error)):
        writer.write(make_request(tmp_path.resolve() / "bundles" / "B1"))


@pytest.mark.parametrize("error, word", [
    (QuotaExhaustedError("quota exhausted (simulated)"), "quota"),
    (RateLimitError("rate limited (simulated)"), "rate limited"),
    (CliFailureError("cli failed (simulated)"), "cli failed"),
])
def test_claude_errors_suspend_the_cycle_and_keep_the_registry_resumable(tmp_path, error, word):
    claude_steps = [error]
    setup, claude, writer = build(tmp_path, claude_steps, rounds=1, candidates=1, stagnation_k=5)
    outcome = setup.run()
    assert outcome.status == "suspended" and word in outcome.reason
    assert setup.cycle().state == outcome.state
    assert setup.registry.list_attempts(CYCLE) == []
    # the cycle resumes from the saved state once Claude works again
    claude.steps = [error, *([adaptive(claude, True)] * 200)]
    resumed = setup.run()
    assert resumed.status == "stopped"


def test_a_persistently_invalid_writer_gives_the_existing_placeholder_attempts(tmp_path):
    setup, claude, writer = build(tmp_path, [], rounds=1, candidates=1, stagnation_k=5, max_rejections=2)
    claude.steps = [adaptive(claude, False, always_wrong=True)] * 200
    setup.run()
    placeholder = setup.registry.get_attempt("C1-R1-c1")
    assert is_placeholder(placeholder) and placeholder.outcome == "inconclusive"
    assert placeholder.executions[0].reason.startswith("no valid proposal after 2 rejections")
    assert setup.registry.list_bundles(CYCLE) == []
    assert len(claude.prompts) % 2 == 0  # two Claude calls (1 repair) per rejected proposal


def test_an_invalid_fix_bundle_ends_the_attempt_as_failed_and_not_retryable(tmp_path):
    def rule(bundle_id, n):
        return crash("KeyError: age") if bundle_id == "C1-R1-c1-b1" else 0.3

    setup, claude, writer = build(
        tmp_path, [], rounds=1, candidates=1, stagnation_k=5, max_rejections=2, rule=rule, fix_retries=3,
    )
    claude.steps = [adaptive(claude, False)] * 200
    setup.run()
    detail = setup.registry.get_attempt("C1-R1-c1")
    last = detail.executions[-1]
    assert last.outcome == "failed" and last.failure_kind == "code_error"
    assert last.resources == {"sandbox_status": "invalid_fix_bundle", "retryable": False}
    assert last.reason.startswith("no valid fix bundle")
    fix_prompts = [p for p in claude.prompts if "## Fix request" in p]
    assert fix_prompts and "KeyError: age" in fix_prompts[0]
    assert detail.closed
