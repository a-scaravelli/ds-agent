"""Every Claude call of the writer is recorded like the role calls."""

import functools
import hashlib
import re

import pytest
from writer_helpers import CYCLE, Env, bundle_files, new_registry, valid_step, writes

from kaggle_agent.llm import CliFailureError, QuotaExhaustedError, RateLimitError
from kaggle_agent.llm.recording import record_call
from kaggle_agent.writer import ClaudeBundleWriter, InvalidBundleError, WriterConfigurationError
from writer_helpers import ScriptedClaude, task


def wrong():
    return writes(bundle_files("B1", manifest_id="OTHER"))


def test_each_call_is_a_row_with_tokens_model_outcome_and_the_prompt_hash(tmp_path):
    env = Env(tmp_path, [wrong(), valid_step("B1")])
    env.writer.write(env.request("B1"))
    rows = env.rows()
    assert len(rows) == 2
    hashes = []
    for turn, row in enumerate(rows, 1):
        assert (row.input_tokens, row.output_tokens, row.model, row.outcome) == (10, 5, "fake-model", "ok")
        match = re.fullmatch(rf"bundle writer: B1 turn {turn} prompt ([0-9a-f]{{12}})", row.purpose)
        assert match, row.purpose
        hashes.append(match.group(1))
        assert row.role is None
    assert hashes[0] != hashes[1]
    calls = env.writer.calls
    assert [c.turn for c in calls] == [1, 2]
    for call, short, prompt in zip(calls, hashes, env.client.prompts):
        assert call.prompt_hash.startswith(short) and len(call.prompt_hash) == 64
        # the recorded prompt is the writer prompt without the JSON contract added by the client
        assert prompt.startswith(prompt[: prompt.index("\n\nReply with a single JSON object")])
        base = prompt[: prompt.index("\n\nReply with a single JSON object")]
        assert hashlib.sha256(base.encode("utf-8")).hexdigest() == call.prompt_hash


@pytest.mark.parametrize("error, outcome", [
    (RateLimitError("limited"), "rate_limited"),
    (QuotaExhaustedError("quota"), "quota"),
    (CliFailureError("crash"), "cli_error"),
])
def test_calls_of_a_failed_writer_are_still_recorded(tmp_path, error, outcome):
    env = Env(tmp_path, [error])
    with pytest.raises(type(error)):
        env.writer.write(env.request("B1"))
    rows = env.rows()
    assert [r.outcome for r in rows] == [outcome]


def test_the_rows_of_a_writer_that_never_repairs_are_recorded_once(tmp_path):
    env = Env(tmp_path, [wrong()] * 3, max_repairs=2)
    with pytest.raises(InvalidBundleError):
        env.writer.write(env.request("B1"))
    rows = env.rows()
    assert len(rows) == 3 and len({r.purpose for r in rows}) == 3
    assert len(env.writer.calls) == 3


def test_a_client_that_already_records_to_the_registry_is_refused(tmp_path):
    registry = new_registry()
    client = ScriptedClaude([], on_call=functools.partial(record_call, registry, CYCLE))
    with pytest.raises(WriterConfigurationError):
        ClaudeBundleWriter(client, registry, CYCLE, tmp_path.resolve() / "ws", task())
    other = new_registry()  # a hook on another registry is not a double recording
    ClaudeBundleWriter(client, other, CYCLE, tmp_path.resolve() / "ws", task())


def test_the_writer_only_adds_llm_call_rows_and_reads_bundles(tmp_path):
    registry = new_registry()
    touched = []

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attribute = getattr(self._inner, name)
            if callable(attribute):
                def wrapper(*args, **kwargs):
                    touched.append(name)
                    return attribute(*args, **kwargs)
                return wrapper
            return attribute

    env = Env(tmp_path, [valid_step("B1")], registry=Spy(registry))
    env.writer.write(env.request("B1"))
    assert set(touched) == {"add_llm_call"}
    for name in touched:
        assert "holdout" not in name and "session" not in name and "engine" not in name
