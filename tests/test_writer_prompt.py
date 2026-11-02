"""What the prompt carries, the tools Claude gets and what it must never receive."""

import inspect

import pandas as pd
from writer_helpers import Env, valid_step

from kaggle_agent.agents import ColumnSummary, DataSummary
from kaggle_agent.writer import (
    DEFAULT_DEPENDENCIES, WRITER_TOOLS, ClaudeBundleWriter, WriterTask, build_prompt,
)
from writer_helpers import make_request, task


def test_the_prompt_carries_role_outputs_findings_task_and_contract(tmp_path):
    env = Env(tmp_path, [valid_step("B7")])
    env.writer.write(env.request("B7"))
    prompt = env.client.prompts[0]
    for text in (
        "hypothesis of fe", "hypothesis of md", "what changed in fe", "what changed in md",
        "findings profile text", "some quality issue",
        "Target column: y", "Metric: accuracy", "Prediction kind: labels",
        "fit(train)", "predict(bundle_id, X)", "BUNDLE_MODEL_DIR", "B7",
        "numpy", "pandas", "scikit-learn", "standard library",
        "probability matrix", "1-D array", "JSON only", "bundle.json",
        "No tool runs code",
    ):
        assert text in prompt, text


def test_the_prompt_asks_for_the_requested_prediction_kind():
    request = make_request("unused", "B1")
    text = build_prompt(request, task(input_kind="probabilities"), DEFAULT_DEPENDENCIES)
    assert "every row summing to 1" in text


def test_tools_are_minimal_with_no_network_and_confined_to_the_staging_directory(tmp_path):
    from writer_helpers import bundle_files, replaces, writes

    env = Env(tmp_path, [writes(bundle_files("B1", manifest_id="X")), replaces(bundle_files("B1"))])
    request = env.request("B1")
    env.writer.write(request)
    assert len(env.client.permissions) == 2
    for permissions in env.client.permissions:
        assert permissions.allowed_tools == WRITER_TOOLS == ("Read", "Write", "Edit", "Glob")
        for forbidden in ("Bash", "WebFetch", "WebSearch"):
            assert forbidden not in permissions.allowed_tools
        working = permissions.working_dir
        assert working.is_absolute() and working == working.resolve()
        assert working.parent == env.root.resolve()
        assert working not in (request.folder, request.folder.parent)


def test_only_the_schema_of_the_development_data_reaches_the_writer(tmp_path):
    sentinel_label, sentinel_cell = 987654321, 123456789.5
    frame = pd.DataFrame({"x1": [sentinel_cell, 1.0], "y": [sentinel_label, 0]})
    summary = DataSummary(
        len(frame), "y",
        tuple(ColumnSummary(c, str(frame[c].dtype), 0.0, int(frame[c].nunique())) for c in frame.columns),
    )
    seen = []

    def inspecting(working_dir):
        seen.append(sorted(p.name for p in working_dir.iterdir()))
        seen.append(sorted(p.name for p in (working_dir / "bundle").iterdir()))
        return valid_step("B1")(working_dir)

    env = Env(tmp_path, [inspecting], task=WriterTask("y", "accuracy", "maximize", "labels", summary))
    env.writer.write(env.request("B1"))
    prompt = env.client.prompts[0]
    assert str(sentinel_label) not in prompt and "123456789" not in prompt
    assert "x1: float64" in prompt and "y: int64" in prompt
    assert seen == [["bundle"], []]  # the staging directory holds no data file


def test_neither_the_constructor_nor_write_accepts_data_an_engine_or_a_holdout():
    names = set(inspect.signature(ClaudeBundleWriter.__init__).parameters) | set(
        inspect.signature(ClaudeBundleWriter.write).parameters)
    for forbidden in ("frame", "data", "engine", "holdout", "accessor", "labels", "dataframe"):
        assert not any(forbidden in n.lower() for n in names), forbidden
