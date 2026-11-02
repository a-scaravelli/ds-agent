"""A scripted Claude writes a valid bundle: the writer returns the verified folder."""

import json

import pytest
from writer_helpers import Env, bundle_files, valid_step, writes

from kaggle_agent.bundle import BundleError, load_bundle
from kaggle_agent.loop.writer import BundleWriter
from kaggle_agent.writer import ClaudeBundleWriter, WriterReport


def test_claude_bundle_writer_implements_bundle_writer_and_writes_the_request_folder(tmp_path):
    env = Env(tmp_path, [valid_step("B1")])
    request = env.request("B1")
    assert issubclass(ClaudeBundleWriter, BundleWriter)
    folder = env.writer.write(request)
    assert folder == request.folder
    loaded = load_bundle(folder)
    assert loaded.bundle_id == request.bundle_id
    assert json.loads((folder / "bundle.json").read_text())["bundle_id"] == "B1"
    assert len(env.client.prompts) == 1
    assert env.root.is_dir() and list(env.root.iterdir()) == []  # staging directory removed
    assert not env.client.permissions[0].working_dir.exists()


@pytest.mark.parametrize("reply", [
    "Here is the bundle:\n```python\ndef fit(train):\n    pass\n```\nDone.",
    "I wrote it { all good } and it should work.",
    "",
])
def test_a_non_json_reply_causes_neither_a_retry_nor_an_error(tmp_path, reply):
    env = Env(tmp_path, [valid_step("B1", reply=reply)])
    folder = env.writer.write(env.request("B1"))
    assert load_bundle(folder).bundle_id == "B1"
    assert len(env.client.prompts) == 1
    rows = env.rows()
    assert len(rows) == 1 and all(r.outcome == "ok" for r in rows)


def test_writer_report_accepts_any_text_and_keeps_a_json_summary():
    assert WriterReport.from_json("```python\nprint(1)\n```") == WriterReport()
    assert WriterReport.from_json("{ not json }") == WriterReport()
    assert WriterReport.from_json('{"summary": "wrote model.py"}').summary == "wrote model.py"
    assert WriterReport.from_json("[1, 2]") == WriterReport()
    assert WriterReport.from_json(None) == WriterReport()


def test_a_valid_bundle_in_a_nested_module_is_accepted(tmp_path):
    files = bundle_files("B1")
    files["bundle.json"] = json.dumps({
        "bundle_id": "B1", "dependencies": [],
        "entry_points": {"fit": "pkg.model:fit", "predict": "pkg.model:predict"},
    })
    files["pkg/model.py"] = files.pop("model.py")
    files["pkg/__init__.py"] = ""
    env = Env(tmp_path, [writes(files)])
    assert load_bundle(env.writer.write(env.request("B1"))).bundle_id == "B1"


def test_writer_errors_are_bundle_errors():
    from kaggle_agent.writer import (
        BundleWriterError, InvalidBundleError, PreviousBundleError, WorkspaceViolationError,
        WriterConfigurationError,
    )

    for kind in (BundleWriterError, InvalidBundleError, PreviousBundleError,
                 WorkspaceViolationError, WriterConfigurationError):
        assert issubclass(kind, BundleError)
