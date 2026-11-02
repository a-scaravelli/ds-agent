"""Verification before returning, bounded repair turns and the typed error."""

import pytest
from writer_helpers import (
    NO_PREDICT_SOURCE, UNDECLARED_SOURCE, Env, bundle_files, replaces, valid_step, writes,
)

from kaggle_agent.bundle import BundleError, BundleLayoutError, load_bundle
from kaggle_agent.writer import InvalidBundleError


def wrong_id_step(bundle_id="B1"):
    return writes(bundle_files(bundle_id, manifest_id="OTHER"))


def test_a_wrong_manifest_id_is_repaired_with_the_exact_check_error(tmp_path):
    env = Env(tmp_path, [wrong_id_step(), valid_step("B1")], max_repairs=2)
    folder = env.writer.write(env.request("B1"))
    assert len(env.client.prompts) == 2
    assert "the manifest has bundle_id 'OTHER', expected 'B1'" in env.client.prompts[1]
    assert "the manifest has bundle_id" not in env.client.prompts[0]
    assert load_bundle(folder).bundle_id == "B1"


@pytest.mark.parametrize("bad, message", [
    (bundle_files("B1", source=NO_PREDICT_SOURCE), "does not define predict()"),
    (bundle_files("B1", source=UNDECLARED_SOURCE), "imports 'scipy'"),
    (bundle_files("B1", dependencies=("scipy",), source=UNDECLARED_SOURCE), "dependency 'scipy'"),
])
def test_entry_points_imports_and_dependencies_are_checked_and_repaired(tmp_path, bad, message):
    env = Env(tmp_path, [writes(bad), valid_step("B1")])
    folder = env.writer.write(env.request("B1"))
    assert message in env.client.prompts[1]
    assert load_bundle(folder).bundle_id == "B1"


def test_a_model_folder_written_by_the_writer_is_rejected(tmp_path):
    files = {**bundle_files("B1"), "model/weights.json": "{}"}
    env = Env(tmp_path, [writes(files), replaces(bundle_files("B1"))])
    env.writer.write(env.request("B1"))
    assert "must not contain a model/ folder" in env.client.prompts[1]


def test_too_many_files_are_rejected(tmp_path):
    from kaggle_agent.writer import BundleLimits

    files = {**bundle_files("B1"), **{f"extra{i}.txt": "x" for i in range(5)}}
    env = Env(tmp_path, [writes(files), replaces(bundle_files("B1"))], limits=BundleLimits(max_files=4))
    env.writer.write(env.request("B1"))
    assert "too many files" in env.client.prompts[1]


def test_a_bundle_never_repaired_raises_after_a_bounded_number_of_calls(tmp_path):
    env = Env(tmp_path, [wrong_id_step()] * 5, max_repairs=2)
    request = env.request("B1")
    with pytest.raises(InvalidBundleError) as caught:
        env.writer.write(request)
    assert len(env.client.prompts) == 3
    assert isinstance(caught.value, BundleError)
    assert caught.value.turns == 3 and "expected 'B1'" in caught.value.last_error
    assert not request.folder.exists()
    assert list(env.root.iterdir()) == []


def test_without_repairs_exactly_one_call_is_made(tmp_path):
    env = Env(tmp_path, [wrong_id_step()] * 3, max_repairs=0)
    with pytest.raises(InvalidBundleError) as caught:
        env.writer.write(env.request("B1"))
    assert len(env.client.prompts) == 1 and caught.value.turns == 1


def test_a_negative_repair_bound_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Env(tmp_path, [], max_repairs=-1)


def test_the_check_error_is_a_layout_error():
    from kaggle_agent.writer import check_bundle

    with pytest.raises(BundleLayoutError):
        check_bundle(__import__("pathlib").Path("/nonexistent-folder-for-test"), "B1", ())
