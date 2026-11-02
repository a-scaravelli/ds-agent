"""Isolation of the staging workspace: escapes, protected paths and the request folder."""

import os
from pathlib import Path

import pytest
from writer_helpers import Env, bundle_files, protected_file, put, replaces, valid_step, writes

from kaggle_agent.sandbox import hash_paths
from kaggle_agent.writer import (
    BundleWriterError, WorkspaceViolationError, default_protected,
)

SRC = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent"


def test_a_write_outside_the_staging_directory_is_detected(tmp_path):
    def escaping(working_dir):
        put(working_dir, bundle_files("B1"))
        (working_dir.parent / "escape.txt").write_text("out", encoding="utf-8")

    env = Env(tmp_path, [escaping, valid_step("B1")])
    request = env.request("B1")
    with pytest.raises(WorkspaceViolationError, match="escape.txt"):
        env.writer.write(request)
    assert len(env.client.prompts) == 1  # no repair turn
    assert not request.folder.exists()
    assert not env.client.permissions[0].working_dir.exists()


def test_a_write_to_a_protected_path_is_detected_and_the_real_sources_are_untouched(tmp_path):
    entry, path = protected_file(tmp_path)
    real = {p.name: v for p, v in zip(default_protected(), hash_paths(default_protected()).values())}

    def touching(working_dir):
        put(working_dir, bundle_files("B1"))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("tampered")

    env = Env(tmp_path, [touching, valid_step("B1")], protected=(entry,))
    with pytest.raises(WorkspaceViolationError, match="tmp/guarded.txt"):
        env.writer.write(env.request("B1"))
    assert len(env.client.prompts) == 1
    assert list(hash_paths(default_protected()).values()) == list(real.values())


def test_a_file_next_to_the_bundle_folder_is_a_violation(tmp_path):
    def stray(working_dir):
        put(working_dir, bundle_files("B1"))
        (working_dir / "notes.txt").write_text("x", encoding="utf-8")

    env = Env(tmp_path, [stray])
    with pytest.raises(WorkspaceViolationError, match="notes.txt"):
        env.writer.write(env.request("B1"))


def test_files_created_in_the_request_folder_during_the_call_are_a_violation(tmp_path):
    env = Env(tmp_path, [])
    request = env.request("B1")

    def into_request_folder(working_dir):
        put(working_dir, bundle_files("B1"))
        request.folder.mkdir()
        (request.folder / "sneaky.py").write_text("x", encoding="utf-8")

    env.client.steps = [into_request_folder]
    with pytest.raises(WorkspaceViolationError, match="request folder"):
        env.writer.write(request)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_a_link_pointing_outside_the_bundle_is_a_repairable_invalid_bundle(tmp_path):
    target = tmp_path.resolve() / "outside.txt"
    target.write_text("secret", encoding="utf-8")

    def linking(working_dir):
        put(working_dir, bundle_files("B1"))
        try:
            os.symlink(target, working_dir / "bundle" / "link.txt")
        except OSError:
            pytest.skip("cannot create symlinks here")

    env = Env(tmp_path, [linking, replaces(bundle_files("B1"))])
    request = env.request("B1")
    env.writer.write(request)
    assert "points outside the bundle folder" in env.client.prompts[1]
    assert not (request.folder / "link.txt").exists()


def test_a_non_empty_request_folder_is_refused_before_any_call(tmp_path):
    env = Env(tmp_path, [valid_step("B1")])
    request = env.request("B1")
    request.folder.mkdir()
    existing = request.folder / "keep.txt"
    existing.write_text("mine", encoding="utf-8")
    with pytest.raises(BundleWriterError):
        env.writer.write(request)
    assert env.client.prompts == []
    assert existing.read_text(encoding="utf-8") == "mine"


def test_an_existing_empty_request_folder_is_accepted(tmp_path):
    env = Env(tmp_path, [valid_step("B1")])
    request = env.request("B1")
    request.folder.mkdir()
    assert env.writer.write(request) == request.folder
    assert (request.folder / "bundle.json").is_file()


def test_other_files_of_the_workspace_may_exist_untouched(tmp_path):
    env = Env(tmp_path, [valid_step("B1")])
    env.root.mkdir()
    (env.root / "old.txt").write_text("old", encoding="utf-8")
    env.writer.write(env.request("B1"))
    assert (env.root / "old.txt").read_text(encoding="utf-8") == "old"
