"""Fix requests: the previous bundle is read, never modified; the result is a new bundle."""

import pytest
from writer_helpers import (
    CYCLE, MODEL_SOURCE, Env, bundle_files, fix_request, put, valid_step,
)

from kaggle_agent.bundle import BundleError, folder_hash_of, load_bundle, register_bundle
from kaggle_agent.registry import BundleRecord
from kaggle_agent.writer import PreviousBundleError


def previous_bundle(folder, marker, bundle_id="B1"):
    put(folder.parent, bundle_files(bundle_id, source=MODEL_SOURCE + f"\n# marker {marker}\n"), folder.name)
    return folder


def new_step():
    return valid_step("B1-f1")


def test_the_prompt_carries_the_previous_code_the_reason_and_the_new_id(tmp_path):
    env = Env(tmp_path, [new_step()])
    old = previous_bundle(env.bundles / "B1", "OLDCODE")
    register_bundle(env.registry, load_bundle(old), CYCLE, reference=str(old))
    env.writer.write(fix_request(env.bundles / "B1-f1"))
    prompt = env.client.prompts[0]
    assert "# marker OLDCODE" in prompt and "KeyError: age" in prompt
    assert "B1-f1" in prompt and "corrected bundle" in prompt and "previous bundle B1 failed" in prompt


def test_the_registry_reference_wins_over_a_sibling_decoy(tmp_path):
    env = Env(tmp_path, [new_step()])
    referenced = previous_bundle(env.base / "elsewhere" / "B1", "REFERENCED")
    previous_bundle(env.bundles / "B1", "DECOY")  # sibling of the new folder, same id
    register_bundle(env.registry, load_bundle(referenced), CYCLE, reference=str(referenced))
    before = folder_hash_of(referenced)
    env.writer.write(fix_request(env.bundles / "B1-f1"))
    prompt = env.client.prompts[0]
    assert "marker REFERENCED" in prompt and "marker DECOY" not in prompt
    assert folder_hash_of(referenced) == before


def test_the_sibling_folder_is_used_only_when_the_registry_gives_nothing(tmp_path):
    # (a) no record
    (tmp_path / "a").mkdir()
    env = Env(tmp_path / "a", [new_step()])
    previous_bundle(env.bundles / "B1", "SIBLING_A")
    env.writer.write(fix_request(env.bundles / "B1-f1"))
    assert "marker SIBLING_A" in env.client.prompts[0]
    # (b) a record whose reference no longer exists
    (tmp_path / "b").mkdir()
    env = Env(tmp_path / "b", [new_step()])
    previous_bundle(env.bundles / "B1", "SIBLING_B")
    env.registry.add_bundle(BundleRecord(
        id="B1", cycle_id=CYCLE, folder_hash="h", reference=str(env.base / "gone" / "B1"), schema={},
    ))
    folder = env.writer.write(fix_request(env.bundles / "B1-f1"))
    assert "marker SIBLING_B" in env.client.prompts[0]
    assert load_bundle(folder).bundle_id == "B1-f1"


def test_a_fix_produces_a_new_bundle_and_leaves_the_old_one_untouched(tmp_path):
    env = Env(tmp_path, [new_step()])
    old = previous_bundle(env.bundles / "B1", "OLD")
    before = folder_hash_of(old)
    request = fix_request(env.bundles / "B1-f1")
    folder = env.writer.write(request)
    assert folder == request.folder != old
    assert load_bundle(folder).bundle_id == "B1-f1"
    assert folder_hash_of(old) == before
    working = env.client.permissions[0].working_dir
    assert old not in working.parents and working not in old.parents


def test_an_unreadable_previous_bundle_fails_before_any_call(tmp_path):
    env = Env(tmp_path, [new_step()])
    with pytest.raises(PreviousBundleError) as caught:
        env.writer.write(fix_request(env.bundles / "B1-f1"))
    assert isinstance(caught.value, BundleError)
    assert env.client.prompts == []
    env.registry.add_bundle(BundleRecord(
        id="B1", cycle_id=CYCLE, folder_hash="h", reference=str(env.base / "gone"), schema={},
    ))
    with pytest.raises(PreviousBundleError):
        env.writer.write(fix_request(env.bundles / "B1-f1"))
    assert env.client.prompts == []  # a dangling record does not help either


def test_the_previous_code_is_capped(tmp_path):
    env = Env(tmp_path, [new_step()])
    big = MODEL_SOURCE + "\n# " + "x" * 60_000 + "\n"
    put(env.bundles, bundle_files("B1", source=big), "B1")
    env.writer.write(fix_request(env.bundles / "B1-f1"))
    assert "(truncated)" in env.client.prompts[0]
    assert len(env.client.prompts[0]) < 40_000
