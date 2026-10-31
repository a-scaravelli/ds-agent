"""Hashes, diff check and read-only copies of the protected files. No network."""

import hashlib
import os
import stat

import pytest
from sandbox_helpers import EVAL_NAME, HOLDOUT_NAME, PROTOCOL_NAME, RUNNER_NAME, make_policy

from kaggle_agent.sandbox import ProtectedPath, SandboxConfigError, diff_touched_paths, hash_paths
from kaggle_agent.sandbox.protected import make_readonly_copies, make_writable


def test_hash_of_file_and_folder_matches_a_hand_computed_sha256(tmp_path):
    policy = make_policy(tmp_path)
    hashes = hash_paths(policy.protected)
    assert set(hashes) == {p.name for p in policy.protected}
    assert hashes[PROTOCOL_NAME] == hashlib.sha256(b'{"metric": "auc"}\n').hexdigest()
    digest = hashlib.sha256()
    digest.update(b"metrics.py\0VALUE = 1\n\0sub/inner.py\0INNER = 2\n\0")
    assert hashes[EVAL_NAME] == digest.hexdigest()


def test_pycache_is_ignored_and_content_changes_the_hash(tmp_path):
    policy = make_policy(tmp_path)
    before = hash_paths(policy.protected)
    (tmp_path / EVAL_NAME / "__pycache__" / "metrics.pyc").write_bytes(b"other")
    assert hash_paths(policy.protected) == before
    (tmp_path / EVAL_NAME / "metrics.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert hash_paths(policy.protected)[EVAL_NAME] != before[EVAL_NAME]


def test_missing_protected_path_is_a_config_error_or_a_marker_after_a_run(tmp_path):
    policy = make_policy(tmp_path)
    (tmp_path / RUNNER_NAME).unlink()
    with pytest.raises(SandboxConfigError):
        hash_paths(policy.protected)
    assert hash_paths(policy.protected, missing_ok=True)[RUNNER_NAME] == "<missing>"


GIT_DIFF = """diff --git a/{p} b/{p}
index 111..222 100644
--- a/{p}
+++ b/{p}
@@ -1 +1 @@
-x
+y
"""


@pytest.mark.parametrize(
    "diff, expected",
    [
        (GIT_DIFF.format(p=EVAL_NAME + "/metrics.py"), EVAL_NAME),
        (GIT_DIFF.format(p=PROTOCOL_NAME), PROTOCOL_NAME),
        (GIT_DIFF.format(p=RUNNER_NAME), RUNNER_NAME),
        (f"--- {EVAL_NAME}/new.py\n+++ {EVAL_NAME}/new.py\n", EVAL_NAME),
        (f"--- /dev/null\n+++ b/{EVAL_NAME}/created.py\n", EVAL_NAME),
        (f"--- a/{EVAL_NAME}/metrics.py\n+++ /dev/null\n", EVAL_NAME),
        (f"diff --git a/x.py b/y.py\nrename from x.py\nrename to {EVAL_NAME}/y.py\n", EVAL_NAME),
        (f"--- a/bundle/../{PROTOCOL_NAME}\t2024-01-01\n+++ b/bundle/../{PROTOCOL_NAME}\n", PROTOCOL_NAME),
        (f"--- a/bundle//..//{EVAL_NAME}//metrics.py\n+++ b/bundle/x\n", EVAL_NAME),
        (f"--- a/./{EVAL_NAME}/metrics.py\n+++ b/./{EVAL_NAME}/metrics.py\n", EVAL_NAME),
    ],
)
def test_diff_touching_protected_entries_is_reported(tmp_path, diff, expected):
    policy = make_policy(tmp_path)
    assert expected in diff_touched_paths(diff, policy.protected)


def test_paths_escaping_the_root_are_suspicious(tmp_path):
    policy = make_policy(tmp_path)
    for text in ("--- a/../src/x.py\n+++ b/../src/x.py\n", "--- ../../etc/passwd\n+++ ../../etc/passwd\n"):
        hits = diff_touched_paths(text, policy.protected)
        assert hits and all(h.startswith("<escapes root>") for h in hits)
    absolute = diff_touched_paths(f"--- {tmp_path / PROTOCOL_NAME}\n+++ {tmp_path / PROTOCOL_NAME}\n", policy.protected)
    assert PROTOCOL_NAME in absolute
    assert diff_touched_paths("--- /etc/passwd\n+++ /etc/passwd\n", policy.protected)


def test_diff_on_other_files_is_not_reported(tmp_path):
    policy = make_policy(tmp_path)
    diff = (
        "--- a/bundle/model.py\n+++ b/bundle/model.py\n"
        "--- a/evaluation_notes.md\n+++ b/evaluation_notes.md\n"
        f"--- a/{EVAL_NAME}_notes.md\n+++ b/{EVAL_NAME}_notes.md\n"
        "--- a/protocol.json.bak\n+++ b/protocol.json.bak\n"
    )
    assert diff_touched_paths(diff, policy.protected) == ()


def test_copies_only_of_exposed_entries_and_read_only(tmp_path):
    policy = make_policy(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    copies = make_readonly_copies(policy.protected, dest)
    try:
        assert {c.name for c in copies} == {EVAL_NAME, PROTOCOL_NAME, RUNNER_NAME}
        assert not (dest / HOLDOUT_NAME).exists()
        assert hash_paths(copies) == {
            k: v for k, v in hash_paths(policy.protected).items() if k != HOLDOUT_NAME
        }
        assert not (dest / EVAL_NAME / "__pycache__").exists()
        assert stat.S_IMODE(os.stat(dest / PROTOCOL_NAME).st_mode) == 0o444
        assert stat.S_IMODE(os.stat(dest / EVAL_NAME).st_mode) == 0o555
    finally:
        make_writable(dest)
    assert os.access(dest / PROTOCOL_NAME, os.W_OK)


def test_protected_name_must_be_relative(tmp_path):
    for bad in ("/abs/x", "../x", ""):
        with pytest.raises(SandboxConfigError):
            ProtectedPath(bad, tmp_path)
