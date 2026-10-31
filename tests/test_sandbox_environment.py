"""Minimal environment of the child. No network."""

import os

import pytest

from kaggle_agent.sandbox import MINIMAL_KEYS, SandboxConfigError, build_child_environment


def test_environment_has_only_minimal_keys_and_is_not_a_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "t")
    monkeypatch.setenv("MY_SECRET", "s")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    before = dict(os.environ)
    env = build_child_environment(tmp_path)
    assert set(env) == set(MINIMAL_KEYS)
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"].startswith(str(tmp_path)) and env["TMPDIR"].startswith(str(tmp_path))
    assert env["SANDBOX_PROTECTED_DIR"] == str(tmp_path / "protected")
    assert "PYTHONPATH" not in env
    assert dict(os.environ) == before


def test_harmless_extra_env_is_added(tmp_path):
    assert build_child_environment(tmp_path, {"SEED": "3"})["SEED"] == "3"


@pytest.mark.parametrize(
    "name",
    ["KAGGLE_KEY", "API_TOKEN", "pythonpath", "PYTHONPATH", "my_secret", "DB_PASSWORD",
     "ANTHROPIC_X", "CLAUDE_Y", "AWS_CREDENTIALS", "HOME", "PATH"],
)
def test_sensitive_or_reserved_extra_env_is_rejected(tmp_path, name):
    with pytest.raises(SandboxConfigError):
        build_child_environment(tmp_path, {name: "v"})
