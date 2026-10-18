"""Credential lookup order, permissions and secrecy (design section 10)."""

import json
import os
import pickle

import pytest

from kaggle_agent.kaggle import (
    Credentials,
    CredentialsNotFoundError,
    InsecureCredentialsError,
    load_credentials,
)
from kaggle_agent.kaggle import credentials as credentials_module


def _file(home, name, content, mode=0o600):
    folder = home / ".kaggle"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def _legacy_json(user="u", key="k-file"):
    return json.dumps({"username": user, "key": key})


def test_new_token_from_env_variable(tmp_path):
    creds = load_credentials(env={"KAGGLE_API_TOKEN": "tok-env"}, home=tmp_path)
    assert creds.source == "env"
    assert creds.reveal_for_adapter() == ("tok-env", None, None)


def test_new_token_from_access_token_file(tmp_path):
    _file(tmp_path, "access_token", "tok-file\n")
    creds = load_credentials(env={}, home=tmp_path)
    assert creds.source == "access_token"
    assert creds.reveal_for_adapter()[0] == "tok-file"


def test_legacy_kaggle_json_fallback(tmp_path):
    _file(tmp_path, "kaggle.json", _legacy_json())
    creds = load_credentials(env={}, home=tmp_path)
    assert creds.source == "legacy_file"
    assert creds.reveal_for_adapter() == (None, "u", "k-file")


def test_legacy_environment_variables_are_the_third_source(tmp_path):
    creds = load_credentials(env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k-env"}, home=tmp_path)
    assert creds.source == "legacy_env"
    assert creds.reveal_for_adapter() == (None, "u", "k-env")
    for partial in ({"KAGGLE_USERNAME": "u"}, {"KAGGLE_KEY": "k"}):
        with pytest.raises(CredentialsNotFoundError):
            load_credentials(env=partial, home=tmp_path)


def test_lookup_order_and_documentation(tmp_path):
    _file(tmp_path, "access_token", "tok-file")
    _file(tmp_path, "kaggle.json", _legacy_json())
    env = {"KAGGLE_API_TOKEN": "tok-env", "KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k-env"}
    chosen = []
    chosen.append(load_credentials(env=env, home=tmp_path).source)
    del env["KAGGLE_API_TOKEN"]
    chosen.append(load_credentials(env=env, home=tmp_path).source)
    (tmp_path / ".kaggle" / "access_token").unlink()
    chosen.append(load_credentials(env=env, home=tmp_path).source)
    env.clear()
    chosen.append(load_credentials(env=env, home=tmp_path).source)
    assert chosen == ["env", "access_token", "legacy_env", "legacy_file"]
    doc = credentials_module.__doc__
    positions = [doc.index(item) for item in (
        "KAGGLE_API_TOKEN", "~/.kaggle/access_token", "KAGGLE_USERNAME", "~/.kaggle/kaggle.json",
    )]
    assert positions == sorted(positions)


def test_absent_or_empty_credentials(tmp_path):
    with pytest.raises(CredentialsNotFoundError) as info:
        load_credentials(env={}, home=tmp_path)
    for place in ("KAGGLE_API_TOKEN", "access_token", "KAGGLE_USERNAME", "kaggle.json"):
        assert place in str(info.value)
    _file(tmp_path, "access_token", "  \n")
    empty = {"KAGGLE_API_TOKEN": "  ", "KAGGLE_USERNAME": "", "KAGGLE_KEY": " "}
    with pytest.raises(CredentialsNotFoundError):
        load_credentials(env=empty, home=tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_files_looser_than_600_are_refused(tmp_path):
    token = _file(tmp_path, "access_token", "tok-secret-file", mode=0o644)
    with pytest.raises(InsecureCredentialsError) as info:
        load_credentials(env={}, home=tmp_path)
    assert "tok-secret-file" not in str(info.value)
    token.unlink()
    _file(tmp_path, "kaggle.json", _legacy_json(key="k-secret-file"), mode=0o644)
    with pytest.raises(InsecureCredentialsError) as info:
        load_credentials(env={}, home=tmp_path)
    assert "k-secret-file" not in str(info.value)
    (tmp_path / ".kaggle" / "kaggle.json").chmod(0o600)
    assert load_credentials(env={}, home=tmp_path).source == "legacy_file"
    _file(tmp_path, "access_token", "tok", mode=0o600)
    assert load_credentials(env={}, home=tmp_path).source == "access_token"


@pytest.mark.parametrize("content", ["not json {", '{"username": "u"}', "[1, 2]"])
def test_malformed_legacy_file_is_not_echoed(tmp_path, content):
    _file(tmp_path, "kaggle.json", content)
    with pytest.raises(CredentialsNotFoundError) as info:
        load_credentials(env={}, home=tmp_path)
    assert content not in str(info.value)


def test_credentials_hide_the_secret():
    creds = Credentials("env", token="tok-hidden-1")
    for text in (repr(creds), str(creds), f"{creds}", f"{creds!r:>40}", "%s" % creds):
        assert "tok-hidden-1" not in text
    with pytest.raises(TypeError):
        pickle.dumps(creds)
    assert not hasattr(creds, "__dict__")
