"""The real adapter, exercised through a stub in place of the library: no network."""

import importlib
import importlib.util
import os
import sys
from datetime import timezone

import pytest

import kaggle_agent.kaggle as kaggle_package
from kaggle_agent.kaggle import (
    CompetitionNotFoundError,
    Credentials,
    DailyLimitReachedError,
    KaggleError,
    RulesNotAcceptedError,
    SubmissionFormatError,
    load_credentials,
)
from kaggle_agent.kaggle import real
from kaggle_agent.kaggle.real import RealKaggleClient
from kaggle_helpers import SECRET, HttpError, StubApi, write_submission

CREDS = Credentials("env", token=SECRET)


def make_real(creds=CREDS):
    holder = {}

    def factory():
        holder["api"] = StubApi()
        return holder["api"]

    client = RealKaggleClient(creds, api_class=factory)
    return client, holder["api"]


@pytest.fixture
def clean_library_modules():
    saved = {k: v for k, v in sys.modules.items() if k == "kaggle" or k.startswith("kaggle.")}
    for name in saved:
        del sys.modules[name]
    yield
    for name in [k for k in sys.modules if k == "kaggle" or k.startswith("kaggle.")]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_package_and_adapter_import_without_the_library(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "kaggle", None)
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(kaggle_package)
    importlib.reload(real)
    with pytest.raises(KaggleError, match="not installed") as info:
        real.RealKaggleClient()
    assert "authenticat" not in str(info.value).lower()


def test_loading_the_api_class_does_not_authenticate(monkeypatch, clean_library_modules):
    if importlib.util.find_spec("kaggle") is None:
        pytest.skip("the kaggle library is not installed")
    monkeypatch.setenv("KAGGLE_API_TOKEN", SECRET)
    cls = real.load_api_class()
    assert cls.__name__ == "KaggleApi"
    # the package __init__ (which authenticates at import) did not run
    assert "kaggle" not in sys.modules
    assert cls._authenticated is False


def test_no_authenticate_before_credentials_are_resolved(monkeypatch):
    events = []
    StubApi.events = events
    monkeypatch.setattr(StubApi, "events", events)

    def fake_load(env=None, home=None):
        events.append("credentials_loaded")
        return CREDS

    monkeypatch.setattr(real, "load_credentials", fake_load)
    client = RealKaggleClient(api_class=StubApi)
    assert events == ["credentials_loaded", "api_created"]
    assert client._api.authenticate_calls == 0
    assert client._api.config_values == {"token": SECRET}


def test_legacy_credentials_are_handed_over_as_username_and_key():
    legacy = Credentials("legacy_file", username="u", key="k-1")
    _, api = make_real(legacy)
    assert api.config_values == {"username": "u", "key": "k-1"}


def test_resolution_uses_the_documented_lookup(tmp_path):
    client = RealKaggleClient(env={"KAGGLE_API_TOKEN": "tok"}, home=tmp_path, api_class=StubApi)
    assert client._api.config_values == {"token": "tok"}
    assert load_credentials(env={"KAGGLE_API_TOKEN": "tok"}, home=tmp_path).source == "env"


def test_secret_is_not_placed_in_the_environment(tmp_path):
    before = dict(os.environ)
    client, api = make_real()
    client.download("comp", tmp_path / "d")
    client.get_competition_info("comp")
    client.submit("comp", write_submission(tmp_path / "s.csv"), "m")
    assert dict(os.environ) == before
    assert api.env_seen and all(env == before for env in api.env_seen)
    assert api.config_values["token"] == SECRET
    assert SECRET not in "".join(os.environ.values())


def test_responses_map_to_model_types(tmp_path):
    client, api = make_real()
    result = client.download("comp", tmp_path / "d")
    assert [f.name for f in result.files] == ["test.csv", "train.csv"]
    assert (tmp_path / "d" / "train.csv").is_file()
    assert not list((tmp_path / "d").glob("*.zip"))
    info = client.get_competition_info("comp")
    assert (info.description, info.rules, info.evaluation) == ("desc", "rules", "eval")
    spec = client.get_submission_spec("comp")
    assert spec.columns == ("id", "target") and spec.expected_ids == ("a", "b", "c")
    api.num_today, api.num_allowed_now = 1, 3
    limit = client.daily_limit("comp")
    assert (limit.limit, limit.used, limit.remaining) == (4, 1, 3)
    done = client.get_submission("comp", "77")
    assert (done.status, done.public_score, done.submission_id) == ("complete", 0.81, "77")
    assert done.submitted_at.tzinfo == timezone.utc
    failed = client.get_submission("comp", "78")
    assert failed.status == "error" and failed.public_score is None
    with pytest.raises(KaggleError):
        client.get_submission("comp", "999")


def test_library_errors_map_to_sanitized_kaggle_errors(tmp_path):
    client, api = make_real()
    api.errors["competition_list_pages"] = HttpError(403, f"Forbidden for token {SECRET}")
    with pytest.raises(RulesNotAcceptedError) as info:
        client.get_competition_info("comp")
    api.errors["competition_download_files"] = HttpError(404, f"not found {SECRET}")
    with pytest.raises(CompetitionNotFoundError) as info404:
        client.download("comp", tmp_path / "d")
    api.errors["competition_get_submission_limits"] = RuntimeError(f"bad token {SECRET}")
    with pytest.raises(KaggleError) as other:
        client.daily_limit("comp")
    api.errors["competition_submissions"] = HttpError(401, SECRET)
    with pytest.raises(KaggleError) as unauthorized:
        client.get_submission("comp", "77")
    for error in (info, info404, other, unauthorized):
        assert SECRET not in str(error.value)
        assert error.value.__cause__ is None and error.value.__suppress_context__


def test_format_and_limit_are_enforced_before_the_library_sends(tmp_path):
    client, api = make_real()
    with pytest.raises(SubmissionFormatError):
        client.submit("comp", write_submission(tmp_path / "bad.csv", ["a,1"]), "m")
    assert "competition_submit" not in api.calls
    good = write_submission(tmp_path / "good.csv")
    api.num_allowed_now = 0
    with pytest.raises(DailyLimitReachedError):
        client.submit("comp", good, "m")
    assert api.calls.count("competition_submit") == 0
    api.num_allowed_now = 1
    record = client.submit("comp", good, "message")
    assert api.calls.count("competition_submit") == 1
    assert record.status == "pending" and record.submission_id == "77" and record.message == "message"
    assert len(record.file_sha256) == 64


def test_unconfirmed_submission_is_an_error(tmp_path):
    client, api = make_real()
    api.submit_ref = 0
    with pytest.raises(KaggleError):
        client.submit("comp", write_submission(tmp_path / "s.csv"), "m")
