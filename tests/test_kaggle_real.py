"""The real adapter, exercised through a stub in place of the library: no network."""

import importlib
import ast
import importlib.metadata
import importlib.util
import os
import sys
from types import SimpleNamespace
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
from kaggle_helpers import (
    SECRET,
    TITANIC_PAGE_NAMES,
    HttpError,
    StubApi,
    make_pages,
    write_submission,
)

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
    _needs_library()
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
    assert info.description == "text of Description"
    assert info.rules == "text of rules"
    assert info.evaluation == "text of Evaluation"
    assert info.data_description == "text of data-description"
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
    api.errors["competition_submissions"] = HttpError(500, SECRET)
    with pytest.raises(KaggleError) as server:
        client.get_submission("comp", "77")
    assert "competition_get_submission_limits: RuntimeError" in str(other.value)
    assert "competition_submissions: HTTP 500" in str(server.value)
    for error in (info, info404, other, unauthorized, server):
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


def _needs_library():
    if importlib.util.find_spec("kaggle") is None:
        pytest.skip("the kaggle library is not installed")
    # six is now a declared dependency; this only guards a library or a dependency not installed here
    saved = {k: v for k, v in sys.modules.items() if k == "kaggle" or k.startswith("kaggle.")}
    try:
        real.load_api_class()
    except KaggleError:
        pytest.skip("the kaggle library is installed but one of its dependencies is missing")
    finally:
        for name in [k for k in sys.modules if k == "kaggle" or k.startswith("kaggle.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_placeholder_carries_the_installed_version(clean_library_modules):
    _needs_library()
    cls = real.load_api_class()
    module = sys.modules[real._API_MODULE]
    assert module.kaggle.__version__ == importlib.metadata.version("kaggle")
    assert "kaggle" not in sys.modules
    assert cls._authenticated is False


def test_response_processor_runs_on_a_canned_response(clean_library_modules, capsys):
    _needs_library()
    cls = real.load_api_class()
    installed = importlib.metadata.version("kaggle")
    api = SimpleNamespace(already_printed_version_warning=False)
    for headers in ({"X-Kaggle-APIVersion": installed}, {}):
        cls._check_response_version(api, SimpleNamespace(headers=headers))
        assert capsys.readouterr().out == ""
        assert api.already_printed_version_warning is False
    cls._check_response_version(api, SimpleNamespace(headers={"X-Kaggle-APIVersion": "999.0.0"}))
    assert "version" in capsys.readouterr().out.lower()
    assert api.already_printed_version_warning is True


def _global_kaggle_attributes(tree):
    found = set()

    def rebinds(func):
        return any(
            isinstance(n, ast.Name) and n.id == "kaggle" and isinstance(n.ctx, ast.Store)
            for n in ast.walk(func)
        )

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and rebinds(node):
            return
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "kaggle"
        ):
            found.add(node.attr)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def test_only_version_and_benchmarks_are_read_from_the_package_object():
    _needs_library()
    root = importlib.util.find_spec("kaggle").submodule_search_locations[0]
    source = os.path.join(root, "api", "kaggle_api_extended.py")
    with open(source, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    # "benchmarks" is used only by the benchmark-task methods, which the adapter never calls
    assert _global_kaggle_attributes(tree) == {"__version__", "benchmarks"}, (
        "the pinned kaggle version reads other attributes of the package object: "
        "update the placeholder in kaggle/real.py"
    )


def test_sanitized_error_names_the_function_and_type(tmp_path):
    client, api = make_real()
    api.errors["competition_get_submission_limits"] = RuntimeError(SECRET)
    with pytest.raises(KaggleError) as info:
        client.daily_limit("comp")
    assert "competition_get_submission_limits: RuntimeError" in str(info.value)
    assert SECRET not in str(info.value)


# -- page names (second bug found by the manual smoke test) --------------------


def test_mixed_case_pages_fill_description_evaluation_rules_and_data_description():
    client, _ = make_real()
    info = client.get_competition_info("titanic")
    assert info.description == "text of Description"
    assert info.evaluation == "text of Evaluation"
    assert info.rules == "text of rules"
    assert info.data_description == "text of data-description"
    assert all((info.description, info.evaluation, info.rules, info.data_description))


def test_pages_are_listed_once_and_without_page_name():
    client, api = make_real()
    client.get_competition_info("titanic")
    assert api.calls.count("competition_list_pages") == 1
    assert api.page_name_args == [None]


def test_every_page_is_kept_in_the_pages_mapping():
    client, _ = make_real()
    info = client.get_competition_info("titanic")
    assert dict(info.pages) == {name: f"text of {name}" for name in TITANIC_PAGE_NAMES}
    assert "Frequently Asked Questions" in info.pages
    with pytest.raises(TypeError):
        info.pages["new"] = "x"


@pytest.mark.parametrize("missing", ["rules", "Description", "Evaluation"])
def test_missing_required_page_raises_naming_it_and_the_available_names(missing):
    client, api = make_real()
    names = [n for n in TITANIC_PAGE_NAMES if n != missing]
    api.pages = make_pages(names)
    with pytest.raises(KaggleError) as info:
        client.get_competition_info("titanic")
    message = str(info.value)
    assert f"'{missing.lower()}'" in message
    listed = ", ".join(repr(n) for n in sorted(names))
    assert listed in message
    assert "text of" not in message and SECRET not in message


def test_missing_data_description_is_an_empty_string():
    client, api = make_real()
    api.pages = make_pages([n for n in TITANIC_PAGE_NAMES if n != "data-description"])
    info = client.get_competition_info("titanic")
    assert info.data_description == ""
    assert "data-description" not in info.pages


def test_page_names_are_normalized_across_case_and_separators():
    pages = [
        SimpleNamespace(name="Data Description", content="first"),
        SimpleNamespace(name="data_description", content="second"),
        SimpleNamespace(name=" EVALUATION ", content="eval"),
        SimpleNamespace(name="Rules", content="rules"),
        SimpleNamespace(name="description", content=None),
    ]
    info = real._info_from_pages("comp", pages)
    assert info.data_description == "first"
    assert info.evaluation == "eval" and info.rules == "rules"
    assert info.description == ""
    assert info.pages["Data Description"] == "first"
    assert info.pages["data_description"] == "second"
    assert real._normalize_page_name("Data  Description") == real._normalize_page_name("data-description")


# -- sample submission selection (first real run, competition 'titanic') ------

TRAIN_TEST = ["train.csv", "test.csv"]


def _spec_for(listed, competition="comp"):
    client, api = make_real()
    api.listed_files = list(listed)
    return client.get_submission_spec(competition), api


# 1
def test_sample_submission_csv_is_still_found():
    spec, api = _spec_for(["train.csv", "sample_submission.csv"])
    assert api.downloaded_names == ["sample_submission.csv"]
    assert spec.columns == ("id", "target") and spec.expected_ids == ("a", "b", "c")


# 2
def test_titanic_gender_submission_is_accepted():
    spec, api = _spec_for([*TRAIN_TEST, "gender_submission.csv"], "titanic")
    assert api.downloaded_names == ["gender_submission.csv"]
    assert spec.columns == ("id", "target")


# 3
@pytest.mark.parametrize("order", [("gender_submission.csv", "sample_submission.csv"),
                                   ("sample_submission.csv", "gender_submission.csv")])
def test_sample_and_submission_name_wins_over_submission_only(order):
    _, api = _spec_for([*TRAIN_TEST, *order])
    assert api.downloaded_names == ["sample_submission.csv"]


# 4
def test_selection_is_case_insensitive():
    _, api = _spec_for(["Gender_Submission.csv", "Sample_Submission.CSV"])
    assert api.downloaded_names == ["Sample_Submission.CSV"]
    _, api = _spec_for(["Train.CSV", "Gender_Submission.CSV"])
    assert api.downloaded_names == ["Gender_Submission.CSV"]


# 5
@pytest.mark.parametrize("names, expected", [
    (["sample_submission_v2.csv", "sample_submission.csv", "my_sample_submission.csv"], "sample_submission.csv"),
    (["sample_submission_v2.csv", "a_sample_submission_b.csv"], "sample_submission_v2.csv"),
    (["long_name_submission.csv", "gender_submission.csv", "a_submission.csv"], "a_submission.csv"),
    (["b_submission.csv", "a_submission.csv"], "a_submission.csv"),
])
def test_best_match_then_shortest_name_whatever_the_listing_order(names, expected):
    import itertools

    for order in itertools.permutations(names):
        _, api = _spec_for([*TRAIN_TEST, *order])
        assert api.downloaded_names == [expected], order


# 6
def test_only_csv_files_qualify():
    client, api = make_real()
    api.listed_files = ["train.csv", "sample_submission.csv.zip", "sample_submission.json", "submission_notes.txt"]
    with pytest.raises(KaggleError, match="has no sample submission file"):
        client.get_submission_spec("comp")
    assert "competition_download_file" not in api.calls and api.downloaded_names == []


# 7
@pytest.mark.parametrize("listed, shown", [
    (["train.csv", "test.csv"], "files seen: 'test.csv', 'train.csv'"),
    ([], "files seen: none"),
])
def test_no_candidate_keeps_the_error_and_lists_the_names_seen(listed, shown):
    client, api = make_real()
    api.listed_files = listed
    with pytest.raises(KaggleError) as info:
        client.get_submission_spec("comp")
    message = str(info.value)
    assert message.startswith("competition 'comp' has no sample submission file")
    assert shown in message and SECRET not in message


# 8
def test_only_the_sample_file_is_downloaded():
    _, api = _spec_for([*TRAIN_TEST, "gender_submission.csv"])
    assert api.downloaded_names == ["gender_submission.csv"]
    assert api.calls.count("competition_download_file") == 1
    assert "competition_download_files" not in api.calls


# 10
def test_six_is_pinned_with_an_explanatory_comment():
    import tomllib
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    deps = tomllib.loads(text)["project"]["dependencies"]
    assert [d for d in deps if d.startswith("six")] == [f"six=={importlib.metadata.version('six')}"]
    comment = "\n".join(line for line in text.splitlines() if line.strip().startswith("#"))
    assert "kaggle==2.2.4" in comment and "upload_file.py" in comment and "does not declare" in comment
