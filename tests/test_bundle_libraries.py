"""xgboost, lightgbm and catboost through the whole chain: checks, loader and the real runner (cases 13 to 16, 18).

No library is imported at module level: it is imported in a helper and inside the bundle code that
runs in the child, so collection never fails where one is missing.
"""

import hashlib
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bundle_helpers import make_runner, write_bundle

from kaggle_agent.bundle import BundleLayoutError, load_bundle
from kaggle_agent.bundle.layout import folder_hash_of
from kaggle_agent.proposals import ALLOWED_LIBRARIES, LIBRARY_APIS
from kaggle_agent.writer import DEFAULT_DEPENDENCIES, check_bundle
from kaggle_agent.writer.prompt import build_prompt
from writer_helpers import make_request, task

ROOT = Path(__file__).resolve().parent.parent
BOOSTING = ("xgboost", "lightgbm", "catboost")

FIT_BODY = '''
import os
from {module} import {cls}

def fit(train):
    y = train["y"]
    x = train.drop(columns=["y"])
    clf = {build}
    clf.fit(x, y)
    clf.{save}(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "{file}"))

def predict(bundle_id, X):
    clf = {cls}()
    clf.{load}(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "{file}"))
    return [int(v) for v in clf.predict(X)]
'''

# per library: the constructor and the file the model is saved to
CONSTRUCTORS = {
    "xgboost": "XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, random_state=0)",
    "lightgbm": "LGBMClassifier(n_estimators=3, num_leaves=4, n_jobs=1, random_state=0, verbosity=-1)",
    "catboost": "CatBoostClassifier(iterations=3, depth=2, thread_count=1, random_seed=0, "
                "allow_writing_files=False, verbose=False)",
}
FILES = {"xgboost": "model.json", "lightgbm": "model.txt", "catboost": "model.cbm"}

SKLEARN_STYLE = '''
import os
from {module} import {cls}

def fit(train):
    clf = {build}
    clf.fit(train.drop(columns=["y"]), train["y"])
    clf.save_model(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "{file}"))

def predict(bundle_id, X):
    clf = {cls}()
    clf.load_model(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "{file}"))
    return [int(v) for v in clf.predict(X)]
'''

# a LightGBM model is saved from its booster and loaded back with lightgbm.Booster
LIGHTGBM = '''
import os
import lightgbm
from lightgbm import LGBMClassifier

def fit(train):
    clf = {build}
    clf.fit(train.drop(columns=["y"]), train["y"])
    clf.booster_.save_model(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "model.txt"))

def predict(bundle_id, X):
    booster = lightgbm.Booster(model_file=os.path.join(os.environ["BUNDLE_MODEL_DIR"], "model.txt"))
    return [int(p > 0.5) for p in booster.predict(X)]
'''
CLASSES = {"xgboost": "XGBClassifier", "catboost": "CatBoostClassifier"}


def source(name: str, build: str | None = None) -> str:
    build = build or CONSTRUCTORS[name]
    if name == "lightgbm":
        return LIGHTGBM.format(build=build)
    return SKLEARN_STYLE.format(module=name, cls=CLASSES[name], build=build, file=FILES[name])


def importable(name: str):
    """Skip the test only if ``import <name>`` itself fails here (with the error as the reason)."""
    try:
        __import__(name)
    except Exception as error:  # ImportError, or the OSError / XGBoostError / LightGBMError of a missing runtime
        pytest.skip(f"{name} is not importable: {type(error).__name__}: {error}")


def frame(rows=60):
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"x1": rng.normal(size=rows), "x2": rng.normal(size=rows)})
    data["y"] = (data["x1"] + 0.3 * rng.normal(size=rows) > 0).astype(int)
    return data


def split(rows=60):
    data = frame(rows)
    return data[["x1", "x2"]], data["y"]


def library_bundle(root, name, **kwargs):
    return write_bundle(root, {"model.py": source(name)}, dependencies=[name], **kwargs)


# 13
@pytest.mark.parametrize("name", BOOSTING)
def test_a_bundle_that_declares_and_imports_a_boosting_library_passes_the_checks(tmp_path, name):
    folder = library_bundle(tmp_path, name)
    loaded = check_bundle(folder, "B1", DEFAULT_DEPENDENCIES)
    assert loaded.manifest.dependencies == (name,) or list(loaded.manifest.dependencies) == [name]
    assert load_bundle(folder).bundle_id == "B1"


def test_a_stray_catboost_info_folder_does_not_break_the_checks(tmp_path):
    folder = library_bundle(tmp_path, "catboost")
    info = folder / "catboost_info"
    info.mkdir()
    for file_name in ("learn_error.tsv", "time_left.tsv", "catboost_training.json"):
        (info / file_name).write_text("0\t0.69\n", encoding="utf-8")
    check_bundle(folder, "B1", DEFAULT_DEPENDENCIES)
    assert load_bundle(folder).bundle_id == "B1"


# 14
@pytest.mark.parametrize("name", BOOSTING)
def test_a_tiny_bundle_trains_saves_and_predicts_in_other_processes(tmp_path, name):
    importable(name)
    _, runner = make_runner(library_bundle(tmp_path, name))
    x, y = split()
    model_dir = tmp_path / "saved"
    runner.fit_model(x, y, 1, model_dir)
    saved = sorted(p.name for p in model_dir.iterdir())
    assert saved == [FILES[name]]  # the model file, loaded back by predict in a fresh process
    predictions = runner.predict_with_model(model_dir, x.iloc[:7], 1)
    assert len(predictions) == 7 and set(predictions.tolist()) <= {0, 1}
    assert len(runner.fit_predict(x, y, x.iloc[:5], 1)) == 5


# 15
def test_a_catboost_bundle_without_allow_writing_files_still_runs_and_leaves_no_folder(tmp_path):
    importable("catboost")
    forgetful = source("catboost", "CatBoostClassifier(iterations=3, depth=2, thread_count=1, "
                                   "random_seed=0, verbose=False)")
    folder = write_bundle(tmp_path, {"model.py": forgetful}, dependencies=["catboost"])
    loaded, runner = make_runner(folder)
    before = folder_hash_of(folder)
    cwd_before = sorted(p.name for p in Path.cwd().iterdir())
    x, y = split()
    model_dir = tmp_path / "saved"
    runner.fit_model(x, y, 1, model_dir)
    predictions = runner.predict_with_model(model_dir, x.iloc[:6], 1)
    assert len(predictions) == 6
    assert folder_hash_of(folder) == before == loaded.folder_hash  # the registered folder is untouched
    assert not any("catboost_info" in str(p) for p in folder.rglob("*"))
    assert sorted(p.name for p in model_dir.iterdir()) == ["model.cbm"]  # no catboost_info copied out
    assert "catboost_info" not in {p.name for p in Path.cwd().iterdir()} - set(cwd_before)


# 16
@pytest.mark.parametrize("name", ("torch", "tensorflow"))
def test_an_undeclared_import_of_an_unlisted_library_is_rejected(tmp_path, name):
    folder = write_bundle(tmp_path, {"model.py": f"import {name}\n\ndef fit(train):\n    pass\n\n"
                                                  "def predict(bundle_id, X):\n    return [0] * len(X)\n"})
    with pytest.raises(BundleLayoutError):
        load_bundle(folder)


def test_a_declared_torch_is_rejected_by_the_writer_checks_with_the_six_allowed_names(tmp_path):
    folder = write_bundle(tmp_path, {"model.py": "import torch\n\ndef fit(train):\n    pass\n\n"
                                                 "def predict(bundle_id, X):\n    return [0] * len(X)\n"},
                          dependencies=["torch"])
    with pytest.raises(BundleLayoutError) as error:
        check_bundle(folder, "B1", DEFAULT_DEPENDENCIES)
    assert "torch" in str(error.value)
    assert all(name in str(error.value) for name in ALLOWED_LIBRARIES)


# 18 (pins) and 6 (the writer prompt lists the same six)
def test_the_boosting_libraries_are_pinned_and_listed_everywhere():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    for name in BOOSTING:
        (pin,) = [d for d in deps if d.startswith(f"{name}==")]
        assert pin.split("==")[1][:1].isdigit()
    assert ALLOWED_LIBRARIES == DEFAULT_DEPENDENCIES == (
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost")
    assert set(LIBRARY_APIS) == set(BOOSTING)
    text = build_prompt(make_request("x"), task(), DEFAULT_DEPENDENCIES)
    assert ", ".join(ALLOWED_LIBRARIES) in text
