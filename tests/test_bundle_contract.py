"""The bundle contract, the loader and validator, and the folder hash (cases 1 to 5)."""

import json
import os
import re

import pytest
from bundle_helpers import constant_bundle, model, write_bundle

from kaggle_agent.bundle import (
    MANIFEST_NAME, BundleLayoutError, Manifest, folder_hash_of, load_bundle, schema_of,
)
from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import BundleRecord
from kaggle_agent.sandbox import ProtectedPath, hash_paths


def test_load_valid_bundle_and_manifest_round_trip(tmp_path):
    folder = write_bundle(
        tmp_path, model("return [0] * len(X)", header="import pandas as pd"),
        dependencies=["pandas"], bundle_id="B7",
    )
    loaded = load_bundle(folder)
    assert loaded.bundle_id == "B7" and loaded.folder == folder.resolve()
    assert re.fullmatch(r"[0-9a-f]{64}", loaded.folder_hash)
    assert loaded.manifest.entry_points == {"fit": "model:fit", "predict": "model:predict"}
    schema = schema_of(loaded.manifest)
    json.dumps(schema)
    BundleRecord(id="B7", cycle_id="C1", folder_hash=loaded.folder_hash, reference="r", schema=schema)
    manifest = loaded.manifest
    assert Manifest.from_dict(manifest.to_dict()) == manifest
    with pytest.raises(ValidationError):
        Manifest.from_dict({**manifest.to_dict(), "surprise": 1})


@pytest.mark.parametrize("entry_points,expected", [
    ({"predict": "model:predict"}, "fit"),
    ({"fit": "model:fit"}, "predict"),
    ({"fit": "model:fit", "predict": "model:nothing"}, "predict"),
    ({"fit": "absent:fit", "predict": "model:predict"}, "fit"),
])
def test_reject_bundle_without_fit_or_predict(tmp_path, entry_points, expected):
    marker = tmp_path / "imported"
    source = model("return [0] * len(X)", header=f"open({str(marker)!r}, 'w').write('x')")
    folder = write_bundle(tmp_path, source, entry_points=entry_points)
    with pytest.raises(BundleLayoutError, match=expected):
        load_bundle(folder)
    assert not marker.exists()  # the bundle code is never imported by the loader


def test_reject_missing_manifest_and_non_folder(tmp_path):
    with pytest.raises(BundleLayoutError, match="not found"):
        load_bundle(tmp_path)
    with pytest.raises(BundleLayoutError, match="not a folder"):
        load_bundle(tmp_path / "absent")
    folder = constant_bundle(tmp_path)
    (folder / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(BundleLayoutError, match="invalid"):
        load_bundle(folder)


def test_reject_paths_that_escape_the_folder(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("def fit(train):\n    pass\n\ndef predict(bundle_id, X):\n    return []\n")
    for entry in ("../outside:fit", "/etc/passwd:fit", "a/b:fit", "model:../fit"):
        folder = write_bundle(
            tmp_path / entry.replace("/", "_"), model("return []"),
            entry_points={"fit": entry, "predict": "model:predict"},
        )
        with pytest.raises(BundleLayoutError):
            load_bundle(folder)
    linked = write_bundle(tmp_path / "linked", {"other.py": "X = 1\n"})
    (linked / "model.py").symlink_to(outside)
    with pytest.raises(BundleLayoutError, match="outside"):
        load_bundle(linked)
    folder = constant_bundle(tmp_path / "dirlink")
    os.symlink(tmp_path, folder / "escape")
    with pytest.raises(BundleLayoutError, match="outside"):
        load_bundle(folder)


def test_undeclared_third_party_imports(tmp_path):
    body = "import pandas as pd\nfrom sklearn.linear_model import Ridge\n"
    source = model("return [0] * len(X)", header=body)
    with pytest.raises(BundleLayoutError, match="sklearn"):
        load_bundle(write_bundle(tmp_path / "a", source, dependencies=["pandas"]))
    load_bundle(write_bundle(tmp_path / "b", source, dependencies=["pandas", "scikit-learn"]))
    load_bundle(write_bundle(tmp_path / "c", source, dependencies=["Pandas==3.0.6", "scikit_learn>=1"]))
    plain = model("return [0] * len(X)", header="import json\nimport helpers\nfrom . import x")
    load_bundle(write_bundle(tmp_path / "d", {**plain, "helpers.py": "import os\n"}))
    nested = model("import numpy\nreturn [0] * len(X)")
    with pytest.raises(BundleLayoutError, match="numpy"):
        load_bundle(write_bundle(tmp_path / "e", nested))


def test_folder_hash_is_deterministic_and_sensitive(tmp_path):
    folder = constant_bundle(tmp_path)
    first = load_bundle(folder).folder_hash
    assert load_bundle(folder).folder_hash == first
    assert first == hash_paths([ProtectedPath("x", folder)])["x"] == folder_hash_of(folder)
    (folder / "model.py").write_text((folder / "model.py").read_text() + "\n# edit\n")
    edited = load_bundle(folder).folder_hash
    assert edited != first
    (folder / "extra.py").write_text("Y = 2\n")
    assert load_bundle(folder).folder_hash not in (first, edited)
