"""The persisted bundle: layout, integrity checks and reuse (case 13)."""

import json
import os

import pytest
from pipeline_helpers import CYCLE, FRAME, make_pipeline

from kaggle_agent.bundle import folder_hash_of
from kaggle_agent.pipeline import PersistedBundleError, load_persisted, persist_bundle, persisted_root
from kaggle_agent.evaluation import FrozenSplit, derive_seed


@pytest.fixture
def run(tmp_path):
    pipe = make_pipeline(tmp_path, guidance=False)
    result = pipe.run()
    return pipe, result, load_persisted(result.report.persisted_path)


def test_the_persisted_folder_has_code_model_and_metadata(run, tmp_path):
    pipe, result, persisted = run
    root = persisted_root(tmp_path / "out", result.report.chosen["bundle_id"])
    assert persisted.root == root
    assert sorted(p.name for p in root.iterdir()) == ["code", "model", "persisted.json"]
    meta = json.loads((root / "persisted.json").read_text())
    chosen = result.report.chosen
    assert (meta["bundle_id"], meta["attempt_id"], meta["cycle_id"]) == (
        chosen["bundle_id"], chosen["attempt_id"], CYCLE)
    assert meta["folder_hash"] == folder_hash_of(root / "code") == pipe.registry.get_bundle(
        chosen["bundle_id"]).folder_hash
    assert meta["protocol_hash"] == result.report.protocol_hash
    assert meta["validation_mean"] == chosen["validation_mean"]
    assert (meta["target"], meta["metric"], meta["input_kind"]) == ("y", "accuracy", "labels")
    assert persisted.protocol == pipe.engine.protocol
    assert persisted.loaded().folder_hash == meta["folder_hash"]
    # fitted on the development rows only, with the seed the engine uses for the holdout
    split = FrozenSplit.create(FRAME, pipe.engine.protocol)
    assert pipe.store.fits[0]["rows"] == len(split.development_rows())
    assert pipe.store.fits[0]["seed"] == meta["seed"] == derive_seed(pipe.engine.protocol.seed, "candidate-holdout")
    assert (root / "model" / "model.json").is_file()


def test_a_changed_code_file_or_model_file_is_refused_with_its_name(run):
    _, _, persisted = run
    code, model = persisted.code_dir / "model.py", persisted.model_dir / "model.json"
    original_code, original_model = code.read_text(), model.read_text()
    code.write_text(original_code + "# tampered\n")
    with pytest.raises(PersistedBundleError, match=r"code/model\.py"):
        load_persisted(persisted.root)
    code.write_text(original_code)
    load_persisted(persisted.root)
    model.write_text('{"label": 7}')
    with pytest.raises(PersistedBundleError, match=r"model/model\.json"):
        load_persisted(persisted.root)
    model.write_text(original_model)
    (persisted.model_dir / "extra.bin").write_bytes(b"1")
    with pytest.raises(PersistedBundleError, match=r"extra\.bin is new"):
        load_persisted(persisted.root)
    os.remove(persisted.model_dir / "extra.bin")
    os.remove(model)
    with pytest.raises(PersistedBundleError, match=r"model\.json is missing"):
        load_persisted(persisted.root)


def test_missing_or_malformed_metadata_is_refused(run, tmp_path):
    _, _, persisted = run
    with pytest.raises(PersistedBundleError):
        load_persisted(tmp_path / "nowhere")
    (persisted.root / "persisted.json").write_text("{}")
    with pytest.raises(PersistedBundleError, match="incomplete"):
        load_persisted(persisted.root)
    (persisted.root / "persisted.json").write_text("not json")
    with pytest.raises(PersistedBundleError):
        load_persisted(persisted.root)


def test_a_second_persist_reuses_the_existing_copy_and_an_invalid_one_is_not_overwritten(run, tmp_path):
    pipe, result, persisted = run
    from kaggle_agent.bundle import load_bundle

    chosen = result.report.chosen
    loaded = load_bundle(pipe.registry.get_bundle(chosen["bundle_id"]).reference)
    stamp = (persisted.root / "persisted.json").stat().st_mtime_ns
    args = dict(attempt_id=chosen["attempt_id"], cycle_id=CYCLE, engine=pipe.engine, frame=FRAME,
                data_hash="data-hash", validation_mean=0.5, validation_dispersion=0.1)
    again = persist_bundle(tmp_path / "out", loaded, pipe.store, **args)
    assert again.meta == persisted.meta and len(pipe.store.fits) == 1
    assert (persisted.root / "persisted.json").stat().st_mtime_ns == stamp
    with pytest.raises(PersistedBundleError, match="different attempt_id"):
        persist_bundle(tmp_path / "out", loaded, pipe.store, **{**args, "attempt_id": "other"})
    (persisted.model_dir / "model.json").write_text("{}")
    with pytest.raises(PersistedBundleError, match="model/"):
        persist_bundle(tmp_path / "out", loaded, pipe.store, **args)
    assert len(pipe.store.fits) == 1
