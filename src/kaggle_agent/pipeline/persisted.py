"""The persisted bundle: code, trained model and metadata of the chosen bundle (design section 8).

Layout of ``<out>/persisted/<bundle_id>/``::

    code/            a copy of the bundle folder (without any ``model/`` folder)
    model/           what ``fit`` saved under ``BUNDLE_MODEL_DIR``, fitted on the development data only
    persisted.json   ids, hashes of the code and of the model, seed, protocol, data hash, metric

The model is fitted before the holdout is touched (``FrozenSplit.development_data`` never
returns holdout rows), with the seed the engine uses for the holdout. The parent never reads
a model file as data: it only hashes the folder, and only the bundle code in a child process
loads it. ``load_persisted`` refuses a folder whose code or model differs from the hashes in
``persisted.json``.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol as Structural

import pandas as pd

from kaggle_agent.bundle import MODEL_FOLDER, BundleRunner, LoadedBundle, folder_hash_of, load_bundle
from kaggle_agent.evaluation import EvaluationEngine, FrozenSplit, Protocol, derive_seed, get_metric
from kaggle_agent.sandbox import Sandbox, SandboxPolicy
from kaggle_agent.sandbox.protected import hash_path

from .errors import PersistedBundleError

PERSISTED_NAME = "persisted"
CODE_FOLDER = "code"
META_FILE = "persisted.json"
HOLDOUT_SEED_NAME = "candidate-holdout"


class ModelStore(Structural):
    """What the pipeline needs from a bundle runner: fit and save, or load and predict.

    ``BundleRunner`` satisfies it; tests use ``pipeline.fake.FakeModelStore``.
    """

    def fit_model(self, x_train: pd.DataFrame, y_train: pd.Series, seed: int, model_dir: Path) -> None:
        """Fit on the given rows and copy the saved model to ``model_dir``."""

    def predict_with_model(self, model_dir: Path, x: pd.DataFrame, seed: int):
        """Predict ``x`` with the model stored in ``model_dir``."""


def runner_model_store_factory(sandbox: Sandbox, policy: SandboxPolicy, *, target_column: str,
                               input_kind: str):
    """A ``model_store_factory`` that builds a real ``BundleRunner`` for a loaded bundle."""
    def factory(loaded: LoadedBundle) -> ModelStore:
        return BundleRunner(sandbox, policy, loaded, target_column=target_column, input_kind=input_kind)

    return factory


def copy_code(source: Path, target: Path) -> None:
    """Copy a bundle folder, leaving out ``__pycache__`` and a top-level ``model/`` folder."""
    source = Path(source)

    def ignore(directory, names):
        skipped = [n for n in names if n == "__pycache__"]
        if Path(directory) == source and (source / MODEL_FOLDER).is_dir():
            skipped.append(MODEL_FOLDER)
        return skipped

    shutil.copytree(source, target, ignore=ignore)


def _file_hashes(folder: Path) -> dict[str, str]:
    """SHA-256 per file of a folder (relative POSIX path), ``__pycache__`` left out."""
    out = {}
    for path in sorted(Path(folder).rglob("*")):
        relative = path.relative_to(folder)
        if path.is_file() and "__pycache__" not in relative.parts:
            out[relative.as_posix()] = hash_path(path)
    return out


def _changed(recorded: dict, current: dict) -> str | None:
    """Name of the first file that differs, appears or disappears; None if equal."""
    for name in sorted(set(recorded) | set(current)):
        if recorded.get(name) != current.get(name):
            what = "is missing" if name not in current else "is new" if name not in recorded else "changed"
            return f"{name} {what}"
    return None


@dataclass(frozen=True)
class PersistedBundle:
    """A persisted bundle folder and the metadata read from ``persisted.json``."""

    root: Path
    meta: dict

    @property
    def code_dir(self) -> Path:
        return self.root / CODE_FOLDER

    @property
    def model_dir(self) -> Path:
        return self.root / MODEL_FOLDER

    @property
    def bundle_id(self) -> str:
        return self.meta["bundle_id"]

    @property
    def protocol(self) -> Protocol:
        return Protocol.from_dict(self.meta["protocol"])

    @property
    def seed(self) -> int:
        return self.meta["seed"]

    def loaded(self) -> LoadedBundle:
        """The bundle code of this folder, validated by ``load_bundle``."""
        return load_bundle(self.code_dir)


def load_persisted(root: Path | str) -> PersistedBundle:
    """Load a persisted bundle and verify the hashes of its code and of its model.

    Raises ``PersistedBundleError`` naming the changed part (``code/<file>`` or ``model/<file>``).
    """
    root = Path(root).resolve()
    meta_path = root / META_FILE
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PersistedBundleError(f"{root} is not a persisted bundle: {error}") from None
    if not isinstance(meta, dict) or not all(
        key in meta for key in ("bundle_id", "folder_hash", "model_hash", "code_files", "model_files",
                                "protocol", "seed")
    ):
        raise PersistedBundleError(f"{META_FILE} of {root} is incomplete")
    persisted = PersistedBundle(root, meta)
    for part, folder, files, aggregate in (
        (CODE_FOLDER, persisted.code_dir, "code_files", "folder_hash"),
        (MODEL_FOLDER, persisted.model_dir, "model_files", "model_hash"),
    ):
        if not folder.is_dir():
            raise PersistedBundleError(f"{part}/ is missing in {root}")
        current = folder_hash_of(folder) if part == CODE_FOLDER else hash_path(folder)
        if current != meta[aggregate]:
            detail = _changed(meta[files], _file_hashes(folder)) or "content changed"
            raise PersistedBundleError(f"the persisted {part} changed: {part}/{detail}")
    return persisted


def persisted_root(out_dir: Path | str, bundle_id: str) -> Path:
    """Where the persisted copy of ``bundle_id`` lives under the output folder."""
    return Path(out_dir).resolve() / PERSISTED_NAME / bundle_id


def persist_bundle(out_dir: Path | str, loaded: LoadedBundle, store: ModelStore, *, attempt_id: str,
                   cycle_id: str, engine: EvaluationEngine, frame: pd.DataFrame, data_hash: str,
                   validation_mean: float, validation_dispersion: float) -> PersistedBundle:
    """Fit ``loaded`` on the development data and write its persisted folder; reuse a valid one.

    The protocol is verified first (a changed protocol blocks the run). Only development rows are
    used, so this never touches the holdout. If the folder already exists and is valid for the
    same attempt, protocol and data, it is returned as it is and nothing is fitted again; an
    existing folder that is invalid or belongs to something else raises
    ``PersistedBundleError``. The folder appears atomically: it is built in a temporary sibling
    and renamed.
    """
    engine.verify()
    protocol = engine.protocol
    protocol_hash = engine.registry.get_cycle(cycle_id).protocol_hash
    target = persisted_root(out_dir, loaded.bundle_id)
    if target.exists():
        existing = load_persisted(target)
        wanted = {"attempt_id": attempt_id, "protocol_hash": protocol_hash, "data_hash": data_hash,
                  "folder_hash": loaded.folder_hash}
        for key, value in wanted.items():
            if existing.meta.get(key) != value:
                raise PersistedBundleError(f"the persisted copy in {target} has a different {key}")
        return existing
    split = FrozenSplit.create(frame, protocol)
    x_dev, y_dev = split.development_data()
    seed = derive_seed(protocol.seed, HOLDOUT_SEED_NAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{loaded.bundle_id}-", dir=target.parent))
    try:
        copy_code(loaded.folder, work / CODE_FOLDER)
        if folder_hash_of(work / CODE_FOLDER) != loaded.folder_hash:
            raise PersistedBundleError("the copy of the bundle code differs from the registered folder hash")
        store.fit_model(x_dev, y_dev, seed, work / MODEL_FOLDER)
        (work / MODEL_FOLDER).mkdir(exist_ok=True)
        meta = {
            "bundle_id": loaded.bundle_id,
            "attempt_id": attempt_id,
            "cycle_id": cycle_id,
            "folder_hash": loaded.folder_hash,
            "model_hash": hash_path(work / MODEL_FOLDER),
            "code_files": _file_hashes(work / CODE_FOLDER),
            "model_files": _file_hashes(work / MODEL_FOLDER),
            "seed": seed,
            "protocol": protocol.to_dict(),
            "protocol_hash": protocol_hash,
            "data_hash": data_hash,
            "target": protocol.target,
            "metric": protocol.metric,
            "input_kind": get_metric(protocol.metric).input_kind,
            "validation_mean": validation_mean,
            "validation_dispersion": validation_dispersion,
        }
        (work / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        work.rename(target)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return load_persisted(target)


__all__ = [
    "CODE_FOLDER", "META_FILE", "ModelStore", "PERSISTED_NAME", "PersistedBundle",
    "copy_code", "load_persisted", "persist_bundle", "persisted_root", "runner_model_store_factory",
]
