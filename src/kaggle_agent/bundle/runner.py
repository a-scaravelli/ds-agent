"""The bundle runner: fixed code that runs a bundle through the sandbox (design section 8).

``BundleRunner`` implements ``evaluation.Candidate``, so the frozen evaluation engine
scores a bundle without changes. ``fit_predict`` runs two stages in two separate child
processes: ``fit`` saves the model under ``BUNDLE_MODEL_DIR`` and ``predict`` loads it, so
every reported metric goes through persisted state. ``fit_model`` and ``predict_with_model``
expose the two stages: the first copies the model folder out of the staging folder, the second
runs ``predict`` on a stored copy. Each stage stages the data in a temporary folder, runs
``child.py`` through the injected ``Sandbox`` (the only way a process is started), and removes
the folder. The parent never opens a file of the model folder.

What the child receives: the training rows with their target and the rows to predict
without it. The validation and holdout labels stay with the engine and are never staged.
The runner protects its own package and the evaluation package, and passes the hashes
taken when the runner was built as ``expected_hashes``: a protected file changed since
then blocks the run before the child starts. Data goes through CSV and predictions come
back as JSON (never pickle), so the bundle cannot run code in the parent.
"""

import dataclasses
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import kaggle_agent
from kaggle_agent import evaluation
from kaggle_agent.evaluation.metrics import INPUT_KINDS
from kaggle_agent.sandbox import (
    SUCCEEDED,
    ProtectedPath,
    Sandbox,
    SandboxPolicy,
    SandboxRequest,
    hash_paths,
)

from .contract import FOLDER_CHANGED, MALFORMED_PREDICTIONS, MODEL_FOLDER
from .errors import BundleError, BundleRunError
from .layout import LoadedBundle, folder_hash_of

BUNDLE_DIR = Path(__file__).resolve().parent
CHILD_SCRIPT = BUNDLE_DIR / "child.py"
CODE_ERROR = "code_error"
NUMERIC_KINDS = ("values", "scores", "probabilities")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_matrix(raw: list, input_kind: str, bad) -> np.ndarray:
    if input_kind not in ("probabilities", "scores"):
        raise bad(f"{input_kind} must be one-dimensional, not a matrix")
    if not all(isinstance(row, list) for row in raw):
        raise bad("rows and single values are mixed")
    width = len(raw[0])
    if width < 2:
        raise bad("a matrix row needs at least 2 values")
    if any(len(row) != width for row in raw):
        raise bad("rows of different width")
    for row in raw:
        for cell in row:
            if cell is None:
                raise bad("null value")
            if not _is_number(cell):
                raise bad(f"{input_kind} must be numbers")
            if isinstance(cell, float) and not math.isfinite(cell):
                raise bad("NaN or infinite value")
    values = np.asarray(raw, dtype=float)
    if input_kind == "probabilities":
        if (values < 0).any() or (values > 1).any():
            raise bad("probabilities must be within [0, 1]")
        if (np.abs(values.sum(axis=1) - 1.0) > 1e-6).any():
            raise bad("each row of probabilities must sum to 1")
    return values


def validate_predictions(raw, rows: int, input_kind: str) -> np.ndarray:
    """Check the raw predictions of a bundle and return them as an array.

    ``rows`` predictions, none null. ``values``, ``scores`` and ``probabilities`` need finite
    numbers, ``probabilities`` within [0, 1]; ``labels`` accept any scalar labels of one kind
    (text or numbers). ``probabilities`` and ``scores`` may also be a rectangular matrix, one
    row per input row with at least 2 finite numbers each (for ``probabilities`` within [0, 1]
    and summing to 1 within 1e-6). Raises ``BundleRunError`` otherwise.
    """
    def bad(reason):
        return BundleRunError(MALFORMED_PREDICTIONS, CODE_ERROR, "malformed predictions: " + reason)

    if not isinstance(raw, list):
        raise bad("expected a list")
    if len(raw) != rows:
        raise bad(f"expected {rows} predictions, got {len(raw)}")
    if any(v is None for v in raw):
        raise bad("null value")
    if any(isinstance(v, dict) for v in raw):
        raise bad("expected numbers or rows of numbers")
    if any(isinstance(v, list) for v in raw):
        return _validate_matrix(raw, input_kind, bad)
    if input_kind in NUMERIC_KINDS:
        if not all(_is_number(v) for v in raw):
            raise bad(f"{input_kind} must be numbers")
    elif not all(_is_number(v) or isinstance(v, (str, bool)) for v in raw):
        raise bad("labels must be scalars")
    elif any(isinstance(v, str) for v in raw) and not all(isinstance(v, str) for v in raw):
        raise bad("labels mix text and numbers")
    if any(isinstance(v, float) and not math.isfinite(v) for v in raw):
        raise bad("NaN or infinite value")
    values = np.asarray(raw)
    if input_kind == "probabilities" and ((values < 0).any() or (values > 1).any()):
        raise bad("probabilities must be within [0, 1]")
    return values


def _dtypes(frame: pd.DataFrame) -> dict:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


class BundleRunner:
    """Runs ``fit`` and ``predict`` of one bundle in a child process; a ``Candidate``.

    ``policy`` carries the timeout (and any protected paths) of the caller; the runner adds
    its own two packages. A protected name that the caller already uses with the same path
    is kept once; with a different path ``BundleError`` is raised. ``input_kind`` is the
    prediction kind of the metric (``INPUT_KINDS``).
    """

    def __init__(self, sandbox: Sandbox, policy: SandboxPolicy, bundle: LoadedBundle, *,
                 target_column: str, input_kind: str) -> None:
        if input_kind not in INPUT_KINDS:
            raise BundleError(f"unknown input_kind {input_kind!r}")
        self.sandbox = sandbox
        self.bundle = bundle
        self.target_column = target_column
        self.input_kind = input_kind
        self.policy = self._extend(policy)
        self.baseline = hash_paths(self.policy.protected)

    @staticmethod
    def _own_paths() -> tuple[ProtectedPath, ...]:
        evaluation_dir = Path(evaluation.__file__).resolve().parent
        return tuple(
            ProtectedPath(f"{kaggle_agent.__name__}/{name}", path)
            for name, path in (("bundle", BUNDLE_DIR), ("evaluation", evaluation_dir))
        )

    def _extend(self, policy: SandboxPolicy) -> SandboxPolicy:
        existing = {p.name: p for p in policy.protected}
        added = []
        for own in self._own_paths():
            current = existing.get(own.name)
            if current is None:
                added.append(own)
            elif current.path.resolve() != own.path:
                raise BundleError(
                    f"protected name {own.name!r} is already used for a different path: {current.path}"
                )
        return dataclasses.replace(policy, protected=(*policy.protected, *added))

    def _check_folder(self) -> None:
        current = folder_hash_of(self.bundle.folder)
        if current != self.bundle.folder_hash:
            raise BundleRunError(
                FOLDER_CHANGED, CODE_ERROR,
                f"the bundle folder hash changed since it was registered ({current} != {self.bundle.folder_hash})",
            )

    def fit_predict(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame,
                    seed: int) -> np.ndarray:
        """Fit on the training rows, then predict ``x_valid`` in a second process.

        The model goes through a temporary model folder between the two child processes; the
        labels of ``x_valid`` are not known here.
        """
        model_dir = Path(tempfile.mkdtemp(prefix="bundle-model-")).resolve()
        try:
            self.fit_model(x_train, y_train, seed, model_dir)
            return self.predict_with_model(model_dir, x_valid, seed)
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)

    def fit_model(self, x_train: pd.DataFrame, y_train: pd.Series, seed: int, model_dir: Path) -> None:
        """Run the ``fit`` stage in one sandbox call and copy the saved ``model/`` to ``model_dir``.

        An existing ``model_dir`` is replaced; a bundle that saves nothing leaves it empty.
        """
        self._check_folder()
        staging = Path(tempfile.mkdtemp(prefix="bundle-run-")).resolve()
        try:
            train = x_train.reset_index(drop=True)
            train[self.target_column] = np.asarray(y_train)
            train.to_csv(staging / "train.csv", index=False)
            self._stage_common(staging, {"train": _dtypes(train)}, seed)
            self._run_stage(staging, "fit")
            model_dir = Path(model_dir)
            if model_dir.exists():
                shutil.rmtree(model_dir)
            saved = staging / "bundle" / MODEL_FOLDER
            if saved.is_dir():
                shutil.copytree(saved, model_dir)
            else:
                model_dir.mkdir(parents=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def predict_with_model(self, model_dir: Path, x: pd.DataFrame, seed: int) -> np.ndarray:
        """Run the ``predict`` stage in a fresh sandbox call on a copy of ``model_dir``."""
        self._check_folder()
        staging = Path(tempfile.mkdtemp(prefix="bundle-run-")).resolve()
        try:
            features = x.drop(columns=[self.target_column], errors="ignore").reset_index(drop=True)
            features.to_csv(staging / "x.csv", index=False)
            self._stage_common(staging, {"x": _dtypes(features)}, seed)
            shutil.copytree(model_dir, staging / "bundle" / MODEL_FOLDER)
            result = self._run_stage(staging, "predict")
            try:
                raw = json.loads((staging / "predictions.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise BundleRunError(
                    MALFORMED_PREDICTIONS, CODE_ERROR, "malformed predictions: no readable predictions.json", result
                ) from None
            try:
                return validate_predictions(raw, len(features), self.input_kind)
            except BundleRunError as error:
                error.result = result
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _stage_common(self, staging: Path, dtypes: dict, seed: int) -> None:
        (staging / "dtypes.json").write_text(json.dumps(dtypes), encoding="utf-8")
        (staging / "request.json").write_text(json.dumps({
            "bundle_id": self.bundle.bundle_id,
            "entry_points": self.bundle.manifest.entry_points,
            "seed": int(seed),
            "target_column": self.target_column,
            "input_kind": self.input_kind,
        }), encoding="utf-8")
        source = self.bundle.folder

        def ignore(directory, names):
            skipped = [n for n in names if n == "__pycache__"]
            if Path(directory) == source and (source / MODEL_FOLDER).is_dir():
                skipped.append(MODEL_FOLDER)
            return skipped

        shutil.copytree(source, staging / "bundle", ignore=ignore)
        if folder_hash_of(staging / "bundle") != self.bundle.folder_hash:
            raise BundleRunError(FOLDER_CHANGED, CODE_ERROR, "the staged copy differs from the registered folder hash")

    def _run_stage(self, staging: Path, stage: str):
        request = SandboxRequest(
            entry_point=CHILD_SCRIPT,
            args=(str(staging), stage),
            expected_hashes=self.baseline,
        )
        result = self.sandbox.run(self.policy, request)
        if result.status != SUCCEEDED:
            raise BundleRunError(result.status, result.failure_kind, result.reason, result)
        return result


class PersistedModelCandidate:
    """A ``Candidate`` that ignores the training data and predicts from a stored model.

    Used for the holdout: the model was fitted earlier on the development data only, and
    ``fit_predict`` just runs the ``predict`` stage on it.
    """

    def __init__(self, runner: "BundleRunner", model_dir: Path) -> None:
        self.runner = runner
        self.model_dir = Path(model_dir)

    def fit_predict(self, x_train, y_train, x_valid, seed: int) -> np.ndarray:
        """Predict ``x_valid`` with the stored model; ``x_train`` and ``y_train`` are not used."""
        return self.runner.predict_with_model(self.model_dir, x_valid, seed)
