"""A fake ``ModelStore`` for tests: no process, no pickle, a small JSON model file."""

import json
from pathlib import Path

from kaggle_agent.bundle import validate_predictions

MODEL_FILE = "model.json"


class FakeModelStore:
    """``fit_model`` writes the most frequent training label to a JSON file, ``predict_with_model`` reads it.

    ``predict_failure`` scripts the predict stage: a ``BundleRunError`` (or any exception)
    instance is raised as it is, which plays a crash; any other value is a raw predictions
    payload passed through the real ``validate_predictions``, so a malformed payload raises
    the genuine ``BundleRunError(MALFORMED_PREDICTIONS, ...)``. ``fit_failure`` does the same
    for the fit stage. Calls are recorded in ``fits`` and ``predicts``.
    """

    def __init__(self, input_kind: str = "labels", predict_failure=None, fit_failure=None) -> None:
        self.input_kind = input_kind
        self.predict_failure = predict_failure
        self.fit_failure = fit_failure
        self.fits: list[dict] = []
        self.predicts: list[dict] = []

    def fit_model(self, x_train, y_train, seed, model_dir) -> None:
        if isinstance(self.fit_failure, BaseException):
            raise self.fit_failure
        self.fits.append({"rows": len(x_train), "seed": seed, "columns": list(x_train.columns)})
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        label = y_train.mode().iloc[0]
        (model_dir / MODEL_FILE).write_text(json.dumps({"label": int(label)}), encoding="utf-8")

    def predict_with_model(self, model_dir, x, seed):
        self.predicts.append({"rows": len(x), "seed": seed, "columns": list(x.columns),
                              "model_dir": str(model_dir)})
        if isinstance(self.predict_failure, BaseException):
            raise self.predict_failure
        if self.predict_failure is not None:
            return validate_predictions(self.predict_failure, len(x), self.input_kind)
        label = json.loads((Path(model_dir) / MODEL_FILE).read_text(encoding="utf-8"))["label"]
        return [label] * len(x)
