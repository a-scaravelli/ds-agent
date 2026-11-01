"""Child side of the bundle runner: fixed code, run as ``python -I -B child.py STAGING STAGE``.

Standard library plus pandas only; it does not import ``kaggle_agent``. ``STAGING`` holds
``request.json``, ``dtypes.json`` and ``bundle/`` (a copy of the bundle folder). ``STAGE`` is
one of two, each run in its own process:

- ``fit``: reads ``train.csv`` (features and target of the training rows), calls ``fit`` and
  leaves whatever the bundle saved under ``BUNDLE_MODEL_DIR`` (``STAGING/bundle/model``);
- ``predict``: reads ``x.csv`` (features of the rows to predict, no target), calls
  ``predict`` (which loads the model from ``BUNDLE_MODEL_DIR``) and writes ``predictions.json``.

Any exception ends the process with a non-zero exit code, which the sandbox reports as
``crashed``.

``-I`` ignores ``PYTHONPATH``, so the copy of the bundle is added to ``sys.path`` here.
"""

import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_frame(staging, name, dtypes):
    frame = pd.read_csv(staging / name)
    for column, dtype in dtypes.items():
        if column in frame.columns:
            try:
                frame[column] = frame[column].astype(dtype)
            except (TypeError, ValueError):
                pass  # keep what the CSV reader inferred
    return frame


def as_list(result):
    if isinstance(result, pd.DataFrame):
        return result.to_numpy().tolist()
    if isinstance(result, (str, bytes)) or result is None:
        return result
    if hasattr(result, "tolist"):
        return result.tolist()
    return list(result)


def main(argv):
    staging = Path(argv[1])
    stage = argv[2]
    if stage not in ("fit", "predict"):
        raise SystemExit(f"unknown stage {stage!r}")
    request = json.loads((staging / "request.json").read_text(encoding="utf-8"))
    dtypes = json.loads((staging / "dtypes.json").read_text(encoding="utf-8"))
    model_dir = staging / "bundle" / "model"
    os.environ["BUNDLE_MODEL_DIR"] = str(model_dir)
    sys.path.insert(0, str(staging / "bundle"))
    random.seed(request["seed"])
    np.random.seed(request["seed"] % 2**32)

    def load(kind):
        module, function = request["entry_points"][kind].split(":")
        return getattr(importlib.import_module(module), function)

    if stage == "fit":
        model_dir.mkdir(parents=True, exist_ok=True)
        load("fit")(read_frame(staging, "train.csv", dtypes["train"]))
    else:
        x = read_frame(staging, "x.csv", dtypes["x"])
        result = as_list(load("predict")(request["bundle_id"], x))
        (staging / "predictions.json").write_text(json.dumps(result, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
