"""The bundle contract: a small interface a bundle folder must satisfy (design section 8).

Layout of a bundle folder::

    my_bundle/
        bundle.json      manifest (below)
        model.py         defines fit and predict (any module path inside the folder)
        helpers.py       any other code of the bundle

``bundle.json``::

    {
      "bundle_id": "B1",
      "dependencies": ["pandas", "scikit-learn"],
      "entry_points": {"fit": "model:fit", "predict": "model:predict"},
      "description": "optional text"
    }

Entry points are ``dotted.module:function`` names relative to the bundle folder.

The two functions:

- ``fit(train)``: ``train`` is a ``pandas.DataFrame`` with the features and the target
  column, rows of the training partition only. The return value is ignored. The trained
  model is saved under the directory named by the environment variable ``BUNDLE_MODEL_DIR``
  (any file format of the bundle's choice): ``fit`` and ``predict`` run in different
  processes, so nothing kept in memory survives between them.
- ``predict(bundle_id, X)``: ``X`` is a ``pandas.DataFrame`` with the same feature
  columns and no target column. It loads the model from ``BUNDLE_MODEL_DIR`` and returns
  one prediction per row, in order: numbers for ``values``, ``scores`` and ``probabilities``
  (the latter within [0, 1]), any scalar label for ``labels``. For ``probabilities`` and
  ``scores`` it may also return a matrix (a list of rows, or a 2-D array or DataFrame) with
  one row per input row, the same width of at least 2 in every row, and for
  ``probabilities`` each row summing to 1 (multiclass).

The child seeds ``random`` and ``numpy.random`` with the seed derived by the evaluation
engine in both processes. A bundle never reports its own score: the parent computes every
metric. Predictions cross the process boundary as JSON only, never pickle; a bundle may use
any format, pickle included, for its own files under ``BUNDLE_MODEL_DIR``, because only the
bundle code in the child reads them. Declared dependencies are checked against the imports
of the code but are not installed (level 2).
"""

import re
from dataclasses import dataclass, field
from typing import Any

from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import check_keys, require_text, require_text_list

MANIFEST_NAME = "bundle.json"
ENTRY_POINT_NAMES = ("fit", "predict")

# Statuses the runner adds to the sandbox ones (see ``BundleRunError``).
MALFORMED_PREDICTIONS = "malformed_predictions"
FOLDER_CHANGED = "folder_changed"
MODEL_DIR_ENV = "BUNDLE_MODEL_DIR"
MODEL_FOLDER = "model"  # top-level folder of a bundle that holds the persisted model

_MANIFEST_FIELDS = ("bundle_id", "dependencies", "entry_points", "description")
_ENTRY_POINT = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):([A-Za-z_]\w*)$")


def parse_entry_point(text: str) -> tuple[str, str]:
    """Split ``module.path:function`` into its two parts, or raise ``ValidationError``."""
    match = _ENTRY_POINT.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValidationError(
            "entry_points",
            f"{text!r} must be 'dotted.module:function' inside the bundle folder "
            "(no absolute path, no '..', no slash)",
        )
    return match.group(1), match.group(2)


@dataclass(frozen=True)
class Manifest:
    """The content of ``bundle.json``."""

    bundle_id: str
    dependencies: tuple[str, ...]
    entry_points: dict[str, str]
    description: str = field(default="")

    def __post_init__(self) -> None:
        require_text("bundle_id", self.bundle_id)
        object.__setattr__(
            self, "dependencies", require_text_list("dependencies", self.dependencies, allow_empty=True)
        )
        if not isinstance(self.entry_points, dict):
            raise ValidationError("entry_points", "must be an object")
        for name in ENTRY_POINT_NAMES:
            if name not in self.entry_points:
                raise ValidationError("entry_points", f"missing the {name!r} entry point")
        extra = sorted(set(self.entry_points) - set(ENTRY_POINT_NAMES))
        if extra:
            raise ValidationError("entry_points", f"unknown entry point {extra[0]!r}")
        for value in self.entry_points.values():
            parse_entry_point(value)
        object.__setattr__(self, "entry_points", dict(self.entry_points))
        if not isinstance(self.description, str):
            raise ValidationError("description", "must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "dependencies": list(self.dependencies),
            "entry_points": dict(self.entry_points),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        check_keys("manifest", data, _MANIFEST_FIELDS)
        return cls(
            bundle_id=data.get("bundle_id"),
            dependencies=data.get("dependencies", []),
            entry_points=data.get("entry_points"),
            description=data.get("description", ""),
        )
