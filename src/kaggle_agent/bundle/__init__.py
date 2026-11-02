"""Bundle contract and runner (design sections 8 and 15, step 7, level 1).

A bundle is a folder with code, declared dependencies and two entry points,
``fit(train)`` and ``predict(bundle_id, X)``. The fixed-code runner executes it in a
child process through ``kaggle_agent.sandbox`` and returns validated predictions; the
parent scores them with the frozen evaluation engine. Level 2 and 3 features
(immutable bundles, virtual environments, containers) are not built.
"""

from .contract import (
    ENTRY_POINT_NAMES,
    FOLDER_CHANGED,
    MALFORMED_PREDICTIONS,
    MANIFEST_NAME,
    MODEL_DIR_ENV,
    MODEL_FOLDER,
    Manifest,
)
from .errors import BundleError, BundleLayoutError, BundleRunError
from .layout import LoadedBundle, folder_hash_of, load_bundle, schema_of
from .runner import BundleRunner, PersistedModelCandidate, validate_predictions
from .scoring import BundleScore, attempt_with_bundle, register_bundle, score_bundle

__all__ = [
    "BundleError",
    "BundleLayoutError",
    "BundleRunError",
    "BundleRunner",
    "BundleScore",
    "ENTRY_POINT_NAMES",
    "FOLDER_CHANGED",
    "LoadedBundle",
    "MALFORMED_PREDICTIONS",
    "MANIFEST_NAME",
    "MODEL_DIR_ENV",
    "MODEL_FOLDER",
    "Manifest",
    "PersistedModelCandidate",
    "attempt_with_bundle",
    "folder_hash_of",
    "load_bundle",
    "register_bundle",
    "schema_of",
    "score_bundle",
    "validate_predictions",
]
