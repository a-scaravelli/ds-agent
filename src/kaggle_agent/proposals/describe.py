"""The optional ``describe.json`` of a bundle: one source of truth for its name, fields and limits.

A bundle may write ``describe.json`` next to its model (in ``BUNDLE_MODEL_DIR``) to say which features it
uses, how derived features are built, its hyperparameters and its own importances. The file is optional:
a missing one is allowed and the cockpit says "not provided". The writer prompt, the role prompts, the
bundle contract and the platform validator (``analytics.describe``) all read the constants of this module,
so they cannot drift. Stdlib only.
"""

DESCRIBE_FILE = "describe.json"
DESCRIBE_VERSION = 1
MAX_DESCRIBE_BYTES = 64 * 1024
MAX_STRING = 200
MAX_FEATURES = 500
MAX_SOURCES = 20
MAX_HYPERPARAMETERS = 100
MAX_NOTES = 1000

DESCRIBE_FIELDS = ("features_used", "derived_features", "hyperparameters", "native_importances",
                   "model_family", "notes")
DERIVED_FIELDS = ("name", "sources", "description")


def describe_paragraph() -> str:
    """The text the writer and the role prompts carry, generated from the constants above."""
    return (
        f"Optional but welcome: next to the saved model, fit may also write {DESCRIBE_FILE} (a JSON object, "
        f"at most {MAX_DESCRIBE_BYTES // 1024} KB) so the data scientist can see what the model uses. "
        f"Allowed keys: {', '.join(DESCRIBE_FIELDS)}. features_used: list of the input column names the "
        f"model uses (at most {MAX_FEATURES}). derived_features: list of objects "
        f"{{{', '.join(DERIVED_FIELDS)}}} where sources lists the input columns a new feature is built "
        "from. hyperparameters: an object of simple values (numbers, text, booleans). native_importances: "
        "an object that maps a feature name to a number (the importance the model itself reports). "
        f"model_family: a short text. notes: a short text. Every text is at most {MAX_STRING} characters. "
        "Never put secrets, keys or file paths in it. The platform validates the file and a missing or "
        "invalid file never fails the run."
    )
