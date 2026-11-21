"""Plain-words texts and the 'not recorded' object shared by the analytics and the cockpit pages."""

REASONS = {
    "run_failed": "the run of this attempt failed, so no predictions were recorded",
    "no_bundle": "this attempt has no succeeded execution with a bundle",
    "older_project": "this project was run before analytics were recorded (no analytics folder)",
    "not_in_index": "nothing was recorded for the bundle of this attempt (older project or a recording problem)",
    "too_large": "the data was over the size limit of the analytics files",
    "budget_exhausted": "the analytics folder of the project reached its size budget",
    "recording_error": "the recorder could not write the file for this attempt",
    "not_readable": "the recorded file could not be read safely",
    "describe_missing": "the bundle did not provide a describe.json",
    "describe_invalid": "the bundle provided a describe.json that is not valid",
    "label_only": "the metric of this run scores labels only, so there are no probabilities or scores",
    "no_snapshot": "the development features were not recorded for this project",
    "not_comparable": "the two attempts have no rows in common",
    "no_scores": "no per-partition scores were recorded",
    "one_class": "the data has one class only",
    "too_many_classes": "this panel is for binary targets or up to 10 classes only",
    "no_importance": "permutation importance was not computed for this attempt",
}


def not_recorded(code: str, detail: str | None = None) -> dict:
    """The object every section returns instead of data: a code, plain words and an optional detail."""
    text = REASONS.get(code, code)
    return {"not_recorded": True, "code": code, "reason": text + (f" ({detail})" if detail else "")}


def is_not_recorded(value) -> bool:
    return isinstance(value, dict) and value.get("not_recorded") is True


HEURISTIC = "This is a heuristic, not proof."
