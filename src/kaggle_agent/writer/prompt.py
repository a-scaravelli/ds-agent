"""The prompt of the writer: a pure function of the request, the task and the previous code.

No I/O. It renders the role outputs, the schema of the development data (never rows or labels) and
the bundle contract taken from the ``bundle`` constants.
"""

import hashlib
import json
from dataclasses import dataclass

from kaggle_agent.agents import DataSummary
from kaggle_agent.bundle import ENTRY_POINT_NAMES, MANIFEST_NAME, MODEL_DIR_ENV, MODEL_FOLDER
from kaggle_agent.loop.writer import BundleRequest

from .workspace import BUNDLE_SUBFOLDER


@dataclass(frozen=True)
class WriterTask:
    """What the bundle must predict and how it is scored; ``summary`` is the schema of the development data."""

    target: str
    metric: str
    direction: str
    input_kind: str
    summary: DataSummary


def prompt_hash(prompt: str) -> str:
    """SHA-256 of the full prompt text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


_KIND_TEXT = {
    "values": "one number per row",
    "labels": "one class label per row (a string or a number, one kind only)",
    "scores": "one number per row, or a matrix with one row per input row and at least 2 columns",
    "probabilities": (
        "one probability in [0, 1] per row, or a matrix with one row per input row, at least 2 "
        "columns, every row summing to 1"
    ),
}


def _schema(summary: DataSummary) -> str:
    lines = [f"- {c.name}: {c.dtype}, missing {c.missing_fraction:.1%}, {c.distinct_values} distinct values"
             for c in summary.columns]
    head = f"{summary.rows} development rows; target column: {summary.target}"
    notes = [f"Note: {n}" for n in summary.notes]
    return "\n".join([head, *lines, *notes])


def _attempt(title: str, attempt) -> str:
    return (
        f"{title}\n- hypothesis: {attempt.hypothesis}\n- rationale: {attempt.rationale}\n"
        f"- what changed: {attempt.what_changed}\n- how it is measured: {attempt.measure}"
    )


def _findings(findings) -> str:
    parts = [f"Profile: {findings.profile}"]
    for label, items in (("Quality issues", findings.quality_issues),
                         ("Leakage hints", findings.leakage_hints),
                         ("Proposed cleaning", findings.proposed_cleaning)):
        if items:
            parts.append(f"{label}:\n" + "\n".join(f"- {i}" for i in items))
    return "\n".join(parts)


def _previous(files: dict[str, str]) -> str:
    return "\n\n".join(f"--- {name} ---\n{text}" for name, text in files.items())


def build_prompt(request: BundleRequest, task: WriterTask, allowed_dependencies: tuple[str, ...],
                 previous: dict[str, str] | None = None) -> str:
    """The first prompt of a write; ``previous`` holds the old bundle code of a fix request."""
    kind = _KIND_TEXT.get(task.input_kind, task.input_kind)
    manifest = {
        "bundle_id": request.bundle_id,
        "dependencies": ["<only the third-party libraries you import>"],
        "entry_points": {name: f"model:{name}" for name in ENTRY_POINT_NAMES},
        "description": "<optional text>",
    }
    outputs = request.outputs
    sections = [
        "You are the bundle writer of a tabular data science agent. You write the code of one model "
        f"bundle as files in the folder ./{BUNDLE_SUBFOLDER} of your working directory, using your file "
        "tools. Write only inside that folder.",
        "No tool runs code: you cannot test the bundle, so write it carefully and keep it simple.",
        f"Bundle id: {request.bundle_id}",
        "## Contract\n"
        f"- {MANIFEST_NAME} (exactly this id; no other keys):\n{json.dumps(manifest, indent=2)}\n"
        "- Entry points are 'dotted.module:function' names inside the bundle folder.\n"
        "- fit(train): train is a pandas.DataFrame with the features and the target column "
        f"'{task.target}'. Train the model and save it under the directory named by the environment "
        f"variable {MODEL_DIR_ENV} (read it with os.environ[{MODEL_DIR_ENV!r}]). fit and predict run in "
        "different processes: nothing kept in memory survives, so predict must load the saved model "
        f"from {MODEL_DIR_ENV}. Do not write a {MODEL_FOLDER}/ folder in the bundle.\n"
        "- predict(bundle_id, X): X is a pandas.DataFrame with the same feature columns and no target "
        f"column. Return {kind}, in row order. Predictions cross the process boundary as JSON only: "
        "return a list, a 1-D array, or a probability matrix (list of rows or 2-D array); no pickle "
        "and no other types. Never report your own score.\n"
        "- Allowed imports: the Python standard library and "
        f"{', '.join(allowed_dependencies)}. List every third-party library you import in "
        "dependencies. Import nothing else, use no network and no subprocess.\n"
        "- Only the bundle folder is checked: do not read or write other files.",
        "## Task\n"
        f"Target column: {task.target}. Metric: {task.metric} ({task.direction}). "
        f"Prediction kind: {task.input_kind}.",
        f"## Development data schema (no rows, no labels)\n{_schema(task.summary)}",
        "## What to build\n" + _attempt("Feature engineering attempt:", outputs.feature_engineering)
        + "\n\n" + _attempt("Modeling attempt:", outputs.modeling),
        f"## Data findings\n{_findings(outputs.findings)}",
    ]
    if request.fix is not None:
        sections.append(
            "## Fix request\n"
            f"The previous bundle {request.fix.previous_bundle_id} failed (retry {request.fix.retry}).\n"
            f"Failure reason: {request.fix.reason}\n"
            f"Write a corrected bundle under the new id {request.bundle_id}. The previous code was:\n\n"
            f"{_previous(previous or {})}"
        )
    sections.append(
        f"When done, reply with a short JSON object like {{\"summary\": \"what you wrote\"}}."
    )
    return "\n\n".join(sections)


def repair_prompt(base: str, error: str) -> str:
    """The base prompt plus the exact check error; the files written so far are still in ./bundle."""
    return (
        f"{base}\n\n## Repair\n"
        f"You already wrote files in ./{BUNDLE_SUBFOLDER}, but the bundle check failed with this "
        f"exact error:\n{error}\n"
        f"Read the files in ./{BUNDLE_SUBFOLDER}, fix them in place so the check passes, and reply "
        "again with the short JSON summary."
    )
