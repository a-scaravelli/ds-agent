"""Helpers of the analytics tests: small tables with hand-computable answers and a finished fake run.

Nothing here touches the network, a real Claude or a real Kaggle. The flow helpers use the real orchestrator
flow (``orchestrator_helpers.World``) with scripted bundle sources; the default sandbox runs the bundle child in
this process.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from cockpit_helpers import client_for
from orchestrator_helpers import CONFIG, RoleAwareClaude, World, drive_to_review
from pipeline_helpers import MAJORITY_MODEL

from kaggle_agent.analytics.oof_format import encode_oof

ROOT = Path(__file__).resolve().parent.parent

PROBA_MODEL = '''\
import json
import os


def fit(train):
    prior = float(train["y"].mean())
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "prior.json"), "w") as handle:
        json.dump({"prior": prior}, handle)


def predict(bundle_id, X):
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "prior.json")) as handle:
        prior = json.load(handle)["prior"]
    out = []
    for value in X["x1"]:
        p = min(0.9, max(0.1, prior + 0.03 * (float(value) - 3)))
        out.append([1 - p, p])
    return out
'''

VALID_DESCRIBE = {
    "features_used": ["x1", "x2", "x1_squared"],
    "derived_features": [{"name": "x1_squared", "sources": ["x1"], "description": "square of x1"}],
    "hyperparameters": {"depth": 3, "learning_rate": 0.1},
    "native_importances": {"x1": 0.7, "x2": 0.2, "x1_squared": 0.1},
    "model_family": "majority",
}


def with_describe(source: str, text: str) -> str:
    """The model source whose ``fit`` also writes ``text`` as describe.json next to the model."""
    return source.replace(
        "def fit(train):\n",
        "def fit(train):\n    import os as _os\n    with open(_os.path.join(_os.environ['BUNDLE_MODEL_DIR'], 'describe.json'), 'w') as _h:\n"
        f"        _h.write({text!r})\n", 1)


class ScriptedClaude(RoleAwareClaude):
    """The fake Claude of the flow tests; ``model_for(n, bundle_id)`` gives the source of each written bundle."""

    def __init__(self, model_for, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_for = model_for

    def _run(self, prompt, permissions, model):
        response = super()._run(prompt, permissions, model)
        if self._kind(prompt) == "writer":
            import re
            bundle_id = re.search(r"Bundle id: (\S+)", prompt).group(1)
            path = Path(permissions.working_dir) / "bundle" / "model.py"
            path.write_text(f"# call {self.counter}\n" + self.model_for(self.counter, bundle_id), encoding="utf-8")
        return response


def make_world(tmp_path, model=None, repetitions=1, evaluation=None, extra_protocol="", **kwargs) -> World:
    """A World whose bundles use ``model`` (a source or a callable ``(n, bundle_id) -> source``)."""
    from orchestrator_helpers import competition
    source = model if callable(model) else (lambda n, b, _m=model or MAJORITY_MODEL: _m)
    comp = competition(evaluation=evaluation) if evaluation else None
    world = World(tmp_path, comp=comp, llm=ScriptedClaude(source), **kwargs)
    text = world.config.read_text(encoding="utf-8").replace(
        "[protocol]\n", f"[protocol]\nrepetitions = {repetitions}\n{extra_protocol}", 1)
    world.config.write_text(text, encoding="utf-8")
    return world


def finished_world(tmp_path, **kwargs) -> World:
    world = make_world(tmp_path, **kwargs)
    drive_to_review(world)
    return world


def halted_world(tmp_path, **kwargs) -> World:
    """The run stops at the first guidance question: the baseline and round 1 are done, the holdout is untouched."""
    world = make_world(tmp_path, **kwargs)
    assert world.run("start")[0] == 10
    world.lead('[plan]\ndecision = "approve"\n')
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_GUIDANCE"
    return world


def client_of(world: World):
    return client_for(str(world.project / "registry.db"), world.project)


def attempts_of(world: World) -> list[str]:
    registry = world.registry()
    try:
        cycle = registry.list_cycles()[0].id
        return [a.attempt_id for a in sorted(registry.list_attempts(cycle), key=lambda a: a.iteration)]
    finally:
        registry.close()


def cycle_of(world: World) -> str:
    registry = world.registry()
    try:
        return registry.list_cycles()[0].id
    finally:
        registry.close()


def index_lines(world: World) -> list[dict]:
    path = world.project / "analytics" / "index.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- small tables ----------------------------------------------------------------------------------


def oof_frame(truth, prediction=None, proba=None, reps=1, folds=2) -> pd.DataFrame:
    """An OOF table: rows 0..n-1 in each repetition, fold = row % folds."""
    n = len(truth)
    parts = []
    for rep in range(reps):
        data = {"row": np.arange(n), "rep": rep, "fold": np.arange(n) % folds, "truth": list(truth)}
        if proba is not None:
            matrix = np.asarray(proba, dtype=float)
            if matrix.ndim == 1:
                matrix = np.column_stack([1 - matrix, matrix])
            for i in range(matrix.shape[1]):
                data[f"p{i}"] = matrix[:, i]
        else:
            data["prediction"] = list(prediction)
        parts.append(pd.DataFrame(data))
    return pd.concat(parts, ignore_index=True)


def oof_bytes(frame: pd.DataFrame) -> bytes:
    pred_cols = [c for c in frame.columns if c not in ("row", "rep", "fold", "truth")]
    preds = frame[pred_cols[0]].to_numpy() if pred_cols == ["prediction"] else frame[pred_cols].to_numpy()
    return encode_oof(frame["row"], frame["rep"], frame["fold"], frame["truth"], preds)


# -- shared finished runs (built once per test session, never modified by a test) -----------------------

import pytest  # noqa: E402


def crash_source(marker: str) -> str:
    """A model whose second ``fit`` of the bundle crashes (the marker file says the first one ran)."""
    return MAJORITY_MODEL.replace(
        "def fit(train):\n",
        "def fit(train):\n    import os as _os\n"
        f"    if _os.path.exists({marker!r}):\n        raise RuntimeError('second fold')\n"
        f"    open({marker!r}, 'w').close()\n", 1)


@pytest.fixture(scope="session")
def proba_world(tmp_path_factory):
    """Probabilities (log loss), 2 repetitions x 3 folds, valid describe.json; stops before the holdout step."""
    text = json.dumps(VALID_DESCRIBE)
    return halted_world(tmp_path_factory.mktemp("proba"), model=lambda n, b: with_describe(PROBA_MODEL, text),
                          repetitions=2, evaluation="Submissions are evaluated on log loss.")


@pytest.fixture(scope="session")
def labels_world(tmp_path_factory):
    """Labels only (accuracy), 1 repetition x 3 folds, no describe.json."""
    return halted_world(tmp_path_factory.mktemp("labels"))


@pytest.fixture(scope="session")
def failure_world(tmp_path_factory):
    """R0 crashes on the second fold and its fix succeeds; R1 always crashes (failed attempt); R2 is fine."""
    base = tmp_path_factory.mktemp("failure")
    marks = base / "marks"
    marks.mkdir()

    def model(n, bundle_id):
        if "-R1-" in bundle_id or (bundle_id.startswith("toy-1-R0") and not bundle_id.endswith("-f1")):
            return crash_source(str(marks / f"{bundle_id}.mark"))
        return MAJORITY_MODEL

    return finished_world(base / "w", model=model)
