"""Helpers of the writer tests: a scripted fake Claude, bundle builders and request builders.

Everything is offline. Paths come from ``tmp_path`` resolved; no environment is read here.
"""

import json
import shutil
import textwrap
from pathlib import Path

from registry_helpers import make_cycle

from kaggle_agent.agents import ColumnSummary, DataFindings, DataSummary
from kaggle_agent.llm import LlmClient, RawResponse, ToolPermissions, Usage
from kaggle_agent.llm.interface import BackoffPolicy
from kaggle_agent.loop.writer import BundleRequest, FixRequest, RoleOutputs
from kaggle_agent.proposals import Attempt
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.sandbox import ProtectedPath
from kaggle_agent.writer import ClaudeBundleWriter, WriterTask

CYCLE = "C1"

MODEL_SOURCE = '''\
import json
import os
from collections import Counter

_FILE = "state.json"


def fit(train):
    majority = Counter(train["y"].tolist()).most_common(1)[0][0]
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], _FILE), "w", encoding="utf-8") as handle:
        json.dump({"majority": majority}, handle)


def predict(bundle_id, X):
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], _FILE), encoding="utf-8") as handle:
        majority = json.load(handle)["majority"]
    return [majority] * len(X)
'''

NO_PREDICT_SOURCE = "def fit(train):\n    pass\n"
UNDECLARED_SOURCE = "import scipy\n\n\ndef fit(train):\n    pass\n\n\ndef predict(bundle_id, X):\n    return [0] * len(X)\n"


def manifest(bundle_id, dependencies=()):
    return json.dumps({
        "bundle_id": bundle_id,
        "dependencies": list(dependencies),
        "entry_points": {"fit": "model:fit", "predict": "model:predict"},
    })


def bundle_files(bundle_id, source=MODEL_SOURCE, dependencies=(), manifest_id=None):
    return {
        "bundle.json": manifest(manifest_id or bundle_id, dependencies),
        "model.py": source,
    }


def put(working_dir, files, subfolder="bundle"):
    """Write ``files`` (name -> text) under ``working_dir/subfolder``."""
    root = Path(working_dir) / subfolder
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def writes(files, reply='{"summary": "done"}'):
    """A script step: write ``files`` into the staging bundle folder, then answer ``reply``."""
    def step(working_dir):
        put(working_dir, files)
        return reply
    return step


def replaces(files, reply='{"summary": "done"}'):
    """A repair step: empty the staging bundle folder (links included), then write ``files``."""
    def step(working_dir):
        root = Path(working_dir) / "bundle"
        for item in list(root.iterdir()):
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
        put(working_dir, files)
        return reply
    return step


def valid_step(bundle_id, **kwargs):
    return writes(bundle_files(bundle_id), **kwargs)


class ScriptedClaude(LlmClient):
    """A fake Claude: each step is ``fn(working_dir) -> reply text or None`` or an exception to raise."""

    def __init__(self, steps, **kwargs):
        kwargs.setdefault("backoff", BackoffPolicy(max_retries=0))
        kwargs.setdefault("sleep", lambda seconds: None)
        super().__init__(**kwargs)
        self.steps = list(steps)
        self.prompts: list[str] = []
        self.permissions: list[ToolPermissions] = []
        self.staging_listing: list[list[str]] = []

    def _run(self, prompt, permissions, model):
        self.prompts.append(prompt)
        self.permissions.append(permissions)
        self.staging_listing.append(sorted(p.name for p in permissions.working_dir.iterdir()))
        step = self.steps[len(self.prompts) - 1]
        if isinstance(step, Exception):
            raise step
        text = step(permissions.working_dir)
        return RawResponse(text=text if text is not None else "{}", usage=Usage(10, 5), model="fake-model")


def data_summary():
    return DataSummary(
        60, "y", (ColumnSummary("x1", "float64", 0.0, 60), ColumnSummary("y", "int64", 0.0, 2))
    )


def task(**kwargs):
    base = dict(target="y", metric="accuracy", direction="maximize", input_kind="labels",
                summary=data_summary())
    return WriterTask(**{**base, **kwargs})


def attempt(name, **kwargs):
    base = dict(id=name, phase="iteration", hypothesis=f"hypothesis of {name}", rationale="r",
                what_changed=f"what changed in {name}", measure="accuracy", exploratory=True)
    return Attempt(**{**base, **kwargs})


def outputs():
    return RoleOutputs(
        DataFindings("findings profile text", ("some quality issue",)),
        attempt("fe"), attempt("md"),
    )


def make_request(folder, bundle_id="B1", fix=None):
    out = outputs()
    return BundleRequest(attempt=out.modeling, outputs=out, bundle_id=bundle_id, folder=Path(folder), fix=fix)


def fix_request(folder, bundle_id="B1-f1", previous="B1", reason="KeyError: age", retry=1):
    return make_request(folder, bundle_id, FixRequest(retry, reason, previous))


def new_registry():
    registry = SqliteRegistry()
    registry.add_cycle(make_cycle(CYCLE))
    return registry


class Env:
    """A writer over a scripted client, with its directories under ``tmp_path``."""

    def __init__(self, tmp_path, steps, registry=None, **kwargs):
        self.base = Path(tmp_path).resolve()
        self.root = self.base / "workspace"
        self.bundles = self.base / "bundles"
        self.bundles.mkdir()
        self.registry = registry or new_registry()
        self.client = ScriptedClaude(steps)
        kwargs.setdefault("task", task())
        kwargs.setdefault("protected", ())
        self.writer = ClaudeBundleWriter(self.client, self.registry, CYCLE, self.root, **kwargs)

    def request(self, bundle_id="B1", fix=None):
        return make_request(self.bundles / bundle_id, bundle_id, fix)

    def rows(self):
        return self.registry.list_llm_calls(CYCLE)


def protected_file(base):
    """A tmp file registered as a protected path; returns (entry, path)."""
    path = Path(base).resolve() / "protected" / "guarded.txt"
    path.parent.mkdir()
    path.write_text("original", encoding="utf-8")
    return ProtectedPath("tmp/guarded.txt", path), path


def dedent(text):
    return textwrap.dedent(text)
