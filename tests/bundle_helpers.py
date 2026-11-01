"""Fake bundles written to a temp folder, and builders for the bundle tests. No network, no Claude call."""

import json
import textwrap
from pathlib import Path

from evaluation_helpers import classification_frame, frozen_engine, make_protocol

from kaggle_agent.bundle import BundleRunner, load_bundle
from kaggle_agent.sandbox import Sandbox, SandboxPolicy, SubprocessSandbox

FRAME = classification_frame()
BUNDLE_DIR = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "bundle"
EVALUATION_DIR = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "evaluation"
# generous: the child imports pandas before it does anything
TIMEOUT = 60
SHORT_TIMEOUT = 4


def write_bundle(root: Path, modules: dict, *, dependencies=(), bundle_id="B1", entry_points=None,
                 name="bundle") -> Path:
    """Write a bundle folder under ``root`` and return it; ``modules`` maps file names to source."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for file_name, source in modules.items():
        path = folder / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    manifest = {
        "bundle_id": bundle_id,
        "dependencies": list(dependencies),
        "entry_points": entry_points or {"fit": "model:fit", "predict": "model:predict"},
    }
    (folder / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def model(predict_body: str, fit_body: str = "pass", header: str = "") -> dict:
    """A single ``model.py`` with the given bodies (statements, already indented by 4 below)."""
    def indent(text):
        return textwrap.indent(textwrap.dedent(text).strip("\n"), "    ")

    source = (
        textwrap.dedent(header).strip("\n") + "\n\n"
        + f"def fit(train):\n{indent(fit_body)}\n\n"
        + f"def predict(bundle_id, X):\n{indent(predict_body)}\n"
    )
    return {"model.py": source}


def constant_bundle(root, value=0, **kwargs) -> Path:
    return write_bundle(root, model(f"return [{value!r}] * len(X)"), **kwargs)


def sleeper_bundle(root, seconds=60) -> Path:
    return write_bundle(
        root, model(f"time.sleep({seconds})\nreturn [0] * len(X)", header="import time"),
    )


def raiser_bundle(root, where="fit") -> Path:
    boom = 'raise RuntimeError("boom from the bundle")'
    return write_bundle(root, model(
        boom if where == "predict" else "return [0] * len(X)",
        boom if where == "fit" else "pass",
    ))


def returning_bundle(root, expression: str, header: str = "") -> Path:
    """A bundle whose predict returns ``expression`` (evaluated in the child)."""
    return write_bundle(root, model(f"return {expression}", header=header))


def policy(timeout=TIMEOUT, **kwargs) -> SandboxPolicy:
    return SandboxPolicy(timeout_seconds=timeout, **kwargs)


class RecordingSandbox(Sandbox):
    """Wraps a real sandbox; keeps requests, results and what the staging folder held."""

    def __init__(self, inner: Sandbox | None = None) -> None:
        self.inner = inner or SubprocessSandbox()
        self.calls = []
        self.results = []
        self.staged = []

    def run(self, policy, request):
        self.calls.append((policy, request))
        staging = Path(request.args[0])
        files = {}
        for path in staging.rglob("*"):
            if path.is_file() and "bundle" not in path.relative_to(staging).parts[:1]:
                files[path.relative_to(staging).as_posix()] = path.read_bytes()
        self.staged.append(files)
        result = self.inner.run(policy, request)
        self.results.append(result)
        return result


class CannedSandbox(Sandbox):
    """A fake sandbox that returns a canned result and starts nothing."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def run(self, policy, request):
        self.calls.append((policy, request))
        return self.result


def make_runner(folder, *, sandbox=None, sandbox_policy=None, input_kind="labels", target="y"):
    loaded = load_bundle(folder)
    runner = BundleRunner(
        sandbox or SubprocessSandbox(), sandbox_policy or policy(), loaded,
        target_column=target, input_kind=input_kind,
    )
    return loaded, runner


def engine_and_registry(**protocol_overrides):
    return frozen_engine(make_protocol(**protocol_overrides))


def real_package_bytes() -> dict:
    """Bytes of every real source file of the bundle and evaluation packages."""
    return {
        str(p): p.read_bytes()
        for root in (BUNDLE_DIR, EVALUATION_DIR)
        for p in sorted(root.rglob("*.py"))
    }


SNOOP_SOURCE = '''
import os, sys

NEEDLES = {needles!r}
NAMES = ("y.csv", "y_valid.csv", "y_holdout.csv", "labels.csv", "holdout.csv", "target.csv",
         "validation.csv", "test_labels.csv")


def scan():
    staging = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    texts = []
    roots = [staging, os.getcwd()]
    hits = 0
    for root in roots:
        for current, dirs, files in os.walk(root):
            if os.path.abspath(current) == os.path.join(staging, "bundle"):
                dirs[:] = []
                continue
            for name in files:
                if name in NAMES:
                    hits += 1
                with open(os.path.join(current, name), "rb") as handle:
                    texts.append(handle.read().decode("utf-8", "replace"))
    texts.append(" ".join(list(os.environ) + list(os.environ.values()) + sys.argv))
    if os.path.exists(os.path.join(staging, "x.csv")):
        with open(os.path.join(staging, "x.csv")) as handle:
            header = handle.readline().strip().split(",")
        if "y" in header:
            hits += 1
    blob = "\\n".join(texts)
    return hits + sum(blob.count(n) for n in NEEDLES)


def fit(train):
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "hits.txt"), "w") as handle:
        handle.write(str(scan()))


def predict(bundle_id, X):
    print("snoop finished")
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "hits.txt")) as handle:
        fit_hits = int(handle.read())
    return [float(fit_hits + scan())] * len(X)
'''


def snooper_bundle(root, needles) -> Path:
    """Looks for ``needles`` and label files in the staging folder, working directory and environment.

    It predicts the number of hits as a constant, so a test asserts the predictions are all 0.
    """
    return write_bundle(root, {"model.py": SNOOP_SOURCE.format(needles=list(needles))})


def protected_writer_bundle(root, target: str, chmod_first=False) -> Path:
    """Tries to append to ``target``: an absolute path, or ``env:NAME`` under ``SANDBOX_PROTECTED_DIR``."""
    return write_bundle(root, model(
        f"""
        import os
        target = {target!r}
        if target.startswith("env:"):
            target = os.path.join(os.environ["SANDBOX_PROTECTED_DIR"], target[4:])
        if {chmod_first!r}:
            os.chmod(target, 0o644)
        with open(target, "a") as handle:
            handle.write("tampered")
        return [0] * len(X)
        """,
    ))
