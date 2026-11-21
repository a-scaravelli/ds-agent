"""The optional describe.json: present, missing, invalid; the prompts that ask for it (cases 26-29)."""

import ast
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from agents_helpers import feature_engineering_input, modeling_input
from analytics_helpers import (  # noqa: F401  (fixtures)
    VALID_DESCRIBE,
    attempts_of,
    client_of,
    index_lines,
    labels_world,
    proba_world,
)
from evaluation_helpers import make_protocol
from orchestrator_helpers import train_csv
from test_agents_limits import with_limits
from writer_helpers import make_request, task

from kaggle_agent.agents import build_feature_engineering_prompt, build_modeling_prompt
from kaggle_agent.analytics.describe import INVALID, MISSING, VALID, validate_describe
from kaggle_agent.bundle import contract
from kaggle_agent.evaluation import FrozenSplit, derive_seed
from kaggle_agent.orchestrator.analytics_record import RecordingCandidate
from kaggle_agent.proposals import describe as spec
from kaggle_agent.proposals import describe_paragraph
from kaggle_agent.sandbox.protected import hash_path
from kaggle_agent.writer import DEFAULT_DEPENDENCIES, build_prompt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kaggle_agent"


# 26
def test_a_valid_describe_is_stored_normalised_and_shown(proba_world):
    lines = [x for x in index_lines(proba_world) if x["kind"] == "describe"]
    assert lines and all(x["status"] == "ok" and len(x["sha256"]) == 64 for x in lines)
    stored = json.loads((proba_world.project / "analytics" / lines[0]["file"]).read_text(encoding="utf-8"))
    assert stored["v"] == 1 and stored["derived_features"][0]["sources"] == ["x1"]
    client = client_of(proba_world)
    attempt = attempts_of(proba_world)[-1]
    data = client.get(f"/api/attempts/{attempt}/features").get_json()["data"]
    content = data["describe"]["content"]
    assert content["features_used"] == VALID_DESCRIBE["features_used"]
    assert content["hyperparameters"] == VALID_DESCRIBE["hyperparameters"]
    assert data["native"]["features"] == VALID_DESCRIBE["native_importances"]
    assert data["families"]["x1_squared"] == "x1"  # a derived feature sits under its source column
    page = client.get(f"/attempts/{attempt}/analytics").get_data(as_text=True)
    assert "Features" in page and "describe" in page.lower()


def test_unknown_keys_are_dropped_with_a_warning_and_the_raw_text_is_not_kept():
    raw = json.dumps({"features_used": ["a"], "secret_plan": "x", "model_family": "linear"}).encode()
    result = validate_describe(raw)
    assert result.status == VALID and "secret_plan" not in result.content
    assert result.content["features_used"] == ["a"] and result.warnings and "secret_plan" in result.warnings[0]


# 27
def test_a_missing_describe_leaves_the_run_alone_and_the_page_says_not_provided(labels_world):
    lines = [x for x in index_lines(labels_world) if x["kind"] == "describe"]
    assert lines and all(x["status"] == "missing" and "file" not in x for x in lines)
    assert validate_describe(None).status == MISSING
    attempt = attempts_of(labels_world)[-1]
    client = client_of(labels_world)
    page = client.get(f"/attempts/{attempt}/analytics").get_data(as_text=True)
    assert "Not provided" in page and "did not write a describe.json" in page
    data = client.get(f"/api/attempts/{attempt}/features").get_json()["data"]
    assert data["describe"]["status"] == "missing" and data["native"] is None


class DescribingInner:
    """A runner with fit_model that writes whatever ``make`` says into the temporary model folder."""

    def __init__(self, make):
        self.make = make

    def fit_model(self, x_train, y_train, seed, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.make(model_dir)

    def predict_with_model(self, model_dir, x, seed):
        return np.zeros(len(x), dtype=int)

    def fit_predict(self, *args):  # not used: fit_model exists
        raise AssertionError


def record_with(tmp_path, make):
    frame = pd.read_csv(io.StringIO(train_csv()))
    protocol = make_protocol(validation_params={"n_folds": 3, "stratify_by": "y"})
    folder = Path(tmp_path).resolve() / "project"
    folder.mkdir()
    candidate = RecordingCandidate(DescribingInner(make), frame, protocol, "B1", folder)
    split = FrozenSplit.create(frame, protocol)
    outputs = []
    for part in split.partitions():
        x_train, y_train = split.data(part.train_rows)
        x_valid, _ = split.data(part.valid_rows)
        outputs.append(candidate.fit_predict(x_train, y_train, x_valid, derive_seed(protocol.seed, "candidate", 0)))
    assert all(list(o) == [0] * len(o) for o in outputs)  # the run is unaffected
    lines = [json.loads(x) for x in (folder / "analytics" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    return folder, next(x for x in lines if x["kind"] == "describe")


def text_writer(text):
    return lambda folder: (folder / "describe.json").write_text(text, encoding="utf-8")


def symlink_writer(folder):
    target = folder.parent / "elsewhere.json"
    target.write_text(json.dumps({"features_used": ["a"]}), encoding="utf-8")
    os.symlink(target, folder / "describe.json")


BAD = {
    "malformed": (text_writer("{not json"), "not valid JSON"),
    "too_big": (text_writer(json.dumps({"notes": "x" * 70_000})), "larger than 64 KB"),
    "wrong_type": (text_writer(json.dumps({"features_used": "age"})), "must be a list"),
    "top_level_list": (text_writer("[1, 2]"), "JSON object"),
    "symlink": (symlink_writer, "symbolic link"),
    "sk_key": (text_writer(json.dumps({"notes": "sk-" + "a1B2c3D4e5F6g7H8"})), "looks like a key"),
    "aws_key": (text_writer(json.dumps({"model_family": "AKIA" + "ABCDEFGH12345678"})), "looks like a key"),
    "private_key": (text_writer(json.dumps({"notes": "-----BEGIN RSA PRIVATE KEY-----"})), "looks like a key"),
    "long_hex": (text_writer(json.dumps({"hyperparameters": {"salt": "0123456789abcdef" * 3}})), "looks like a key"),
}


# 28
@pytest.mark.parametrize("name", sorted(BAD))
def test_an_invalid_describe_is_stored_as_invalid_with_a_reason_and_no_content(tmp_path, name):
    make, reason = BAD[name]
    folder, line = record_with(tmp_path, make)
    assert line["status"] == INVALID and reason in line["reason"] and "file" not in line
    assert not list((folder / "analytics").glob("describe/*"))
    everything = b"".join(p.read_bytes() for p in (folder / "analytics").rglob("*") if p.is_file() and p.suffix != ".gz")
    for secret in (b"sk-a1B2c3D4", b"AKIAABCDEFGH", b"PRIVATE KEY", b"0123456789abcdef"):
        assert secret not in everything  # the key-like value is never stored


def test_a_valid_describe_through_the_recorder_gets_a_hash_and_a_normalised_copy(tmp_path):
    folder, line = record_with(tmp_path, text_writer(json.dumps({**VALID_DESCRIBE, "extra": 1})))
    assert line["status"] == "ok" and line["warnings"] and (folder / "analytics" / line["file"]).is_file()
    assert "extra" not in (folder / "analytics" / line["file"]).read_text(encoding="utf-8")


# 28
def test_the_attempt_page_says_provided_but_invalid(tmp_path):
    from analytics_helpers import make_world
    from orchestrator_helpers import drive_to_review  # noqa: F401
    from pipeline_helpers import MAJORITY_MODEL

    from analytics_helpers import halted_world, with_describe
    world = halted_world(tmp_path, model=lambda n, b: with_describe(MAJORITY_MODEL, "{broken"))
    attempt = attempts_of(world)[-1]
    client = client_of(world)
    page = client.get(f"/attempts/{attempt}/analytics").get_data(as_text=True)
    assert "provided but invalid" in page and "not valid JSON" in page
    data = client.get(f"/api/attempts/{attempt}/features").get_json()["data"]
    assert data["describe"]["status"] == "invalid" and data["describe"]["content"] is None


# 29
def test_the_prompts_carry_the_paragraph_and_describe_is_never_required():
    paragraph = " ".join(describe_paragraph().split())
    for name in spec.DESCRIBE_FIELDS:
        assert name in paragraph
    assert str(spec.MAX_DESCRIBE_BYTES // 1024) in paragraph and str(spec.MAX_FEATURES) in paragraph
    writer = " ".join(build_prompt(make_request("x"), task(), DEFAULT_DEPENDENCIES).split())
    assert paragraph in writer
    for build, make in ((build_feature_engineering_prompt, feature_engineering_input), (build_modeling_prompt, modeling_input)):
        assert paragraph in " ".join(build(with_limits(make())).split())
    assert "describe.json" in contract.__doc__ and "optional" in contract.__doc__
    for path in (SRC / "bundle" / "layout.py", SRC / "writer" / "checks.py"):
        assert "describe" not in path.read_text(encoding="utf-8").lower(), path.name
    guards = (ROOT / "tests" / "test_orchestrator_guards.py").read_text(encoding="utf-8")
    tree = ast.parse(guards)
    pinned = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "PINNED")
    for name in ("bundle", "writer", "loop", "evaluation", "registry"):
        assert hash_path(SRC / name) == pinned[name], name
    # loop, evaluation and registry kept the pins of before this ticket (updated deliberately by earlier tickets only)
    assert pinned["evaluation"].startswith("c267079f") and pinned["registry"].startswith("04350ec5")
    assert pinned["loop"].startswith("83835939")
