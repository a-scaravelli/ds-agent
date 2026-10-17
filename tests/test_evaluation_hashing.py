"""Hash of protocol and evaluation sources, computed at runtime (no network)."""

import re
import shutil
from pathlib import Path

import pytest
from evaluation_helpers import frozen_engine, make_protocol

import kaggle_agent.evaluation as evaluation
from kaggle_agent.evaluation import (
    ProtocolHashMismatchError,
    evaluation_code_hash,
    protocol_hash,
)

PACKAGE = Path(evaluation.__file__).parent


def _copy(tmp_path):
    target = tmp_path / "evaluation"
    shutil.copytree(PACKAGE, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def test_code_hash_changes_with_py_sources_only(tmp_path):
    copy = _copy(tmp_path)
    original = evaluation_code_hash(copy)
    assert original == evaluation_code_hash(PACKAGE)
    (copy / "notes.txt").write_text("not code")
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "x.py").write_text("ignored = 1")
    assert evaluation_code_hash(copy) == original
    target = copy / "threshold.py"
    target.write_text(target.read_text(encoding="utf-8") + " ")
    changed = evaluation_code_hash(copy)
    assert changed != original
    protocol = make_protocol()
    assert protocol_hash(protocol, changed) != protocol_hash(protocol, original)


def test_modified_evaluation_code_blocks_the_run(tmp_path):
    copy = _copy(tmp_path)
    engine, _ = frozen_engine(package_dir=copy)
    engine.verify()
    (copy / "engine.py").write_text((copy / "engine.py").read_text(encoding="utf-8") + "\n#")
    with pytest.raises(ProtocolHashMismatchError):
        engine.verify()


def test_protocol_change_changes_the_hash():
    code = evaluation_code_hash()
    assert protocol_hash(make_protocol(), code) != protocol_hash(make_protocol(threshold_k=3), code)
    assert protocol_hash(make_protocol(), code) == protocol_hash(make_protocol(), code)


def test_code_hash_is_computed_at_runtime_not_a_manual_constant():
    first, second = evaluation_code_hash(), evaluation_code_hash()
    assert first == second and re.fullmatch(r"[0-9a-f]{64}", first)
    for path in PACKAGE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(EVALUATION_)?(CODE_)?VERSION\s*=", source, flags=re.M), path


def test_library_versions_are_outside_the_hash():
    source = (PACKAGE / "hashing.py").read_text(encoding="utf-8")
    code = source.split('"""', 2)[2]
    assert "__version__" not in code and "pandas" not in code and "sklearn" not in code
