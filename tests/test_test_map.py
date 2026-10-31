"""The test map points to test functions that really exist."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_MAP = ROOT / ".asl" / "test-map.json"


def test_test_map_entries_exist():
    cases = json.loads(TEST_MAP.read_text(encoding="utf-8"))["cases"]
    keys = sorted(int(k) for k in cases)
    assert keys == list(range(1, keys[-1] + 1))  # consecutive; other tickets may add theirs
    assert keys[-1] >= 28  # the 28 cases of the sandbox ticket
    for case, entries in cases.items():
        assert entries, f"case {case} has no test"
        for entry in entries:
            path = ROOT / entry["file"]
            assert path.is_file(), entry
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            assert entry["name"].startswith("test_") and entry["name"] in names, (case, entry)
