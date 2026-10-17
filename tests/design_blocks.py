"""Minimal, explicit reader of the YAML example blocks in docs/design.md (no PyYAML)."""

import ast
import re
from pathlib import Path

DESIGN = Path(__file__).resolve().parent.parent / "docs" / "design.md"


def yaml_blocks() -> list[str]:
    text = DESIGN.read_text(encoding="utf-8")
    return re.findall(r"```yaml\n(.*?)```", text, flags=re.S)


def parse_block(block: str) -> dict:
    """Read `key: value` lines; values starting with `"` or `[` go through literal_eval."""
    out = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(": ")
        value = value.strip()
        out[key.strip()] = ast.literal_eval(value) if value[:1] in ('"', "[") else value
    return out


def block_with_key(key: str) -> dict:
    for block in yaml_blocks():
        parsed = parse_block(block)
        if key in parsed:
            return parsed
    raise AssertionError(f"no yaml block with key {key!r} in the design")
