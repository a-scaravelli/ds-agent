"""Deterministic fingerprint of an attempt (design section 7)."""

import hashlib
import json
import math

from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import require_text


def _normalize(value: object, path: str) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("config", f"{path} must be finite")
        return int(value) if value == int(value) else value
    if isinstance(value, (list, tuple)):
        return [_normalize(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError("config", f"{path} has a non-string key")
            out[key] = _normalize(item, f"{path}.{key}")
        return out
    raise ValidationError("config", f"{path} is not JSON-compatible ({type(value).__name__})")


def normalize_config(config: dict) -> str:
    """Canonical JSON: sorted keys, tuples as lists, integral floats as ints."""
    if not isinstance(config, dict):
        raise ValidationError("config", "must be an object")
    normalized = _normalize(config, "config")
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_fingerprint(
    config: dict, code_hash: str, data_hash: str, protocol_hash: str
) -> str:
    """SHA-256 hex digest over normalized config, code, data and protocol hashes."""
    parts = [
        normalize_config(config),
        require_text("code_hash", code_hash),
        require_text("data_hash", data_hash),
        require_text("protocol_hash", protocol_hash),
    ]
    # JSON-encoding the list keeps the parts unambiguously separated
    payload = json.dumps(parts, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
