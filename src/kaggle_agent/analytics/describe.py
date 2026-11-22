"""The platform validation of ``describe.json`` (the contract is in ``proposals.describe``).

Pure: raw bytes in, a result out. ``valid`` carries a normalised copy (known keys only, strings cut and
checked); ``invalid`` carries a reason and NO content; a key-like string is never stored or shown.
"""

import json
import math
import re
from dataclasses import dataclass, field

from kaggle_agent.proposals import describe as contract

VALID, INVALID, MISSING = "valid", "invalid", "missing"
_KEY_LIKE = re.compile(
    r"sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY|[0-9a-fA-F]{32,}"
    r"|[A-Za-z0-9+/=_-]{40,}"
)


@dataclass(frozen=True)
class DescribeResult:
    status: str
    content: dict | None = None
    reason: str | None = None
    warnings: tuple = field(default_factory=tuple)


class _Invalid(Exception):
    pass


def looks_like_a_key(text: str) -> bool:
    return _KEY_LIKE.search(text) is not None


def _text(name: str, value, limit: int = contract.MAX_STRING) -> str:
    if not isinstance(value, str):
        raise _Invalid(f"{name} must be a text")
    if len(value) > limit:
        raise _Invalid(f"{name} is longer than {limit} characters")
    return value


def _scan(value) -> None:
    if isinstance(value, str):
        if looks_like_a_key(value):
            raise _Invalid("a value looks like a key or a secret (it is not stored)")
    elif isinstance(value, list):
        for item in value:
            _scan(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan(str(key))
            _scan(item)


def _names(name: str, value) -> list[str]:
    if not isinstance(value, list):
        raise _Invalid(f"{name} must be a list")
    if len(value) > contract.MAX_FEATURES:
        raise _Invalid(f"{name} has more than {contract.MAX_FEATURES} items")
    return [_text(f"{name}[]", item) for item in value]


def _derived(value) -> list[dict]:
    if not isinstance(value, list):
        raise _Invalid("derived_features must be a list")
    if len(value) > contract.MAX_FEATURES:
        raise _Invalid(f"derived_features has more than {contract.MAX_FEATURES} items")
    out = []
    for item in value:
        if not isinstance(item, dict):
            raise _Invalid("derived_features items must be objects")
        name = _text("derived_features.name", item.get("name"))
        sources = item.get("sources", [])
        if not isinstance(sources, list) or len(sources) > contract.MAX_SOURCES:
            raise _Invalid(f"derived_features.sources must be a list of at most {contract.MAX_SOURCES}")
        entry = {"name": name, "sources": [_text("derived_features.sources[]", s) for s in sources]}
        if item.get("description") is not None:
            entry["description"] = _text("derived_features.description", item["description"])
        out.append(entry)
    return out


def _number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _Invalid(f"{name} must be a finite number")
    return float(value)


def _scalar(name: str, value):
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _number(name, value)
    return _text(name, value)


def _hyperparameters(value) -> dict:
    if not isinstance(value, dict):
        raise _Invalid("hyperparameters must be an object")
    if len(value) > contract.MAX_HYPERPARAMETERS:
        raise _Invalid(f"hyperparameters has more than {contract.MAX_HYPERPARAMETERS} items")
    out = {}
    for key, item in value.items():
        label = f"hyperparameters.{_text('key', key)}"
        if isinstance(item, list):
            if len(item) > 20:
                raise _Invalid(f"{label} has more than 20 values")
            out[key] = [_scalar(label, v) for v in item]
        else:
            out[key] = _scalar(label, item)
    return out


def _importances(value) -> dict:
    if not isinstance(value, dict):
        raise _Invalid("native_importances must be an object")
    if len(value) > contract.MAX_FEATURES:
        raise _Invalid(f"native_importances has more than {contract.MAX_FEATURES} items")
    return {_text("native_importances key", k): _number(f"native_importances.{k}", v) for k, v in value.items()}


def validate_describe(raw: bytes | None) -> DescribeResult:
    """``missing`` for None, ``invalid(reason)`` for any problem, else ``valid`` with the normalised copy."""
    if raw is None:
        return DescribeResult(MISSING, reason="the bundle did not write describe.json")
    try:
        if len(raw) > contract.MAX_DESCRIBE_BYTES:
            raise _Invalid(f"the file is larger than {contract.MAX_DESCRIBE_BYTES // 1024} KB")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise _Invalid("the file is not valid JSON") from None
        if not isinstance(value, dict):
            raise _Invalid("the top level must be a JSON object")
        _scan(value)
        warnings = tuple(f"unknown key {str(k)[:40]!r} was dropped" for k in value
                         if k not in contract.DESCRIBE_FIELDS)
        content: dict = {"v": contract.DESCRIBE_VERSION}
        if "features_used" in value:
            content["features_used"] = _names("features_used", value["features_used"])
        if "derived_features" in value:
            content["derived_features"] = _derived(value["derived_features"])
        if "hyperparameters" in value:
            content["hyperparameters"] = _hyperparameters(value["hyperparameters"])
        if "native_importances" in value:
            content["native_importances"] = _importances(value["native_importances"])
        for key in ("model_family", "notes"):
            if key in value:
                content[key] = _text(key, value[key], contract.MAX_NOTES if key == "notes" else contract.MAX_STRING)
        return DescribeResult(VALID, content, None, warnings)
    except _Invalid as error:
        return DescribeResult(INVALID, None, str(error))
