"""Ask the model for a JSON payload and validate it with the proposal types.

Knows no concrete type: any class with ``from_json`` and dataclass fields works
(``ShortPlan``, ``Attempt``, ...). Nothing here raises anything but
``ValidationError``.
"""

import dataclasses
import json
import re

from kaggle_agent.proposals import ValidationError

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)


def json_contract(output_type: type) -> str:
    """The instruction that makes the model answer with one JSON object."""
    required, optional = [], []
    for field in dataclasses.fields(output_type):
        has_default = (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )
        (optional if has_default else required).append(field.name)
    text = (
        "Reply with a single JSON object and nothing else (no commentary). "
        f"Required keys: {', '.join(required)}."
    )
    if optional:
        text += f" Optional keys: {', '.join(optional)}."
    return text


def extract_json(text: str) -> str:
    """The JSON object inside ``text``: bare, in a code fence or surrounded by prose."""
    stripped = text.strip()
    fence = _FENCE.search(stripped)
    if fence:
        return fence.group(1).strip()
    if stripped.startswith("{"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_structured(text: object, output_type: type):
    """Parse and validate ``text`` as ``output_type``; raise ``ValidationError`` on any problem."""
    name = getattr(output_type, "__name__", "response")
    if not isinstance(text, str):
        raise ValidationError(name, "the response is not text")
    try:
        return output_type.from_json(extract_json(text))
    except ValidationError:
        raise
    except Exception as exc:  # a malformed payload must never crash the caller
        raise ValidationError(name, f"could not be parsed ({type(exc).__name__})") from None
