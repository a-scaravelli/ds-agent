"""Ask the model for a JSON payload and validate it with the proposal types.

Knows no concrete type: any class with ``from_json`` and dataclass fields works
(``ShortPlan``, ``Attempt``, ...). Nothing here raises anything but
``ValidationError``.
"""

import dataclasses
import json
import re
import types
import typing

from kaggle_agent.proposals import OBJECT_LABELS, ValidationError

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)


_ANY = "any JSON value"
_REASON_LIMIT = 200


def _json_type(annotation: object) -> str:
    """JSON type of a field annotation; unmapped annotations fall back to any JSON value."""
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation == tuple[str, ...]:
        return "list of strings"
    args = typing.get_args(annotation)
    if args and (typing.get_origin(annotation) in (typing.Union, types.UnionType)):
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1 and len(args) == 2:
            return f"{_json_type(rest[0])} or null"
    return _ANY


def json_contract(output_type: type) -> str:
    """The instruction that makes the model answer with one JSON object.

    Types come from the annotations; constraints come from the optional
    ``contract_hints()`` of the type, kept next to the type itself.
    """
    try:
        annotations = typing.get_type_hints(output_type)
    except Exception:  # unresolvable annotations: types fall back to any JSON value
        annotations = {}
    required, optional, lines = [], [], []
    for field in dataclasses.fields(output_type):
        has_default = (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )
        (optional if has_default else required).append(field.name)
        kind = _json_type(annotations.get(field.name))
        lines.append(f"- {field.name}: {kind}, {'optional' if has_default else 'required'}")
    text = (
        "Reply with a single JSON object and nothing else (no commentary), "
        "and use only these keys. "
        f"Required keys: {', '.join(required)}."
    )
    if optional:
        text += f" Optional keys: {', '.join(optional)}."
    text += "\nTypes:\n" + "\n".join(lines)
    hints_method = getattr(output_type, "contract_hints", None)
    hints = hints_method() if callable(hints_method) else {}
    if hints:
        constraints = [f"- {k}: {v}" for k, v in hints.items() if k != "__all__"]
        if "__all__" in hints:
            constraints.append(f"- whole object: {hints['__all__']}")
        text += "\nConstraints:\n" + "\n".join(constraints)
    return text


def sanitize_rejection(exc: ValidationError, output_type: type) -> str:
    """'field: reason' of a validation error, safe to store: never the model's text.

    The field is kept only if it is a field of the type, the type name or an
    object label of ``proposals``; anything else (for instance an unknown key
    invented by the model) becomes 'response'.
    """
    name = getattr(output_type, "__name__", "response")
    known = {f.name for f in dataclasses.fields(output_type)}
    known |= {name, name.lower(), *OBJECT_LABELS}
    field = re.sub(r"\[\d+\]$", "", str(exc.field))
    if field not in known:
        field = "response"
    return f"{field}: {exc.reason}"[:_REASON_LIMIT]


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
