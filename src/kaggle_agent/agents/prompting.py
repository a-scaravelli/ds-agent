"""Prompt rendering shared by the roles: instructions plus the typed input as JSON."""

import json


def render_prompt(instructions: str, role_input: object) -> str:
    """The role instructions followed by the input, serialized deterministically.

    ``role_input`` is any typed input with ``to_dict()``. The JSON contract of
    the output is added later by ``LlmClient.ask``.
    """
    payload = json.dumps(role_input.to_dict(), sort_keys=True, ensure_ascii=False)
    return f"{instructions.strip()}\n\nInput (JSON):\n{payload}"
