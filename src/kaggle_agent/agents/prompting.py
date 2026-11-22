"""Prompt rendering shared by the roles: instructions plus the typed input as JSON."""

import json

from kaggle_agent.proposals import LIBRARY_APIS, LIBRARY_RULES, describe_paragraph


def libraries_note(allowed_libraries: tuple[str, ...]) -> str:
    """What the roles may propose with the listed libraries, derived from the list.

    A library that is not in the list (a configured override) leaves no trace in the text.
    """
    boosting = [name for name in LIBRARY_APIS if name in allowed_libraries]
    if not boosting:
        return ""
    apis = ", ".join(f"{LIBRARY_APIS[name]} ({name})" for name in boosting)
    rules = "; ".join(f"{name}: {LIBRARY_RULES[name]}" for name in boosting)
    return (
        f"Besides the models of scikit-learn, the bundle can use {apis}, through their "
        "scikit-learn-compatible classes. Rules for the bundle: model files only under the "
        f"directory named by BUNDLE_MODEL_DIR; explicit thread counts (thread_count or n_jobs) "
        f"and explicit seeds. Per library, {rules}."
    )


def describe_note() -> str:
    """The optional describe.json request, shared with the writer prompt (ticket 29)."""
    return (
        "The bundle writer turns your attempt into code. " + describe_paragraph()
        + " So name the input columns and the derived features (with their source columns) in "
        "what_changed."
    )


STEERING_NOTE = (
    "The DS steers this run (the \"steering\" object of the input): follow the focus, use the pinned features "
    "and the families to try, and stay away from the avoided features and families. Nothing is checked in "
    "code: the reviewer reads the result against this steering. Do not mention that you were steered."
)
CARRY_NOTE = (
    "This cycle continues an earlier run (the \"carry\" object of the input): it holds the digest of the "
    "attempt to start from, the best and failed attempts, the guidance and reviewer texts of that run. "
    "Re-implement and improve the idea in your own words: no code is carried. If same_protocol is false, "
    "the carried scores were measured under another protocol and are not comparable with the new ones."
)
REQUEST_NOTE = (
    "The DS wrote the hypothesis of this candidate (the \"request\" object of the input): propose it as "
    "your attempt, as faithfully as the evidence allows, and respect its constraints. It is evaluated under "
    "the same frozen protocol as every other candidate."
)


def render_prompt(
    instructions: str, role_input: object, limits_note: str = "", libraries: bool = False,
    describe: bool = False,
) -> str:
    """The role instructions followed by the input, serialized deterministically.

    ``role_input`` is any typed input with ``to_dict()``. The JSON contract of
    the output is added later by ``LlmClient.ask``. ``limits_note`` (and, with
    ``libraries``, the note about the allowed libraries) is appended to the
    instructions only when the input carries ``limits``, so an input without
    limits renders exactly as before.
    """
    text = instructions.strip()
    limits = getattr(role_input, "limits", None)
    if limits is not None and limits_note:
        text = f"{text}\n\n{limits_note.strip()}"
        if libraries and (extra := libraries_note(limits.allowed_libraries)):
            text = f"{text}\n{extra}"
        if describe:
            text = f"{text}\n{describe_note()}"
    for field, note in (("steering", STEERING_NOTE), ("carry", CARRY_NOTE), ("request", REQUEST_NOTE)):
        if getattr(role_input, field, None) is not None:
            text = f"{text}\n\n{note}"
    payload = json.dumps(role_input.to_dict(), sort_keys=True, ensure_ascii=False)
    return f"{text}\n\nInput (JSON):\n{payload}"
