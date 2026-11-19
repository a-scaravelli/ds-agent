"""Exit codes of the orchestrator in plain words (a test compares the keys with ``orchestrator.codes.ALL``).

The web cannot import the orchestrator, so the numbers are repeated here. Each value is
``(state, text)``: the state the cockpit shows and the sentence for the DS.
"""

CODE_TEXT = {
    0: ("finished", "The run finished. The cycle is done; read the report."),
    1: ("failed", "The run could not start or stopped with an error (for example a second run was already active)."),
    2: ("failed", "The run was refused: the configuration or the project does not match. Check the settings page."),
    10: ("waiting", "The run is waiting for you: it needs a decision before it can go on."),
    11: ("suspended", "The run was suspended (quota or rate limit, or the Claude login expired). Nothing was lost."),
    12: ("failed", "The lead file (lead.toml) is not valid. Fix it or answer from the page, then resume."),
    13: ("failed", "The run crashed unexpectedly. The registry is intact: press Resume to continue."),
    14: ("paused", "The run is paused at a safe checkpoint, as you asked. Press Resume to continue."),
    15: ("stopped", "The run was stopped by you. The registry is intact: press Resume to continue."),
    20: ("failed", "Submit: the review was not accepted yet."),
    21: ("failed", "Submit: the submission file is not valid."),
    22: ("failed", "Submit: no submissions left today."),
    23: ("finished", "Submit: sent, the score is still pending."),
    24: ("failed", "Submit: a previous send may have reached Kaggle; check Kaggle by hand."),
}

STATE_TEXT = {
    "idle": "Nothing is running.",
    "running": "The run is working.",
    "stopping": "The run is being stopped.",
    "waiting": "The run is waiting for you.",
    "suspended": "The run is suspended.",
    "finished": "The run finished.",
    "failed": "The run failed.",
    "paused": "The run is paused.",
    "stopped": "The run was stopped by you.",
}


def describe_code(code) -> tuple[str, str]:
    return CODE_TEXT.get(code, ("failed", f"The run ended with the unknown exit code {code}."))
