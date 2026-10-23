"""Environment of the CLI process: subscription only, never the paid API (design section 8).

Standard library only. ``os.environ`` is never modified.
"""

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Variables that move the billing of the CLI to a paid API or to a cloud provider.
PAID_API_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)

TOKEN_VARIABLE = "CLAUDE_CODE_OAUTH_TOKEN"


def build_cli_environment(env: Mapping[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    """A copy of ``env`` without the paid-API variables, and the sorted names removed.

    A WARNING names the removed variables, never their values.
    """
    removed = tuple(sorted(name for name in PAID_API_VARIABLES if name in env))
    cleaned = {k: v for k, v in env.items() if k not in PAID_API_VARIABLES}
    if removed:
        logger.warning(
            "removed %s from the environment of the Claude CLI: only the subscription is used",
            ", ".join(removed),
        )
    return cleaned, removed


def redact(text: str, env: Mapping[str, str]) -> str:
    """``text`` without the access token or the paid-API values found in ``env``."""
    for name in (TOKEN_VARIABLE, *PAID_API_VARIABLES):
        value = env.get(name)
        if value:
            text = text.replace(value, "[redacted]")
    return text
