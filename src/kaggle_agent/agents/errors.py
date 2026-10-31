"""Errors of the role agents.

``RoleRejectedError`` is an ``LlmError``, so a caller that already handles the
adapter's errors handles it too. Its message is built only from the role, the
number of attempts and the sanitized rejection reason: never from the text of
the model.
"""

from kaggle_agent.llm import LlmError


class RoleRejectedError(LlmError):
    """The model kept returning invalid output for a role after the bounded retries.

    ``reason`` is the sanitized 'field: reason' recorded by the client for the
    last rejected reply.
    """

    def __init__(self, role: str, attempts: int, reason: str) -> None:
        self.role = role
        self.attempts = attempts
        self.reason = reason
        super().__init__(
            f"role {role!r} produced no valid output after {attempts} attempts; "
            f"last rejection: {reason}"
        )


class RoleConfigurationError(Exception):
    """The role helpers were wired in a way that would corrupt the registry.

    Not an ``LlmError``: it is a programming error of the caller, raised before
    any model call.
    """
