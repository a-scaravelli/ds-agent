"""Errors of the language model adapter. No message may contain raw CLI output or a token.

Callers of ``LlmClient.ask`` see only subclasses of ``LlmError``.
"""


class LlmError(Exception):
    """Base class of every error raised by the adapter."""


class RateLimitError(LlmError):
    """The CLI reported a rate limit and the bounded backoff is exhausted.

    ``retry_after`` is the wait suggested by the CLI, in seconds, when known.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class QuotaExhaustedError(LlmError):
    """The subscription quota is used up; waiting a few seconds does not help.

    ``resets_at`` is the reset time (ISO 8601, UTC) when the CLI reports it.
    """

    def __init__(self, message: str, resets_at: str | None = None) -> None:
        self.resets_at = resets_at
        super().__init__(message)


class CliFailureError(LlmError):
    """The CLI could not be run or failed (exit code, timeout, unreadable output)."""


class InvalidResponseError(LlmError):
    """The model kept returning unusable output after the bounded retries."""

    def __init__(self, last_error: str, attempts: int) -> None:
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(
            f"no valid structured response after {attempts} attempts; last error: {last_error}"
        )
