"""Typed errors of the bundle writer.

Every error here is a ``bundle.BundleError``, so the loop's existing mapping applies unchanged: a
rejection in the propose phase, and bounded fix retries followed by a non-retryable failed
execution in the fix phase. Errors of the Claude client (``LlmError``: rate limit, quota, CLI
failure) are NOT wrapped: they propagate as they are, so the cycle is suspended and resumable.
"""

from kaggle_agent.bundle import BundleError


class BundleWriterError(BundleError):
    """Base class of the errors of this package."""


class InvalidBundleError(BundleWriterError):
    """The bundle is still invalid after the repair turns.

    ``last_error`` is the last check error and ``turns`` the number of Claude calls made.
    """

    def __init__(self, last_error: str, turns: int) -> None:
        self.last_error = last_error
        self.turns = turns
        super().__init__(f"invalid bundle after {turns} turn(s); last check error: {last_error}")


class WorkspaceViolationError(BundleWriterError):
    """Claude wrote outside the staging workspace or changed a protected path."""


class PreviousBundleError(BundleWriterError):
    """The previous bundle of a fix request could not be read."""


class WriterConfigurationError(BundleWriterError):
    """The writer was wired in a way that would record the Claude calls twice."""
