"""Errors of the bundle package."""

from kaggle_agent.sandbox import CRASHED, SandboxResult

from .contract import MALFORMED_PREDICTIONS


class BundleError(Exception):
    """Base class of the errors of this package (configuration and layout problems)."""


class BundleLayoutError(BundleError):
    """The bundle folder breaks the contract (missing entry point, escaping path, ...)."""


class BundleRunError(BundleError):
    """A run of the bundle did not produce valid predictions.

    ``status`` is a sandbox status (``timed_out``, ``crashed``, ``protected_modified``,
    ``diff_rejected``) or one of ``malformed_predictions`` and ``folder_changed``.
    ``failure_kind`` is from the registry vocabulary. ``retryable_as_code_fix`` is True
    only for ``crashed`` and ``malformed_predictions``: integrity failures
    (``protected_modified``, ``diff_rejected``, ``folder_changed``) are never retried.
    """

    RETRYABLE = (CRASHED, MALFORMED_PREDICTIONS)

    def __init__(self, status: str, failure_kind: str, reason: str,
                 result: SandboxResult | None = None) -> None:
        self.status = status
        self.failure_kind = failure_kind
        self.reason = reason
        self.result = result
        self.retryable_as_code_fix = status in self.RETRYABLE
        super().__init__(f"{status}: {reason}")

