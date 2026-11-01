"""Errors of the cycle loop."""


class LoopError(Exception):
    """Base class of the errors of this package."""


class TransitionError(LoopError):
    """A state or a transition the state machine does not allow."""


class CycleSuspended(LoopError):
    """The cycle cannot go on now (rate limit, quota, CLI failure, rejected coordinator).

    The registry is left as it was; running the cycle again continues from the saved state.
    """

    def __init__(self, reason: str, state: str | None = None) -> None:
        self.reason = reason
        self.state = state
        super().__init__(reason)
