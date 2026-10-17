"""Errors of the evaluation engine; invalid data and protocols use ``ValidationError``."""


class EvaluationError(Exception):
    """Base class of the errors raised by the evaluation engine."""


class ProtocolNotFrozenError(EvaluationError):
    """The cycle has no frozen protocol hash: nothing can be evaluated yet."""


class ProtocolHashMismatchError(EvaluationError):
    """The protocol or the evaluation code differs from what was frozen: the run is blocked."""

    def __init__(self, frozen: str, current: str) -> None:
        self.frozen = frozen
        self.current = current
        super().__init__(f"protocol hash changed: frozen {frozen[:12]}, current {current[:12]}")


class HoldoutSessionError(EvaluationError):
    """A holdout read broke the session rules (not open, closed, reused or attempt already read)."""
