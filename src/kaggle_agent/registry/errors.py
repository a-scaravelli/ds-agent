"""Errors of the registry (field validation reuses ``proposals.ValidationError``)."""


class RegistryError(Exception):
    """Raised when an operation breaks a registry invariant."""


class NotFoundError(RegistryError):
    """Raised when a cycle, attempt or suggestion does not exist."""


class DuplicateAttemptError(RegistryError):
    """Raised when a proposed attempt has the fingerprint of an earlier one.

    Carries ``existing_attempt_id`` and ``fingerprint`` so the loop can pass
    the reason for the rejection back to the model (design section 2).
    """

    def __init__(self, existing_attempt_id: str, fingerprint: str) -> None:
        self.existing_attempt_id = existing_attempt_id
        self.fingerprint = fingerprint
        super().__init__(
            f"duplicate of attempt {existing_attempt_id} (fingerprint {fingerprint})"
        )
