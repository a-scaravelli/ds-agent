"""Errors of the sandbox subpackage."""


class SandboxError(Exception):
    """Base class for sandbox errors (the sandbox cannot run at all)."""


class SandboxConfigError(SandboxError):
    """The policy or the request is invalid: nothing was started."""
