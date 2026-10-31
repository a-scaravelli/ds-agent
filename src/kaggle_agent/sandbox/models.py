"""Data models of the light sandbox: policy, request and result (design section 8)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .environment import validate_extra_env
from .errors import SandboxConfigError

SUCCEEDED = "succeeded"
TIMED_OUT = "timed_out"
CRASHED = "crashed"
PROTECTED_MODIFIED = "protected_modified"
DIFF_REJECTED = "diff_rejected"
STATUSES = (SUCCEEDED, TIMED_OUT, CRASHED, PROTECTED_MODIFIED, DIFF_REJECTED)

# Literals on purpose: the sandbox does not import the registry. A test checks that every
# value is in ``registry.FAILURE_KINDS``.
FAILURE_KIND_BY_STATUS = {
    SUCCEEDED: None,
    TIMED_OUT: "timeout",
    CRASHED: "code_error",
    PROTECTED_MODIFIED: "code_error",
    DIFF_REJECTED: "code_error",
}

DEFAULT_MAX_OUTPUT_BYTES = 1_000_000


@dataclass(frozen=True)
class ProtectedPath:
    """A file or folder the generated code must not change.

    ``name`` is relative to the repository root (POSIX style) and is the key used in
    hashes and diffs. ``expose`` is True for what the child may read (it sees a
    read-only copy under ``SANDBOX_PROTECTED_DIR``) and False for what it must never
    see, such as the holdout: that one is hashed but never copied.
    """

    name: str
    path: Path
    expose: bool = True

    def __post_init__(self):
        if not self.name or self.name.startswith("/") or ".." in self.name.split("/"):
            raise SandboxConfigError(f"protected name must be relative to the root: {self.name!r}")
        object.__setattr__(self, "path", Path(self.path))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": str(self.path), "expose": self.expose}


@dataclass(frozen=True)
class SandboxPolicy:
    """What a run is allowed to do. The timeout is mandatory and has no default.

    ``hidden_paths`` lists what the child must never receive (credential files, the
    registry file, the holdout): the sandbox refuses arguments and variables that
    mention them and never copies them. It is not an OS barrier, see the README.
    """

    timeout_seconds: float
    protected: tuple[ProtectedPath, ...] = ()
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    hidden_paths: tuple[Path, ...] = ()

    def __post_init__(self):
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise SandboxConfigError("timeout_seconds must be a number")
        if not self.timeout_seconds > 0:
            raise SandboxConfigError("timeout_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise SandboxConfigError("max_output_bytes must be positive")
        names = [p.name for p in self.protected]
        if len(set(names)) != len(names):
            raise SandboxConfigError("duplicate protected names")
        object.__setattr__(self, "protected", tuple(self.protected))
        object.__setattr__(self, "hidden_paths", tuple(Path(p) for p in self.hidden_paths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "protected": [p.to_dict() for p in self.protected],
            "max_output_bytes": self.max_output_bytes,
            "hidden_paths": [str(p) for p in self.hidden_paths],
        }


@dataclass(frozen=True)
class SandboxRequest:
    """One run: either a ``command`` or a Python ``entry_point`` with ``args``, not both.

    ``diff`` is checked before anything starts. ``expected_hashes`` maps protected
    names to the frozen SHA-256 to compare before starting. ``extra_env`` is added to
    the minimal environment (secret-looking names and ``PYTHONPATH`` are rejected).
    """

    command: tuple[str, ...] | None = None
    entry_point: Path | None = None
    args: tuple[str, ...] = ()
    diff: str | None = None
    expected_hashes: Mapping[str, str] | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if (self.command is None) == (self.entry_point is None):
            raise SandboxConfigError("give exactly one of command and entry_point")
        if self.command is not None:
            command = tuple(str(c) for c in self.command)
            if not command:
                raise SandboxConfigError("command must not be empty")
            object.__setattr__(self, "command", command)
        if self.entry_point is not None:
            object.__setattr__(self, "entry_point", Path(self.entry_point))
        object.__setattr__(self, "args", tuple(str(a) for a in self.args))
        validate_extra_env(self.extra_env)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command) if self.command is not None else None,
            "entry_point": str(self.entry_point) if self.entry_point is not None else None,
            "args": list(self.args),
            "diff": self.diff,
            "expected_hashes": dict(self.expected_hashes) if self.expected_hashes else None,
            "extra_env": sorted(self.extra_env),
        }


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of a run. Failures are returned as results, never raised.

    ``status`` is one of ``STATUSES``. ``failure_kind`` maps it onto the registry
    vocabulary (``timed_out`` to ``timeout``; ``crashed``, ``protected_modified`` and
    ``diff_rejected`` to ``code_error``), because no new failure kind was added.

    Callers MUST check ``status``, not only ``failure_kind``. ``protected_modified``
    and ``diff_rejected`` map to ``code_error`` only for that reason; they are
    integrity failures and must NEVER be retried as a code fix. ``crashed`` is the only
    ``code_error`` status that is a candidate for fix retries.
    """

    status: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration_seconds: float = 0.0
    hashes_before: Mapping[str, str] = field(default_factory=dict)
    hashes_after: Mapping[str, str] = field(default_factory=dict)
    changed_paths: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise SandboxConfigError(f"status must be one of {', '.join(STATUSES)}")

    @property
    def failure_kind(self) -> str | None:
        """The registry failure kind for this status, or None when it succeeded."""
        return FAILURE_KIND_BY_STATUS[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_kind": self.failure_kind,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "duration_seconds": self.duration_seconds,
            "hashes_before": dict(self.hashes_before),
            "hashes_after": dict(self.hashes_after),
            "changed_paths": list(self.changed_paths),
            "reason": self.reason,
        }
