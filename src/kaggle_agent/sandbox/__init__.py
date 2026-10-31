"""Isolated execution of model-generated code (design sections 8 and 16).

Level 1 is a light sandbox: a subprocess with a timeout, a minimal environment and a
SHA-256 hash of the protected files. Isolation is partial, see the README.
"""

from .environment import MINIMAL_KEYS, PROTECTED_DIR_VARIABLE, build_child_environment
from .errors import SandboxConfigError, SandboxError
from .interface import Sandbox
from .models import (
    CRASHED,
    DIFF_REJECTED,
    FAILURE_KIND_BY_STATUS,
    PROTECTED_MODIFIED,
    STATUSES,
    SUCCEEDED,
    TIMED_OUT,
    ProtectedPath,
    SandboxPolicy,
    SandboxRequest,
    SandboxResult,
)
from .protected import diff_touched_paths, hash_paths
from .subprocess_sandbox import SubprocessSandbox

__all__ = [
    "CRASHED",
    "DIFF_REJECTED",
    "FAILURE_KIND_BY_STATUS",
    "MINIMAL_KEYS",
    "PROTECTED_DIR_VARIABLE",
    "PROTECTED_MODIFIED",
    "ProtectedPath",
    "STATUSES",
    "SUCCEEDED",
    "Sandbox",
    "SandboxConfigError",
    "SandboxError",
    "SandboxPolicy",
    "SandboxRequest",
    "SandboxResult",
    "SubprocessSandbox",
    "TIMED_OUT",
    "build_child_environment",
    "diff_touched_paths",
    "hash_paths",
]
