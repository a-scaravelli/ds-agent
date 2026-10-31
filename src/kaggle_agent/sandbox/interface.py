"""The sandbox interface: small enough for a container implementation to replace it (level 3)."""

from abc import ABC, abstractmethod

from .models import SandboxPolicy, SandboxRequest, SandboxResult


class Sandbox(ABC):
    """Runs generated code under a policy and returns a ``SandboxResult``.

    Invariants every implementation keeps:

    - a request whose diff touches a protected file is rejected (``diff_rejected``) and
      nothing is started;
    - a protected-file hash that differs from the expected one, before or after the run,
      blocks the run (``protected_modified``), even if the child exits with 0;
    - failures of the child (timeout, crash) are returned as results, not raised; only an
      invalid policy or request, or a platform that cannot run, raises;
    - the child gets a minimal environment with no credentials and neither the holdout nor
      the registry file;
    - callers check ``result.status``: ``protected_modified`` and ``diff_rejected`` are never
      retried as code fixes (see ``SandboxResult``).

    ``SubprocessSandbox`` (level 1) starts the child with ``python -I``, which ignores
    ``PYTHONPATH`` and the user site, and the environment never carries import paths.
    The bundle runner (step 7) cannot rely on the environment for imports: it must pass
    every import path explicitly, for example through ``SANDBOX_PROTECTED_DIR`` or absolute
    paths in the arguments.
    """

    @abstractmethod
    def run(self, policy: SandboxPolicy, request: SandboxRequest) -> SandboxResult:
        """Run ``request`` under ``policy``."""
