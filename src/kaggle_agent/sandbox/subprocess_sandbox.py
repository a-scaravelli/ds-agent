"""Level 1 sandbox: a child process with reduced permissions (design sections 8 and 16).

Isolation is partial: the child runs as the same OS user, see the README. POSIX only.
Standard library only; this is the only sandbox module that imports ``subprocess``.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .environment import build_child_environment
from .errors import SandboxConfigError, SandboxError
from .interface import Sandbox
from .models import (
    CRASHED,
    DIFF_REJECTED,
    PROTECTED_MODIFIED,
    SUCCEEDED,
    TIMED_OUT,
    SandboxPolicy,
    SandboxRequest,
    SandboxResult,
)
from .protected import (
    COPY_PREFIX,
    diff_touched_paths,
    hash_paths,
    make_readonly_copies,
    make_writable,
)

STDERR_TAIL_CHARS = 500
READER_JOIN_SECONDS = 5.0


class _Capture:
    """Reads a pipe to the end, keeping at most ``limit`` bytes."""

    def __init__(self, stream, limit: int):
        self.data = bytearray()
        self.truncated = False
        self._stream = stream
        self._limit = limit
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while chunk := self._stream.read(65536):
                room = self._limit - len(self.data)
                if room > 0:
                    self.data.extend(chunk[:room])
                if len(chunk) > room:
                    self.truncated = True
        except (OSError, ValueError):
            pass

    def finish(self) -> str:
        self.thread.join(READER_JOIN_SECONDS)
        try:
            self._stream.close()
        except OSError:
            pass
        return bytes(self.data).decode("utf-8", errors="replace")


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class SubprocessSandbox(Sandbox):
    """Runs the request in a child process in its own process group.

    Flow: diff check, hash before (and comparison with ``expected_hashes``), run folder with
    an empty ``work/`` and read-only copies in ``protected/``, child with a minimal
    environment and a timeout, group kill, hash after on originals and copies, cleanup.
    """

    def run(self, policy: SandboxPolicy, request: SandboxRequest) -> SandboxResult:
        if os.name != "posix":
            raise SandboxError("SubprocessSandbox needs a POSIX system")
        argv = self._argv(request)
        self._check_hidden(policy, argv, request)

        if request.diff:
            touched = diff_touched_paths(request.diff, policy.protected)
            if touched:
                return SandboxResult(
                    status=DIFF_REJECTED,
                    changed_paths=touched,
                    reason="diff touches protected files: " + ", ".join(touched),
                )

        before = hash_paths(policy.protected)
        expected = dict(request.expected_hashes or {})
        unknown = sorted(set(expected) - set(before))
        if unknown:
            raise SandboxConfigError("expected_hashes names are not protected: " + ", ".join(unknown))
        differing = tuple(sorted(n for n, h in expected.items() if before[n] != h))
        if differing:
            return SandboxResult(
                status=PROTECTED_MODIFIED,
                hashes_before=before,
                changed_paths=differing,
                reason="protected files changed before the run: " + ", ".join(differing),
            )

        run_dir = Path(tempfile.mkdtemp(prefix="sandbox-"))
        try:
            return self._run_child(policy, request, argv, run_dir, before)
        finally:
            make_writable(run_dir)
            shutil.rmtree(run_dir, ignore_errors=True)

    @staticmethod
    def _argv(request: SandboxRequest) -> list[str]:
        if request.command is not None:
            return [*request.command, *request.args]
        entry = os.path.abspath(request.entry_point)
        return [sys.executable, "-I", "-B", entry, *request.args]

    @staticmethod
    def _check_hidden(policy: SandboxPolicy, argv: list[str], request: SandboxRequest) -> None:
        texts = [*argv, *request.extra_env.values()]
        for hidden in policy.hidden_paths:
            for text in texts:
                if str(hidden) and str(hidden) in text:
                    raise SandboxConfigError(f"the child must not receive hidden path {hidden.name}")

    def _run_child(self, policy, request, argv, run_dir: Path, before) -> SandboxResult:
        work = run_dir / "work"
        for name in ("work", "home", "tmp", "protected"):
            (run_dir / name).mkdir()
        copies = make_readonly_copies(policy.protected, run_dir / "protected")
        copies_before = hash_paths(copies)
        env = build_child_environment(run_dir, request.extra_env)

        started = time.monotonic()
        proc = subprocess.Popen(
            argv,
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            umask=0o077,
        )
        out = _Capture(proc.stdout, policy.max_output_bytes)
        err = _Capture(proc.stderr, policy.max_output_bytes)
        timed_out = False
        try:
            proc.wait(timeout=policy.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc.pid)
            proc.wait()
        _kill_group(proc.pid)  # leftover descendants of a normal exit
        stdout, stderr = out.finish(), err.finish()
        duration = time.monotonic() - started

        after = hash_paths(policy.protected, missing_ok=True)
        copies_after = hash_paths(copies, missing_ok=True)
        changed = [n for n in before if before[n] != after[n]]
        changed += [COPY_PREFIX + n for n in copies_before if copies_before[n] != copies_after[n]]

        code = None if timed_out else proc.returncode
        if changed:
            status = PROTECTED_MODIFIED
            reason = "protected files changed during the run: " + ", ".join(changed)
            if timed_out:
                reason += f" (the run also timed out after {policy.timeout_seconds} s)"
        elif timed_out:
            status = TIMED_OUT
            reason = f"timed out after {policy.timeout_seconds} s; the process group was killed"
        elif proc.returncode != 0:
            status = CRASHED
            if proc.returncode < 0:
                reason = f"killed by signal {-proc.returncode}"
            else:
                reason = f"exit code {proc.returncode}"
            tail = stderr.strip()[-STDERR_TAIL_CHARS:]
            if tail:
                reason += f": {tail}"
        else:
            status, reason = SUCCEEDED, ""
        return SandboxResult(
            status=status,
            exit_code=code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=out.truncated,
            stderr_truncated=err.truncated,
            duration_seconds=duration,
            hashes_before=before,
            hashes_after=after,
            changed_paths=tuple(changed),
            reason=reason,
        )
