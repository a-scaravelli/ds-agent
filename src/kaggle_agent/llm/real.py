"""Real adapter: runs ``claude -p`` with the user's subscription (design section 8).

The only module of the package that imports ``subprocess``. Pytest never runs
it against the real CLI: tests inject ``runner``. The format of the envelope
and of the errors follows the verification recorded in design section 8.
"""

import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from .environment import build_cli_environment, redact
from .errors import CliFailureError, QuotaExhaustedError, RateLimitError
from .interface import LlmClient
from .models import RawResponse, ToolPermissions, Usage

DEFAULT_TIMEOUT_SECONDS = 900.0

_QUOTA = re.compile(
    r"usage limit|hit your limit|limit reached|quota|weekly limit|5-hour limit|session limit",
    re.IGNORECASE,
)
_RATE_LIMIT = re.compile(
    r"rate.?limit|too many requests|\b429\b|overloaded|\b529\b", re.IGNORECASE
)
_RESET_EPOCH = re.compile(r"\|\s*(\d{9,11})\b")
_RETRY_AFTER = re.compile(r"retry (?:in|after) (\d+(?:\.\d+)?) ?(?:s|sec|seconds)\b", re.IGNORECASE)
_NOT_LOGGED_IN = re.compile(r"not logged in|please run /login|invalid api key", re.IGNORECASE)
_SAFE_WORD = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def _word(value: object) -> str | None:
    """A short machine word from the envelope, or None: never free text."""
    return value if isinstance(value, str) and _SAFE_WORD.match(value) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(envelope: Mapping) -> Usage:
    usage = envelope.get("usage")
    if not isinstance(usage, Mapping):
        return Usage()
    inputs = [
        _int_or_none(usage.get(k))
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    ]
    known = [v for v in inputs if v is not None]
    return Usage(
        input_tokens=sum(known) if known else None,
        output_tokens=_int_or_none(usage.get("output_tokens")),
    )


def _reported_model(envelope: Mapping) -> str | None:
    """The model the CLI says it used: the busiest key of ``modelUsage``, else ``model``."""
    usage = envelope.get("modelUsage")
    if isinstance(usage, Mapping) and usage:

        def weight(item: tuple) -> int:
            stats = item[1] if isinstance(item[1], Mapping) else {}
            return (_int_or_none(stats.get("inputTokens")) or 0) + (
                _int_or_none(stats.get("outputTokens")) or 0
            )

        return str(max(usage.items(), key=weight)[0])
    model = envelope.get("model")
    return model if isinstance(model, str) and model else None


class RealClaudeClient(LlmClient):
    """Runs the Claude Code CLI non-interactively; never receives a paid-API key."""

    def __init__(
        self,
        runner: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
        env: Mapping[str, str] | None = None,
        executable: str = "claude",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        model: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        **kwargs,
    ) -> None:
        super().__init__(model=model, clock=clock, sleep=sleep, **kwargs)
        self._runner = runner
        self._env = env if env is not None else os.environ
        self._executable = executable
        self._timeout = timeout_seconds

    def build_argv(self, permissions: ToolPermissions, model: str | None) -> list[str]:
        """Command line: tools come only from ``permissions``; no permission bypass."""
        argv = [
            self._executable,
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
        ]
        if model is not None:
            argv += ["--model", model]
        tools = list(permissions.allowed_tools)
        argv += ["--tools", *(tools or [""])]
        if tools:
            argv += ["--allowedTools", *tools]
        return argv

    def _run(self, prompt: str, permissions: ToolPermissions, model: str | None) -> RawResponse:
        env, stripped = build_cli_environment(self._env)
        argv = self.build_argv(permissions, model)
        try:
            proc = self._runner(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=str(permissions.working_dir),
                env=env,
                timeout=self._timeout,
                check=False,
            )
        except FileNotFoundError:
            raise CliFailureError(
                f"the Claude CLI executable '{self._executable}' was not found"
            ) from None
        except subprocess.TimeoutExpired:
            raise CliFailureError(f"the Claude CLI timed out after {self._timeout} seconds") from None
        except OSError as exc:
            raise CliFailureError(f"the Claude CLI could not be started ({type(exc).__name__})") from None
        return self._interpret(proc, env, stripped)

    def _interpret(self, proc, env: Mapping[str, str], stripped: tuple[str, ...]) -> RawResponse:
        stdout = proc.stdout if isinstance(proc.stdout, str) else ""
        stderr = proc.stderr if isinstance(proc.stderr, str) else ""
        code = proc.returncode
        try:
            envelope = json.loads(stdout)
        except ValueError:
            envelope = None
        if not isinstance(envelope, dict):
            self._classify(redact(f"{stdout}\n{stderr}", self._env), None)
            if code != 0:
                raise CliFailureError(f"the Claude CLI exited with code {code}")
            raise CliFailureError("the Claude CLI output is not a readable JSON envelope")
        result = envelope.get("result")
        failed = code != 0 or envelope.get("is_error") is True
        if failed:
            text = redact(f"{result if isinstance(result, str) else ''}\n{stderr}", self._env)
            self._classify(text, envelope)
            if _NOT_LOGGED_IN.search(text):
                raise CliFailureError(
                    "the Claude CLI is not authenticated: log it in or set CLAUDE_CODE_OAUTH_TOKEN"
                )
            reason = _word(envelope.get("terminal_reason")) or _word(envelope.get("subtype"))
            detail = f", reason {reason}" if reason else ""
            raise CliFailureError(f"the Claude CLI reported an error (exit code {code}{detail})")
        if not isinstance(result, str):
            raise CliFailureError("the Claude CLI envelope has no text result")
        session = envelope.get("session_id")
        return RawResponse(
            text=result,
            usage=_usage(envelope),
            session_id=session if isinstance(session, str) else None,
            model=_reported_model(envelope),
            stripped_variables=stripped,
        )

    @staticmethod
    def _classify(text: str, envelope: Mapping | None) -> None:
        """Raise the typed error for a quota or rate-limit report; return otherwise."""
        if _QUOTA.search(text):
            resets_at = None
            match = _RESET_EPOCH.search(text)
            if match:
                resets_at = datetime.fromtimestamp(int(match.group(1)), timezone.utc).isoformat()
            raise QuotaExhaustedError(
                "the subscription quota of the Claude CLI is exhausted", resets_at=resets_at
            )
        status = envelope.get("api_error_status") if envelope else None
        if status in (429, 529) or _RATE_LIMIT.search(text):
            match = _RETRY_AFTER.search(text)
            raise RateLimitError(
                "the Claude CLI reported a rate limit",
                retry_after=float(match.group(1)) if match else None,
            )
