"""Deterministic fake of the Claude client (design section 8). No process, no network, no I/O.

The script is a sequence of steps, one per CLI invocation. The fake keeps its
own manual clock, so durations and backoff sleeps are exact and never wait.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .errors import CliFailureError, QuotaExhaustedError, RateLimitError
from .interface import BackoffPolicy, LlmClient
from .models import RawResponse, ToolPermissions, Usage

_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


class FakeScriptExhaustedError(CliFailureError):
    """The fake was called more times than its script has steps."""


@dataclass(frozen=True)
class Reply:
    """The CLI answers with ``text``."""

    text: str
    usage: Usage = Usage()
    duration: float = 0.0
    model: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class Malformed:
    """The CLI answers with text that is not JSON."""

    text: str = "this is not json"
    usage: Usage = Usage()
    duration: float = 0.0


@dataclass(frozen=True)
class RateLimited:
    """The CLI reports a rate limit."""

    retry_after: float | None = None
    duration: float = 0.0


@dataclass(frozen=True)
class QuotaExhausted:
    """The CLI reports that the subscription quota is used up."""

    resets_at: str | None = None
    duration: float = 0.0


@dataclass(frozen=True)
class CliCrash:
    """The CLI fails (non-zero exit, unreadable output, ...)."""

    reason: str = "simulated CLI failure"
    duration: float = 0.0


@dataclass(frozen=True)
class FakeCall:
    """What the fake was asked: for assertions."""

    prompt: str
    permissions: ToolPermissions
    model: str | None


@dataclass
class ManualClock:
    """A clock that moves only when told to; ``sleep`` advances it and is recorded."""

    time: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def now(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.time += seconds


class FakeLlmClient(LlmClient):
    """Plays a script of steps; ``calls`` records every invocation."""

    def __init__(
        self,
        script: Sequence[object],
        model: str | None = None,
        clock: ManualClock | None = None,
        **kwargs,
    ) -> None:
        self.manual_clock = clock or ManualClock()
        kwargs.setdefault("backoff", BackoffPolicy())
        super().__init__(
            model=model,
            clock=self.manual_clock.now,
            sleep=self.manual_clock.sleep,
            wall_clock=lambda: _EPOCH + timedelta(seconds=self.manual_clock.now()),
            **kwargs,
        )
        self._script = list(script)
        self.calls: list[FakeCall] = []

    @property
    def sleeps(self) -> list[float]:
        return self.manual_clock.sleeps

    def _run(self, prompt: str, permissions: ToolPermissions, model: str | None) -> RawResponse:
        self.calls.append(FakeCall(prompt, permissions, model))
        index = len(self.calls) - 1
        if index >= len(self._script):
            raise FakeScriptExhaustedError(
                f"fake script exhausted: call {index + 1} but only {len(self._script)} steps"
            )
        step = self._script[index]
        self.manual_clock.advance(getattr(step, "duration", 0.0))
        if isinstance(step, RateLimited):
            raise RateLimitError("rate limited (simulated)", retry_after=step.retry_after)
        if isinstance(step, QuotaExhausted):
            raise QuotaExhaustedError("quota exhausted (simulated)", resets_at=step.resets_at)
        if isinstance(step, CliCrash):
            raise CliFailureError(step.reason)
        if isinstance(step, (Reply, Malformed)):
            return RawResponse(
                text=step.text,
                usage=step.usage,
                session_id=getattr(step, "session_id", None),
                model=getattr(step, "model", None),
            )
        raise TypeError(f"unknown fake step: {step!r}")


def valid_plan_json(**overrides) -> str:
    """A valid ``ShortPlan`` as JSON text."""
    data = {
        "metric": "accuracy",
        "validation": "5-fold stratified",
        "feature_engineering": ["fill missing values"],
        "models": ["logistic regression"],
        "baseline": "majority class",
        **overrides,
    }
    return json.dumps(data)


def valid_attempt_json(**overrides) -> str:
    """A valid ``Attempt`` as JSON text."""
    data = {
        "id": "A1",
        "phase": "iteration",
        "hypothesis": "scaling helps",
        "rationale": "features have different ranges",
        "what_changed": "add standard scaling",
        "measure": "cross-validated accuracy",
        "exploratory": True,
        **overrides,
    }
    return json.dumps(data)
