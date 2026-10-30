"""Replaceable interface of the headless Claude client (design sections 8 and 11).

Invariants every implementation keeps:

- Subscription only: the CLI process never receives a paid-API variable
  (``ANTHROPIC_API_KEY`` and the like). The access token is never logged,
  stored in the registry, put in a result or an exception message, or passed
  to anything but the CLI process.
- ``ask`` is a template method: it ALWAYS adds the JSON contract, validates
  the answer with the proposal types, retries a bounded number of times and
  records the consumption, and only calls the ``_run`` hook to talk to the
  model. An implementation overrides ``_run`` and cannot skip the checks.
- ``_run`` may raise ``RateLimitError``, ``QuotaExhaustedError`` or
  ``CliFailureError``. Any other exception is mapped to ``CliFailureError``,
  so the caller sees only ``LlmError``.
- A rate limit is retried with a bounded backoff (``BackoffPolicy``) using the
  injected ``sleep``; an exhausted quota is raised at once, without waiting.
- A malformed or invalid answer is retried at most ``max_validation_retries``
  times, passing the validation error back with the original prompt (stateless,
  no session resume), then ``InvalidResponseError`` is raised. The reason
  recorded for a rejected answer is sanitized: field and reason of the
  ``ValidationError``, no text of the model and no secret.
- Tools and working directory come only from the ``ToolPermissions`` the
  caller passes; the adapter never widens them.
- The model is ``None`` (default of the CLI) unless the client or the call
  sets it; the effective model of every call is recorded. The model per role
  is the choice of the caller: ``role`` is only recorded, never mapped to a model.
- ``role`` is optional and, when given, one of ``ROLES`` (checked before any
  invocation); it is copied into every ``CallRecord`` of the call, retries included.
- Consumption is only reported (``CallRecord`` per CLI invocation, in
  ``consumption()``, in ``LlmResult`` and through ``on_call``). The client never
  writes to the registry and enforces no budget.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from kaggle_agent.proposals import ValidationError, validate_role

from .errors import (
    CliFailureError,
    InvalidResponseError,
    LlmError,
    QuotaExhaustedError,
    RateLimitError,
)
from .models import (
    OUTCOME_CLI_ERROR,
    OUTCOME_INVALID_OUTPUT,
    OUTCOME_OK,
    OUTCOME_QUOTA,
    OUTCOME_RATE_LIMITED,
    CallRecord,
    Consumption,
    RawResponse,
    ToolPermissions,
    Usage,
    summarize,
)
from .structured import json_contract, parse_structured, sanitize_rejection


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BackoffPolicy:
    """Bounded exponential backoff for rate limits. Final values belong to the budget ticket."""

    max_retries: int = 3
    base: float = 5.0
    factor: float = 2.0
    max_delay: float = 120.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.base < 0 or self.max_delay < 0:
            raise ValueError("base and max_delay must not be negative")
        if self.factor < 1:
            raise ValueError("factor must be at least 1")

    def delay(self, retry_index: int, retry_after: float | None) -> float:
        """Seconds to wait before retry number ``retry_index`` (from 0), capped at ``max_delay``."""
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.max_delay)
        return min(self.base * self.factor**retry_index, self.max_delay)


@dataclass(frozen=True)
class LlmResult:
    """A validated value with the consumption of the calls that produced it."""

    value: object
    calls: tuple[CallRecord, ...]
    consumption: Consumption
    stripped_variables: tuple[str, ...] = ()


class LlmClient(ABC):
    """Asks the model for structured output and reports what it consumed."""

    def __init__(
        self,
        model: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], datetime] = _utc_now,
        backoff: BackoffPolicy = BackoffPolicy(),
        max_validation_retries: int = 2,
        on_call: Callable[[CallRecord], None] | None = None,
    ) -> None:
        if max_validation_retries < 0:
            raise ValueError("max_validation_retries must not be negative")
        self._model = model
        self._clock = clock
        self._sleep = sleep
        self._wall_clock = wall_clock
        self._backoff = backoff
        self._max_validation_retries = max_validation_retries
        self._on_call = on_call
        self._log: list[CallRecord] = []

    @abstractmethod
    def _run(self, prompt: str, permissions: ToolPermissions, model: str | None) -> RawResponse:
        """One CLI invocation. Called only by ``ask``."""

    def consumption(self) -> Consumption:
        """Totals over every call made by this client, retries and failures included."""
        return summarize(self._log)

    def call_log(self) -> tuple[CallRecord, ...]:
        """Every call made by this client, in order."""
        return tuple(self._log)

    def ask(
        self,
        prompt: str,
        output_type: type,
        permissions: ToolPermissions,
        purpose: str,
        model: str | None = None,
        role: str | None = None,
    ) -> LlmResult:
        """Ask for an ``output_type`` and return it validated.

        ``model`` overrides the client's for this call; ``None`` keeps it.
        ``role`` is recorded in the call records. Once the role is accepted,
        raises only ``LlmError`` subclasses; an unknown role raises
        ``ValidationError`` before any invocation.
        """
        if role is not None:
            validate_role(role)
        requested = model if model is not None else self._model
        full_prompt = f"{prompt}\n\n{json_contract(output_type)}"
        current = full_prompt
        calls: list[CallRecord] = []
        rate_retries = 0
        validation_retries = 0
        while True:
            started_at = self._wall_clock().isoformat()
            started = self._clock()
            raw: RawResponse | None = None
            failure: LlmError | None = None
            try:
                raw = self._run(current, permissions, requested)
            except LlmError as exc:
                failure = exc
            except Exception as exc:  # an implementation bug must not crash the caller
                failure = CliFailureError(
                    f"unexpected {type(exc).__name__} while running the CLI"
                )
            duration = max(self._clock() - started, 0.0)

            def record(outcome: str, rejection_reason: str | None = None) -> None:
                reported = raw.model if raw is not None else None
                call = CallRecord(
                    purpose=purpose,
                    outcome=outcome,
                    started_at=started_at,
                    duration_seconds=duration,
                    usage=raw.usage if raw is not None else Usage(),
                    attempt=len(calls) + 1,
                    model=reported or requested,
                    stripped_variables=raw.stripped_variables if raw is not None else (),
                    rejection_reason=rejection_reason,
                    role=role,
                )
                calls.append(call)
                self._log.append(call)
                if self._on_call is not None:
                    self._on_call(call)

            if isinstance(failure, RateLimitError):
                record(OUTCOME_RATE_LIMITED)
                if rate_retries >= self._backoff.max_retries:
                    raise failure
                self._sleep(self._backoff.delay(rate_retries, failure.retry_after))
                rate_retries += 1
                continue
            if isinstance(failure, QuotaExhaustedError):
                record(OUTCOME_QUOTA)
                raise failure
            if failure is not None:
                record(OUTCOME_CLI_ERROR)
                raise failure

            assert raw is not None
            try:
                value = parse_structured(raw.text, output_type)
            except ValidationError as exc:
                record(OUTCOME_INVALID_OUTPUT, sanitize_rejection(exc, output_type))
                if validation_retries >= self._max_validation_retries:
                    raise InvalidResponseError(str(exc), validation_retries + 1) from None
                validation_retries += 1
                current = (
                    f"{full_prompt}\n\nYour previous reply was rejected: {exc}\n"
                    "Reply again with only the corrected JSON object."
                )
                continue
            record(OUTCOME_OK)
            return LlmResult(
                value=value,
                calls=tuple(calls),
                consumption=summarize(calls),
                stripped_variables=raw.stripped_variables,
            )
