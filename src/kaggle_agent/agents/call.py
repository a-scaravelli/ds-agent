"""The shared call helper of the roles: one model call, its ``llm_calls`` rows and its trace.

The only module of ``agents`` that imports ``registry``. A role call has two
parts, kept apart so a role with a candidate can link its trace to the
candidate:

- ``invoke_role`` asks the model through ``LlmClient.ask``, stores one
  ``llm_calls`` row per CLI invocation (retries included) and returns the
  validated value with a ``PendingTrace``. Nothing is written to
  ``role_traces`` yet.
- ``record_trace`` writes the trace, with the ``candidate_id`` the caller gives.

``call_role`` does both with no candidate. A failed or rejected call is always
traced at once, with no candidate and no output, and the error is raised: a
rejected call keeps its outcome but never the raw text of the model.

This helper stores the ``llm_calls`` rows itself, to learn the id the trace
links to. Do not also build the client with
``on_call=functools.partial(record_call, registry, ...)`` for the same
registry: the rows would be written twice. That wiring is refused with
``RoleConfigurationError``; other wrappers around ``record_call`` cannot be
detected. The choice of model stays with the caller (``model`` is passed
through to ``ask``); there is no budget and no filesystem access here.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from kaggle_agent.llm import (
    InvalidResponseError,
    LlmClient,
    LlmError,
    ToolPermissions,
    recording,
)
from kaggle_agent.proposals import Attempt
from kaggle_agent.registry import Registry, RoleTrace

from .errors import RoleConfigurationError, RoleRejectedError

PURPOSES = {
    "coordinator": "coordinator: plan and direct the round",
    "data": "data: profile the data and propose cleaning",
    "feature_engineering": "feature engineering: propose a candidate",
    "modeling": "modeling: propose a candidate model",
    "reviewer": "reviewer: interpret the round results",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TraceContext:
    """Where a role call belongs in the registry: the cycle and the round."""

    cycle_id: str
    round_id: str


@dataclass(frozen=True)
class PendingTrace:
    """A trace of a successful call, not written yet.

    ``output`` and ``output_type`` describe the final typed result. Commit it
    with ``record_trace``.
    """

    cycle_id: str
    role: str
    round_id: str
    input_type: str
    input: dict
    outcome: str
    llm_call_id: int | None
    output_type: str
    output: dict
    created_at: str


@dataclass(frozen=True)
class InvokedRole:
    """The validated result of a role call and the trace waiting to be written."""

    value: object
    trace: PendingTrace


@dataclass(frozen=True)
class CandidateRoleResult:
    """The attempt proposed by a role with a candidate and its trace, not written yet."""

    output: Attempt
    trace: PendingTrace


def record_trace(
    registry: Registry, pending: PendingTrace, candidate_id: str | None = None
) -> RoleTrace:
    """Write a pending trace; the registry checks that ``candidate_id`` is an attempt of the cycle."""
    trace = RoleTrace(
        cycle_id=pending.cycle_id,
        role=pending.role,
        round_id=pending.round_id,
        input_type=pending.input_type,
        input=pending.input,
        outcome=pending.outcome,
        created_at=pending.created_at,
        candidate_id=candidate_id,
        output_type=pending.output_type,
        output=pending.output,
        llm_call_id=pending.llm_call_id,
    )
    registry.add_role_trace(trace)
    return trace


def _check_wiring(client: LlmClient, registry: Registry) -> None:
    hook = client.on_call
    if (
        isinstance(hook, functools.partial)
        and hook.func is recording.record_call
        and hook.args
        and hook.args[0] is registry
    ):
        raise RoleConfigurationError(
            "the client already records its calls in this registry with record_call; "
            "the role helpers store the llm_calls rows themselves, so build the client "
            "without that on_call hook"
        )


def invoke_role(
    client: LlmClient,
    registry: Registry,
    context: TraceContext,
    role: str,
    purpose: str,
    prompt: str,
    output_type: type,
    role_input: object,
    permissions: ToolPermissions,
    model: str | None = None,
    finalize: Callable[[object], object] | None = None,
    now: Callable[[], datetime] = utc_now,
) -> InvokedRole:
    """Ask the model, store the ``llm_calls`` rows and return the value with a pending trace.

    ``finalize`` turns the validated model output into the final typed result
    (for example it sets an id assigned by fixed code); the trace holds the
    final result. On ``InvalidResponseError`` the trace is written at once and
    ``RoleRejectedError`` is raised; any other ``LlmError`` is traced and
    re-raised unchanged.
    """
    _check_wiring(client, registry)
    before = len(client.call_log())
    failure: LlmError | None = None
    value = None
    try:
        value = client.ask(
            prompt, output_type, permissions, purpose, model=model, role=role
        ).value
    except LlmError as exc:
        failure = exc
    records = client.call_log()[before:]
    last_id: int | None = None
    for record in records:
        entry = recording.entry_from_record(context.cycle_id, record)
        last_id = registry.add_llm_call(entry)
    outcome = records[-1].outcome if records else "cli_error"
    input_type = type(role_input).__name__
    if failure is None:
        final = finalize(value) if finalize is not None else value
        pending = PendingTrace(
            cycle_id=context.cycle_id,
            role=role,
            round_id=context.round_id,
            input_type=input_type,
            input=role_input.to_dict(),
            outcome=outcome,
            llm_call_id=last_id,
            output_type=type(final).__name__,
            output=final.to_dict(),
            created_at=now().isoformat(),
        )
        return InvokedRole(final, pending)
    registry.add_role_trace(
        RoleTrace(
            cycle_id=context.cycle_id,
            role=role,
            round_id=context.round_id,
            input_type=input_type,
            input=role_input.to_dict(),
            outcome=outcome,
            created_at=now().isoformat(),
            llm_call_id=last_id,
        )
    )
    if isinstance(failure, InvalidResponseError):
        reason = (records[-1].rejection_reason if records else None) or "invalid output"
        raise RoleRejectedError(role, failure.attempts, reason) from None
    raise failure


def call_role(
    client: LlmClient,
    registry: Registry,
    context: TraceContext,
    role: str,
    purpose: str,
    prompt: str,
    output_type: type,
    role_input: object,
    permissions: ToolPermissions,
    model: str | None = None,
    finalize: Callable[[object], object] | None = None,
    now: Callable[[], datetime] = utc_now,
) -> object:
    """``invoke_role`` plus ``record_trace`` with no candidate; returns the typed result."""
    invoked = invoke_role(
        client, registry, context, role, purpose, prompt, output_type, role_input,
        permissions, model=model, finalize=finalize, now=now,
    )
    record_trace(registry, invoked.trace)
    return invoked.value
