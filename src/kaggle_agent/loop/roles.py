"""The five roles as plain callables; the pipeline passes outputs between them.

A ``RoleSet`` holds one callable per role. Each takes the arguments of the matching
``agents.run_*`` function minus the injected client, registry, permissions, model and clock:

- ``coordinator(context, role_input) -> CoordinatorOutput``
- ``data(context, role_input) -> DataFindings``
- ``feature_engineering(context, role_input) -> CandidateRoleResult``
- ``modeling(context, role_input) -> CandidateRoleResult``
- ``reviewer(context, role_input, report_id) -> ReviewerReport``

``context`` is the ``TraceContext`` (cycle and round). Roles never call each other.
``client_roles`` wraps the real ``run_*`` functions around an ``LlmClient``; tests use it with a
``FakeLlmClient`` or build a ``RoleSet`` of plain fake callables.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from kaggle_agent.agents import (
    run_coordinator,
    run_data,
    run_feature_engineering,
    run_modeling,
    run_reviewer,
)
from kaggle_agent.llm import LlmClient, ToolPermissions
from kaggle_agent.registry import Registry


@dataclass(frozen=True)
class RoleSet:
    """One callable per role (see the module docstring for the signatures)."""

    coordinator: Callable
    data: Callable
    feature_engineering: Callable
    modeling: Callable
    reviewer: Callable


def client_roles(
    client: LlmClient,
    registry: Registry,
    permissions: ToolPermissions,
    model: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> RoleSet:
    """The real role functions bound to a client and a registry."""
    extra = {"model": model}
    if now is not None:
        extra["now"] = now
    return RoleSet(
        coordinator=lambda ctx, inp: run_coordinator(client, registry, ctx, inp, permissions, **extra),
        data=lambda ctx, inp: run_data(client, registry, ctx, inp, permissions, **extra),
        feature_engineering=lambda ctx, inp: run_feature_engineering(
            client, registry, ctx, inp, permissions, **extra
        ),
        modeling=lambda ctx, inp: run_modeling(client, registry, ctx, inp, permissions, **extra),
        reviewer=lambda ctx, inp, report_id: run_reviewer(
            client, registry, ctx, inp, permissions, report_id, **extra
        ),
    )
