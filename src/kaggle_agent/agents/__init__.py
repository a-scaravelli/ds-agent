"""The five role agents as typed prompts on ``LlmClient`` (design sections 8 and 14, level 1).

Roles: coordinator, data, feature engineering, modeling and reviewer. Each has a
typed input made of summaries (never labels, holdout data, credentials or the
registry), a typed output that reuses the proposal types where they exist, a
prompt builder and a ``run_<role>`` function. A call goes through
``LlmClient.ask`` with the role and the purpose, validates the structured
output, retries invalid output with the client's own mechanism, then rejects it
with ``RoleRejectedError``. Every call leaves a ``RoleTrace`` in the registry.

Rules:

- Roles do not call each other and do not import each other: the caller passes
  the output of one role to the next. The pipeline that does so is a later step.
- The choice of model stays with the caller (``model`` argument); no model name
  or role-to-model map lives here.
- The reviewer output has no field that decides: it is text only (never a gate).
- ``call`` is the only module that imports ``registry``. The roles with a
  candidate (feature engineering, modeling) return a ``PendingTrace`` that the
  caller writes with ``record_trace`` after registering the attempt.
"""

from .call import (
    PURPOSES,
    CandidateRoleResult,
    InvokedRole,
    PendingTrace,
    TraceContext,
    call_role,
    invoke_role,
    record_trace,
)
from .coordinator import (
    CoordinatorInput,
    CoordinatorOutput,
    build_coordinator_prompt,
    run_coordinator,
)
from .data import DataInput, build_data_prompt, run_data
from .errors import RoleConfigurationError, RoleRejectedError
from .feature_engineering import (
    FeatureEngineeringInput,
    build_feature_engineering_prompt,
    run_feature_engineering,
)
from .modeling import ModelingInput, build_modeling_prompt, run_modeling
from .prompting import render_prompt
from .reviewer import (
    CandidateResult,
    ReviewerInput,
    ReviewerNotes,
    build_reviewer_prompt,
    run_reviewer,
)
from .summaries import (
    AttemptDigest,
    ColumnSummary,
    DataFindings,
    DataSummary,
    MetricValue,
    RegistrySummary,
)

__all__ = [
    "AttemptDigest",
    "CandidateResult",
    "CandidateRoleResult",
    "ColumnSummary",
    "CoordinatorInput",
    "CoordinatorOutput",
    "DataFindings",
    "DataInput",
    "DataSummary",
    "FeatureEngineeringInput",
    "InvokedRole",
    "MetricValue",
    "ModelingInput",
    "PURPOSES",
    "PendingTrace",
    "RegistrySummary",
    "ReviewerInput",
    "ReviewerNotes",
    "RoleConfigurationError",
    "RoleRejectedError",
    "TraceContext",
    "build_coordinator_prompt",
    "build_data_prompt",
    "build_feature_engineering_prompt",
    "build_modeling_prompt",
    "build_reviewer_prompt",
    "call_role",
    "invoke_role",
    "record_trace",
    "render_prompt",
    "run_coordinator",
    "run_data",
    "run_feature_engineering",
    "run_modeling",
    "run_reviewer",
]
