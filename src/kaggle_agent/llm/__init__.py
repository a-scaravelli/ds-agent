"""Headless Claude adapter: structured output, retries, consumption (design section 8).

The real adapter (``kaggle_agent.llm.real``) is not imported here, so the
package works without spawning processes and never touches the CLI on import.
"""

from .errors import (
    AUTH_MESSAGE,
    AuthenticationError,
    CliFailureError,
    InvalidResponseError,
    LlmError,
    QuotaExhaustedError,
    RateLimitError,
    is_authentication_reason,
)
from .fake import (
    CliCrash,
    FakeLlmClient,
    Malformed,
    ManualClock,
    QuotaExhausted,
    RateLimited,
    Reply,
    valid_attempt_json,
    valid_plan_json,
)
from .interface import BackoffPolicy, LlmClient, LlmResult
from .models import CallRecord, Consumption, RawResponse, ToolPermissions, Usage
from .recording import entry_from_record, record_call

__all__ = [
    "AUTH_MESSAGE",
    "AuthenticationError",
    "BackoffPolicy",
    "CallRecord",
    "CliCrash",
    "CliFailureError",
    "Consumption",
    "FakeLlmClient",
    "InvalidResponseError",
    "LlmClient",
    "LlmError",
    "LlmResult",
    "Malformed",
    "ManualClock",
    "QuotaExhausted",
    "QuotaExhaustedError",
    "RateLimitError",
    "RateLimited",
    "RawResponse",
    "Reply",
    "ToolPermissions",
    "Usage",
    "entry_from_record",
    "is_authentication_reason",
    "record_call",
    "valid_attempt_json",
    "valid_plan_json",
]
