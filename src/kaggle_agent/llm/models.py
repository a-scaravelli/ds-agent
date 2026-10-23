"""Data model of the language model adapter (design sections 8 and 11).

``ToolPermissions`` is the only, narrow interface the sandbox (ticket 6) will
need: which tools the CLI may use and in which directory it runs. The adapter
never widens it. No model name is written anywhere in this package: a model
of ``None`` means "the default of the CLI".
"""

from dataclasses import dataclass
from pathlib import Path

OUTCOME_OK = "ok"
OUTCOME_RATE_LIMITED = "rate_limited"
OUTCOME_QUOTA = "quota"
OUTCOME_CLI_ERROR = "cli_error"
OUTCOME_INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class ToolPermissions:
    """Tools the CLI may use and the directory it runs in, both given by the caller.

    ``allowed_tools`` is empty by default: no tool at all. ``working_dir`` must
    be an absolute path of an existing directory.
    """

    working_dir: Path
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.allowed_tools, str):
            raise ValueError("allowed_tools must be a sequence of tool names, not a string")
        tools = tuple(self.allowed_tools)
        for tool in tools:
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError("allowed_tools must hold non-empty strings")
        object.__setattr__(self, "allowed_tools", tools)
        path = Path(self.working_dir)
        if not path.is_absolute():
            raise ValueError("working_dir must be an absolute path")
        if not path.is_dir():
            raise ValueError("working_dir must be an existing directory")
        object.__setattr__(self, "working_dir", path)


@dataclass(frozen=True)
class Usage:
    """Tokens of one call; None when the CLI did not report them."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class RawResponse:
    """What an implementation returns for one successful CLI invocation."""

    text: str
    usage: Usage = Usage()
    session_id: str | None = None
    model: str | None = None
    stripped_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class CallRecord:
    """Consumption of one CLI invocation, successful or not.

    ``model`` is the model the CLI reported, else the one requested, else None.
    ``attempt`` counts invocations inside one ``ask`` (retries included).
    """

    purpose: str
    outcome: str
    started_at: str
    duration_seconds: float
    usage: Usage
    attempt: int
    model: str | None = None
    stripped_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class Consumption:
    """Totals over a set of calls. Tokens are None when no call reported them."""

    calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float = 0.0
    rate_limit_events: int = 0


def _sum_known(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def summarize(calls: "list[CallRecord] | tuple[CallRecord, ...]") -> Consumption:
    """Add up calls: count, known tokens, duration and rate limit events."""
    return Consumption(
        calls=len(calls),
        input_tokens=_sum_known([c.usage.input_tokens for c in calls]),
        output_tokens=_sum_known([c.usage.output_tokens for c in calls]),
        duration_seconds=sum(c.duration_seconds for c in calls),
        rate_limit_events=sum(1 for c in calls if c.outcome == OUTCOME_RATE_LIMITED),
    )
