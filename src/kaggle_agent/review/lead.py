"""The lead's side of the cycle: what the cycle asks and what the lead may answer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import require_text

CONTINUE = "continue"
GUIDANCE = "guidance"
PROTOCOL_CHANGE = "protocol_change"
_KINDS = (CONTINUE, GUIDANCE, PROTOCOL_CHANGE)


@dataclass(frozen=True)
class LeadContext:
    """What the lead is told when asked: the cycle, its state and a short text."""

    cycle_id: str
    state: str
    summary: str = ""
    round_id: str | None = None
    round_position: int | None = None


@dataclass(frozen=True)
class Approve:
    """The lead approves the plan: the protocol is frozen and the cycle goes on."""


@dataclass(frozen=True)
class Revise:
    """The lead asks for a new plan; ``text`` says what to change."""

    text: str

    def __post_init__(self) -> None:
        require_text("text", self.text)


@dataclass(frozen=True)
class Accept:
    """The lead accepts the final report; the cycle stops."""


@dataclass(frozen=True)
class LeadResponse:
    """The answer at AWAITING_GUIDANCE.

    ``continue`` lets the cycle go on without guidance, ``guidance`` carries text for the next
    round, ``protocol_change`` asks to change the frozen protocol: that is not guidance and the
    cycle refuses it (it needs a new plan approval).
    """

    kind: str
    text: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValidationError("kind", f"must be one of {', '.join(_KINDS)}")
        if self.kind == GUIDANCE:
            require_text("text", self.text)
        if not isinstance(self.text, str):
            raise ValidationError("text", "must be a string")


class LeadInput(ABC):
    """Where the lead's decisions come from (a simulated lead in tests).

    A decision method returns None when the lead has not answered: the cycle then waits and
    keeps its state. Nothing is ever approved or accepted by default.
    """

    @abstractmethod
    def plan_decision(self, context: LeadContext) -> Approve | Revise | None:
        """Approve the plan or ask for a revision."""

    @abstractmethod
    def guidance(self, context: LeadContext) -> LeadResponse | None:
        """Guidance, or the choice to continue, after a round."""

    @abstractmethod
    def review_decision(self, context: LeadContext) -> Accept | None:
        """Accept the final report."""

    @abstractmethod
    def notice(self, message: str) -> None:
        """A message from the cycle to the lead (for example a refused protocol change)."""
