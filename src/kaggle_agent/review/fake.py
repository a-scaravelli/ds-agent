"""A scripted lead for tests: each decision is read from a list, in order."""

from collections.abc import Iterable

from .lead import Accept, Approve, LeadContext, LeadInput, LeadResponse, Revise


class SimulatedLead(LeadInput):
    """Plays scripted answers; ``None`` in a script means "no answer yet".

    An exhausted script answers the default (approve, continue, accept) unless ``exhausted`` is
    ``"silent"``, in which case it answers None. Every question and notice is recorded.
    """

    def __init__(
        self,
        plans: Iterable = (),
        guidance: Iterable = (),
        reviews: Iterable = (),
        exhausted: str = "default",
    ) -> None:
        if exhausted not in ("default", "silent"):
            raise ValueError("exhausted must be 'default' or 'silent'")
        self._plans = list(plans)
        self._guidance = list(guidance)
        self._reviews = list(reviews)
        self._exhausted = exhausted
        self.questions: list[tuple[str, LeadContext]] = []
        self.notices: list[str] = []

    def _next(self, script: list, default):
        if script:
            return script.pop(0)
        return None if self._exhausted == "silent" else default

    def plan_decision(self, context: LeadContext) -> Approve | Revise | None:
        self.questions.append(("plan", context))
        return self._next(self._plans, Approve())

    def guidance(self, context: LeadContext) -> LeadResponse | None:
        self.questions.append(("guidance", context))
        return self._next(self._guidance, LeadResponse("continue"))

    def review_decision(self, context: LeadContext) -> Accept | None:
        self.questions.append(("review", context))
        return self._next(self._reviews, Accept())

    def notice(self, message: str) -> None:
        self.notices.append(message)

    def asked(self, kind: str) -> list[LeadContext]:
        """The contexts of the questions of one kind (``plan``, ``guidance`` or ``review``)."""
        return [context for name, context in self.questions if name == kind]
