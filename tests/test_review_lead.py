"""The lead interface and the simulated lead: no answer is never an approval."""

import pytest

from kaggle_agent.proposals import ValidationError
from kaggle_agent.review import (
    Accept,
    Approve,
    LeadContext,
    LeadInput,
    LeadResponse,
    Revise,
    SimulatedLead,
)

CONTEXT = LeadContext("C1", "AWAITING_GUIDANCE")


def test_response_types_are_validated():
    assert LeadResponse("continue").text == ""
    assert LeadResponse("guidance", "try trees").kind == "guidance"
    for bad in (lambda: LeadResponse("guidance"), lambda: LeadResponse("approve"),
                lambda: Revise("  ")):
        with pytest.raises(ValidationError):
            bad()


def test_simulated_lead_plays_its_script_then_the_default_or_silence():
    lead = SimulatedLead(plans=[Revise("shorter"), None], guidance=[LeadResponse("guidance", "g")])
    assert isinstance(lead, LeadInput)
    assert lead.plan_decision(CONTEXT) == Revise("shorter")
    assert lead.plan_decision(CONTEXT) is None
    assert lead.plan_decision(CONTEXT) == Approve()
    assert lead.guidance(CONTEXT).text == "g"
    assert lead.guidance(CONTEXT).kind == "continue"
    assert lead.review_decision(CONTEXT) == Accept()
    lead.notice("hello")
    assert lead.notices == ["hello"] and len(lead.asked("guidance")) == 2

    silent = SimulatedLead(exhausted="silent")
    assert silent.plan_decision(CONTEXT) is None
    assert silent.guidance(CONTEXT) is None
    assert silent.review_decision(CONTEXT) is None
    with pytest.raises(ValueError):
        SimulatedLead(exhausted="x")
