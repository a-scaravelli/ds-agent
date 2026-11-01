"""Lead interface of the autonomous cycle (design sections 2, 13 and 14, level 1).

The cycle asks the lead for three decisions through the injected ``LeadInput``: approve the
plan, give guidance after a round and accept the final report. Each answer is a small frozen
value; ``None`` means "no answer yet" and suspends the cycle, so a wait is never an implicit
approval. ``SimulatedLead`` is the scripted lead used in tests. Real command or file input is
a later step. This package is pure: standard library and ``proposals`` only.
"""

from .fake import SimulatedLead
from .lead import (
    CONTINUE,
    GUIDANCE,
    PROTOCOL_CHANGE,
    Accept,
    Approve,
    LeadContext,
    LeadInput,
    LeadResponse,
    Revise,
)

__all__ = [
    "Accept",
    "Approve",
    "CONTINUE",
    "GUIDANCE",
    "LeadContext",
    "LeadInput",
    "LeadResponse",
    "PROTOCOL_CHANGE",
    "Revise",
    "SimulatedLead",
]
