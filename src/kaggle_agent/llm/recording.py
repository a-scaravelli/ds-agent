"""Record the consumption of a call in the registry (design sections 7 and 11).

The only module of this subpackage that imports ``registry``. The client
itself never writes to the registry; the caller does, for instance with
``on_call=functools.partial(record_call, registry, cycle_id)``.
"""

from kaggle_agent.registry import LlmCallEntry, Registry

from .models import CallRecord


def record_call(registry: Registry, cycle_id: str, call: CallRecord) -> LlmCallEntry:
    """Store one call under a cycle: outcome, duration, tokens and model, nothing else."""
    entry = LlmCallEntry(
        cycle_id=cycle_id,
        purpose=call.purpose,
        outcome=call.outcome,
        started_at=call.started_at,
        duration_seconds=call.duration_seconds,
        input_tokens=call.usage.input_tokens,
        output_tokens=call.usage.output_tokens,
        attempt=call.attempt,
        model=call.model,
    )
    registry.add_llm_call(entry)
    return entry
