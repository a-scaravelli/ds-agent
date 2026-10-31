"""Data layer of the read-only web interface (design sections 7 and 15, step 10).

Every function takes a ``Registry`` and returns plain dicts or the registry's
own read models. Only read methods of the registry are called here: nothing in
this module adds, sets, closes, links or opens anything.

Unknown ids raise ``NotFoundError`` (the app answers 404) and an invalid role
raises ``ValidationError`` (the app answers 400).

Metric direction: the registry interface does not expose the frozen protocol
(a ``Cycle`` holds only ``protocol_hash``, and ``Protocol.direction`` lives in
``evaluation``, which this package must not import). The direction therefore
comes from the ``best`` argument of ``metrics_page``; this is the single place
to read it from the registry if a later change makes it readable.
"""

from urllib.parse import urlparse

from kaggle_agent.proposals import validate_role
from kaggle_agent.registry import NotFoundError, Registry

PIPELINE_PREFIX = "pipeline:"
BEST_CHOICES = ("max", "min")
DEFAULT_BEST = "max"


def pipeline_reference(artifacts) -> dict | None:
    """Find the pipeline reference among artifacts: the first ``pipeline:<ref>`` entry.

    Convention (the pipeline step does not exist yet, this is the only place
    that knows the format): an artifact string starting with ``pipeline:``.
    Returns ``{"ref": str, "url": str | None}``; ``url`` is set only for
    http and https references, so the web never links to local files.
    """
    for artifact in artifacts:
        if isinstance(artifact, str) and artifact.startswith(PIPELINE_PREFIX):
            ref = artifact[len(PIPELINE_PREFIX):].strip()
            parsed = urlparse(ref)
            is_web = parsed.scheme in ("http", "https") and bool(parsed.netloc)
            return {"ref": ref, "url": ref if is_web else None}
    return None


def list_cycles(registry: Registry) -> list:
    """All cycles in insertion order."""
    return registry.list_cycles()


def _candidate_row(registry: Registry, summary) -> dict:
    detail = registry.get_attempt(summary.attempt_id)
    last = detail.executions[-1] if detail.executions else None
    return {
        "summary": summary,
        "what_changed": detail.attempt.what_changed,
        "metrics": list(last.metrics) if last else [],
        "runs": len(detail.executions),
        "closed": detail.closed,
    }


def cycle_page(registry: Registry, cycle_id: str) -> dict:
    """A cycle with its rounds, their candidates and the timeline of all attempts."""
    cycle = registry.get_cycle(cycle_id)
    rounds = registry.list_rounds_with_candidates(cycle_id)
    attempts = registry.list_attempts(cycle_id)
    in_round = {c.attempt_id for r in rounds for c in r.candidates}
    return {
        "cycle": cycle,
        "rounds": [
            {
                "detail": r,
                "candidates": [_candidate_row(registry, c) for c in r.candidates],
            }
            for r in rounds
        ],
        "unrounded": [
            _candidate_row(registry, a) for a in attempts if a.attempt_id not in in_round
        ],
        "timeline": attempts,
        "bundles": registry.list_bundles(cycle_id),
    }


def round_page(registry: Registry, cycle_id: str, round_id: str) -> dict:
    """One round: tasks, candidates, reviewer reports, guidance and traces of that round."""
    cycle = registry.get_cycle(cycle_id)
    detail = next(
        (r for r in registry.list_rounds_with_candidates(cycle_id) if r.round.id == round_id),
        None,
    )
    if detail is None:
        raise NotFoundError(f"round {round_id!r} does not exist in cycle {cycle_id!r}")
    return {
        "cycle": cycle,
        "detail": detail,
        "candidates": [_candidate_row(registry, c) for c in detail.candidates],
        "reports": registry.list_reviewer_reports(cycle_id, round_id),
        "guidance": [g for g in registry.list_guidance(cycle_id) if g.round_id == round_id],
        "traces": registry.list_role_traces(cycle_id, round_id=round_id),
    }


def partitions_of(registry: Registry, cycle_id: str) -> list[str]:
    """Partitions that appear in the execution metrics of the cycle's attempts, sorted."""
    found: set[str] = set()
    for summary in registry.list_attempts(cycle_id):
        for execution in registry.get_attempt(summary.attempt_id).executions:
            found.update(m.partition for m in execution.metrics)
    return sorted(found)


def metrics_page(
    registry: Registry, cycle_id: str, partition: str | None = None, best: str | None = None
) -> dict:
    """Metric series of a cycle per round and per iteration for one partition.

    ``best`` is ``max`` or ``min`` (default ``max``): the registry does not
    store the direction of the metric, see the module docstring.
    """
    cycle = registry.get_cycle(cycle_id)
    best = best if best in BEST_CHOICES else DEFAULT_BEST
    partitions = partitions_of(registry, cycle_id)
    if partition is None and partitions:
        partition = "validation" if "validation" in partitions else partitions[0]
    if partition is None or partition not in partitions:
        return {
            "cycle": cycle, "partitions": partitions, "partition": partition, "best": best,
            "by_round": [], "by_iteration": [],
        }
    return {
        "cycle": cycle,
        "partitions": partitions,
        "partition": partition,
        "best": best,
        "by_round": registry.metric_series_by_round(cycle_id, partition),
        "by_iteration": registry.metric_series(cycle_id, partition),
    }


def attempt_page(registry: Registry, attempt_id: str) -> dict:
    """Candidate detail: proposal, runs, failures, pipeline link, traces and bundle."""
    detail = registry.get_attempt(attempt_id)
    cycle_id = detail.cycle_id
    artifacts = [a for e in detail.executions for a in e.artifacts]
    bundle_id = detail.attempt.bundle_id
    bundle = None
    if bundle_id is not None:
        bundle = next((b for b in registry.list_bundles(cycle_id) if b.id == bundle_id), None)
    summary = next(s for s in registry.list_attempts(cycle_id) if s.attempt_id == attempt_id)
    round_id = next(
        (r.round.id for r in registry.list_rounds_with_candidates(cycle_id)
         if any(c.attempt_id == attempt_id for c in r.candidates)),
        None,
    )
    return {
        "cycle": registry.get_cycle(cycle_id),
        "detail": detail,
        "summary": summary,
        "round_id": round_id,
        "bundle_id": bundle_id,
        "bundle": bundle,
        "pipeline": pipeline_reference(artifacts),
        "traces": [t for t in registry.list_role_traces(cycle_id) if t.candidate_id == attempt_id],
    }


def reports_page(registry: Registry, cycle_id: str) -> dict:
    """Reviewer reports of a cycle grouped by round, in round order."""
    cycle = registry.get_cycle(cycle_id)
    reports = registry.list_reviewer_reports(cycle_id)
    groups = []
    for r in registry.list_rounds_with_candidates(cycle_id):
        own = [x for x in reports if x.round_id == r.round.id]
        if own:
            groups.append({"round": r, "reports": own})
    return {"cycle": cycle, "groups": groups}


def traces_page(
    registry: Registry, cycle_id: str, role: str | None = None, round_id: str | None = None
) -> dict:
    """Role traces of a cycle, optionally of one role and one round."""
    cycle = registry.get_cycle(cycle_id)
    if role:
        validate_role(role)
    round_ids = [r.round.id for r in registry.list_rounds_with_candidates(cycle_id)]
    if round_id and round_id not in round_ids:
        raise NotFoundError(f"round {round_id!r} does not exist in cycle {cycle_id!r}")
    return {
        "cycle": cycle,
        "role": role or None,
        "round_id": round_id or None,
        "round_ids": round_ids,
        "traces": registry.list_role_traces(cycle_id, role or None, round_id or None),
    }


def guidance_page(registry: Registry, cycle_id: str) -> dict:
    """Lead guidance of a cycle with the suggestions created from it."""
    cycle = registry.get_cycle(cycle_id)
    items = []
    for g in registry.list_guidance(cycle_id):
        items.append({"guidance": g, "suggestion": registry.get_suggestion(g.id)})
    return {"cycle": cycle, "items": items}


def suggestions_page(registry: Registry) -> list[dict]:
    """All suggestions with the cycles spawned from each."""
    return [
        {"suggestion": s, "cycles": registry.cycles_of_suggestion(s.id)}
        for s in registry.list_suggestions()
    ]
