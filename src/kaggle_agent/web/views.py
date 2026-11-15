"""Small view helpers reused by ``data.py`` (design sections 7 and 15).

Every function takes a ``Registry`` and only calls read methods of it: nothing here adds, sets,
closes, links or opens anything. ``data.py`` builds the dicts of the JSON API and the pages from
these helpers and from the registry directly.

Metric direction: the registry interface does not expose the frozen protocol (a ``Cycle`` holds
only ``protocol_hash``, and ``Protocol.direction`` lives in ``evaluation``, which this package
must not import). ``data.py`` reads it from ``project.json`` and falls back to the comparisons and
to the ``best`` query parameter (``BEST_CHOICES``, default ``DEFAULT_BEST``).
"""

import re

from kaggle_agent.registry import Registry

PIPELINE_PREFIX = "pipeline:"
_WEB_URL = re.compile(r"https?://[^\s/?#]+\S*", re.IGNORECASE)
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
            return {"ref": ref, "url": ref if _WEB_URL.fullmatch(ref) else None}
    return None


def list_cycles(registry: Registry) -> list:
    """All cycles in insertion order."""
    return registry.list_cycles()


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
