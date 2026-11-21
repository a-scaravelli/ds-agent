"""Feature families: derived features under their source column, then known-column prefixes, then stems."""

import re

_SPLIT = re.compile(r"[_=.:|]|__")


def _stem(name: str) -> str:
    return _SPLIT.split(name, 1)[0]


def feature_families(features: list[str], derived: list[dict] | None = None,
                     known_columns: list[str] | None = None) -> dict[str, str]:
    """``{feature: family}``; deterministic. ``derived`` is ``[{name, sources}]`` of describe.json."""
    sources = {d["name"]: d.get("sources") or [] for d in (derived or []) if isinstance(d, dict) and "name" in d}
    names = sorted(set(features) | set(sources))
    known = sorted(set(known_columns or []) | {s for srcs in sources.values() for s in srcs}, key=lambda c: (-len(c), c))
    stems: dict[str, int] = {}
    for name in names:
        stems[_stem(name)] = stems.get(_stem(name), 0) + 1
    out: dict[str, str] = {}
    for name in names:
        if sources.get(name):
            out[name] = sources[name][0]
            continue
        if name in known:
            out[name] = name
            continue
        match = next((c for c in known if len(name) > len(c) and name.startswith(c)
                      and name[len(c)] in "_=.:|"), None)
        if match:
            out[name] = match
        elif stems[_stem(name)] > 1 and _stem(name):
            out[name] = _stem(name)
        else:
            out[name] = name
    return out
