"""Case-insensitive search over notes, tags and suggestions of the learnings notebook."""


def search_items(items: list[dict], query: str | None) -> list[dict]:
    """Items whose text, tags or title contain every word of ``query``; an empty query lists all."""
    words = [w for w in (query or "").lower().split() if w]
    if not words:
        return list(items)
    out = []
    for item in items:
        haystack = " ".join([str(item.get("text") or ""), str(item.get("title") or ""),
                             " ".join(str(t) for t in item.get("tags") or ())]).lower()
        if all(word in haystack for word in words):
            out.append(item)
    return out
