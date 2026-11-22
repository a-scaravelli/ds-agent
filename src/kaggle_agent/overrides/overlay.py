"""Pure functions that apply the DS overrides on top of what the registry stores (it is never rewritten)."""

import dataclasses


def valid_bar_overrides(records: list[dict], outcomes: dict[str, str], rounds_started: int,
                        last_complete: int) -> tuple[list[dict], list[tuple[dict, str]], list[dict]]:
    """Split the bar-override records into (valid, stale with reason, still pending).

    ``rounds_started`` is the number of iteration rounds opened, ``last_complete`` the number of the last
    complete one. A record already marked applied stays valid; a stale marker stays stale. A round-scope
    override is valid only for the last complete round while it is also the last one opened; a cycle-scope
    one from round k on is stale when round k is behind the last complete round.
    """
    valid, stale, pending = [], [], []
    for record in records:
        known = outcomes.get(record["id"])
        if known == "applied":
            valid.append(record)
            continue
        if known == "stale":
            continue
        number = record.get("round")
        if not isinstance(number, int):
            stale.append((record, "the record has no round"))
        elif record.get("scope") == "cycle":
            if number < last_complete:
                stale.append((record, f"round {number} is behind round {last_complete}, the last complete one"))
            else:
                valid.append(record)
        elif number > last_complete:
            pending.append(record)
        elif number == last_complete == rounds_started:
            valid.append(record)
        else:
            stale.append((record, f"addressed to round {number}, the last complete round is "
                                  f"{last_complete}"))
    return valid, stale, pending


def apply_bar_overrides(comparisons, overrides: list[dict], round_of: dict[str, int]) -> list:
    """The comparisons with ``is_real`` set for the overridden candidates (difference and threshold kept)."""
    if not overrides:
        return list(comparisons)
    out = []
    for comparison in comparisons:
        accepted = False
        number = round_of.get(comparison.candidate_attempt_id)
        for record in overrides:
            if record.get("scope") == "cycle":
                accepted |= number is not None and number >= record["round"] and comparison.difference > 0
            else:
                accepted |= record.get("attempt") == comparison.candidate_attempt_id
        out.append(dataclasses.replace(comparison, is_real=True) if accepted and not comparison.is_real
                   else comparison)
    return out


def resolve_choice(program_choice: str | None, records: list[dict], usable: set[str],
                   outcomes: dict[str, str] | None = None) -> tuple[str | None, dict | None]:
    """The chosen attempt: the latest valid ``choose_model`` of the DS wins, the program's pick is kept aside.

    A record is valid when its attempt is among the ``usable`` ones (closed, succeeded, scored) and it was
    not marked stale. Returns (chosen attempt id, the DS record or None).
    """
    outcomes = outcomes or {}
    chosen = None
    for record in records:
        if record.get("kind") == "choose_model" and record.get("attempt") in usable \
                and outcomes.get(record.get("id")) != "stale":
            chosen = record
    return (chosen["attempt"], chosen) if chosen else (program_choice, None)
