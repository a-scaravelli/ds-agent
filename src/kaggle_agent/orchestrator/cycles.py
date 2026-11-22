"""Cycles in one project and the carry-over store (ticket 30).

A project starts with ONE cycle, whose identity stays in ``project.json`` (``read_project``). A fork adds
a cycle to the SAME project: ``cycles.json`` (written only here, atomically replaced) lists the extra
cycles. An absent file means the legacy single cycle, so a project from before this ticket opens unchanged.

Each entry: ``cycle_id`` (``<slug>-2``, ``-3``...), ``request_id`` (a repeated launch of the same fork never
makes a second cycle), ``origin`` (``cycle_id`` and ``attempt_id`` or None), the frozen ``protocol`` and
``protocol_config`` of the new cycle, ``protocol_label`` (``same`` or ``changed``) with ``changes`` (for
example ``{"threshold_k": [old, new]}``), ``reason``, ``who`` and ``created_at``.

What is stored for the roles: ``carry/<cycle>.json``, an immutable snapshot written BEFORE the cycle exists:
the digest of the origin attempt (hypothesis, rationale, what changed, how to measure, validation score), the
best attempts, the failures, the guidance and reviewer texts, the DS notes and whether the protocol hash is
the same. It copies descriptions and numbers, NEVER code. ``steering_roles`` passes it to the coordinator,
feature engineering and modeling as the typed ``carry`` input; the registry fingerprint still rejects identical
code under the same protocol as a duplicate.
"""

import dataclasses
import json
import re
from pathlib import Path

from kaggle_agent.proposals import CarryOver

from .config import ProjectPaths
from .errors import ProjectError
from .project import ProjectState, write_atomic

CYCLES_FILE = "cycles.json"
CARRY_DIR = "carry"
CYCLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SAME, CHANGED = "same", "changed"


def cycles_path(folder: Path) -> Path:
    return Path(folder) / CYCLES_FILE


def read_cycles(folder: Path) -> list[dict]:
    """The fork entries in order; a missing file is the legacy single cycle (an empty list)."""
    path = cycles_path(folder)
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))["cycles"]
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise TypeError("cycles must be a list of objects")
        return items
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ProjectError(f"{path} is not a valid cycles file: {error}") from None


def write_cycles(folder: Path, entries: list[dict]) -> None:
    write_atomic(cycles_path(folder), json.dumps({"version": 1, "cycles": entries}, indent=2, sort_keys=True))


def active_state(paths: ProjectPaths, base: ProjectState) -> ProjectState:
    """The state of the active cycle: the base state, or the last fork entry laid over it."""
    entries = read_cycles(paths.folder)
    if not entries:
        return base
    last = entries[-1]
    return dataclasses.replace(base, cycle_id=last["cycle_id"], protocol=last["protocol"],
                               protocol_config=last["protocol_config"])


def new_cycle_id(slug: str, entries: list[dict]) -> str:
    return f"{slug}-{len(entries) + 2}"


def entry_for(entries: list[dict], cycle_id: str) -> dict | None:
    return next((e for e in entries if e.get("cycle_id") == cycle_id), None)


def entry_by_request(entries: list[dict], request_id: str) -> dict | None:
    return next((e for e in entries if e.get("request_id") == request_id), None)


def add_entry(folder: Path, entry: dict) -> dict:
    """Append an entry unless its ``request_id`` is known; returns the stored one (idempotent)."""
    entries = read_cycles(folder)
    known = entry_by_request(entries, entry["request_id"])
    if known is not None:
        return known
    write_cycles(folder, entries + [entry])
    return entry


# -- the carry-over -------------------------------------------------------------------------------


def carry_path(folder: Path, cycle_id: str) -> Path:
    if not CYCLE_ID.fullmatch(cycle_id):
        raise ProjectError(f"bad cycle id {cycle_id!r}")
    return Path(folder) / CARRY_DIR / f"{cycle_id}.json"


def read_carry(folder: Path, cycle_id: str) -> CarryOver | None:
    path = carry_path(folder, cycle_id)
    if not path.exists():
        return None
    try:
        return CarryOver.from_dict(json.loads(path.read_text(encoding="utf-8"))["carry"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ProjectError(f"{path} is not a valid carry-over file: {error}") from None


def write_carry(folder: Path, cycle_id: str, carry: CarryOver) -> None:
    """Write the snapshot once; an existing file is never rewritten (it is immutable)."""
    path = carry_path(folder, cycle_id)
    if not path.exists():
        write_atomic(path, json.dumps({"version": 1, "cycle_id": cycle_id, "carry": carry.to_dict()},
                                      indent=2, sort_keys=True))


def _validation(detail) -> float | None:
    if detail.outcome != "succeeded" or not detail.executions:
        return None
    return next((m.mean for m in detail.executions[-1].metrics if m.partition == "validation"), None)


def fold_notes(records: list[dict], run_id: str, attempt_id: str | None) -> list[str]:
    """The live notes (add, edit, delete) of the run and the attempt, from ``notes.jsonl`` records."""
    live: dict[str, dict] = {}
    for record in records:
        if record.get("op") == "tags":
            continue
        key = record.get("note_id")
        if record.get("op") == "add":
            live[key] = record
        elif record.get("op") == "edit" and key in live:
            live[key] = {**live[key], "text": record.get("text")}
        elif record.get("op") == "delete":
            live.pop(key, None)
    wanted = {("run", run_id)} | ({("attempt", attempt_id)} if attempt_id else set()) | {("competition", "*")}
    return [n["text"] for n in live.values() if (n.get("target_type"), n.get("target_id")) in wanted
            and isinstance(n.get("text"), str) and n["text"].strip()]


def build_carry(registry, origin_cycle: str, origin_attempt: str | None, direction: str,
                same_protocol: bool, protocol_note: str = "", notes=()) -> CarryOver:
    """The carry-over of a fork from the origin registry (read only); descriptions and numbers, no code."""
    sign = 1 if direction == "maximize" else -1
    details = []
    for summary in registry.list_attempts(origin_cycle):
        detail = registry.get_attempt(summary.attempt_id)
        details.append(detail)
    by_id = {d.attempt.id: d for d in details}
    attempt = ()
    if origin_attempt is not None:
        detail = by_id[origin_attempt]  # KeyError is checked by the caller (an unknown attempt)
        a = detail.attempt
        attempt = ({"id": a.id, "hypothesis": a.hypothesis, "rationale": a.rationale,
                    "what_changed": a.what_changed, "measure": a.measure, "outcome": detail.outcome,
                    "validation_score": _validation(detail)},)
    scored = [(d, _validation(d)) for d in details if d.closed]
    scored = [(d, s) for d, s in scored if s is not None]
    scored.sort(key=lambda item: -sign * item[1])
    top = tuple({"id": d.attempt.id, "hypothesis": d.attempt.hypothesis, "validation_score": s}
                for d, s in scored[:3])
    failed = tuple({"id": d.attempt.id, "hypothesis": d.attempt.hypothesis,
                    "failure_kind": d.failure_kind or d.outcome} for d in details
                   if d.closed and d.outcome == "failed")[-5:]
    return CarryOver(
        origin_cycle=origin_cycle, origin_attempt=origin_attempt, same_protocol=same_protocol,
        protocol_note=protocol_note, attempt=attempt, top_attempts=top, failures=failed,
        guidance=tuple(g.text for g in registry.list_guidance(origin_cycle)),
        reviewer=tuple(r.text for r in registry.list_reviewer_reports(origin_cycle)),
        notes=tuple(notes),
    )
