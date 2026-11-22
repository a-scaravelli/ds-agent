"""Helpers of the ticket 30 tests: a finished fake cycle, decision records as the web writes them, the cockpit files."""

import json
from pathlib import Path

from orchestrator_helpers import (
    GUIDE_CONTINUE,
    PLAN_APPROVE,
    REVIEW_ACCEPT,
    World,
    drive_to_review,
    drive_to_stopped,
)

from kaggle_agent.orchestrator.cockpit import CockpitFiles, append_jsonl, read_jsonl
from kaggle_agent.registry import SqliteRegistry

FIRST = "toy-1"
SECOND = "toy-2"
REQUEST = "request-0001"


def files(world: World) -> CockpitFiles:
    return CockpitFiles(world.project)


def decide(world: World, kind: str, cycle: str | None = FIRST, **fields) -> dict:
    """Append a decision record exactly like ``web.writes._save_decision`` does."""
    records, _ = read_jsonl(files(world).decisions)
    record = {"id": f"d{len(records) + 1:03d}", "request_id": f"r{len(records) + 1:07d}", "kind": kind,
              "who": "tester", "at": "2026-01-01T12:00:00+00:00", **fields}
    if cycle is not None:
        record["cycle"] = cycle
    append_jsonl(files(world).decisions, record)
    return record


def applied(world: World) -> list[dict]:
    return read_jsonl(files(world).applied)[0]


def reads(world: World, **fields) -> dict:
    record = {"id": f"h{len(read_jsonl(files(world).holdout_reads)[0]) + 1}", "cycle": FIRST, "kind": "first",
              "reason": "", "who": "tester", "at": "2026-01-01T12:00:00+00:00", **fields}
    append_jsonl(files(world).holdout_reads, record)
    return record


def registry(world: World):
    return SqliteRegistry(str(world.project / "registry.db"), read_only=True)


def attempts(world: World, cycle: str = FIRST) -> list[str]:
    reg = registry(world)
    try:
        return [a.attempt_id for a in reg.list_attempts(cycle)]
    finally:
        reg.close()


def stopped_world(tmp_path, **kwargs) -> World:
    world = World(tmp_path, **kwargs)
    drive_to_stopped(world)
    return world


def fork(world: World, attempt: str | None = None, request: str = REQUEST, *extra):
    args = ["fork", "--from-run", FIRST, "--request-id", request, *extra]
    if attempt:
        args += ["--from-attempt", attempt]
    return world.run(*args)


def cockpit_json(world: World, name: str):
    return json.loads((world.project / name).read_text(encoding="utf-8"))
