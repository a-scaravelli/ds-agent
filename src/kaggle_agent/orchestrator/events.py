"""The event log: ``cockpit/events.jsonl``, append-only, written only by the orchestrator process.

Each event is ``{"id", "at", "type", "cycle", "data"}``: ``id`` grows by one (it continues after a
restart from the last complete line), ``at`` is UTC, ``data`` is a small payload. Payloads go through
a per-type allowlist of keys with scalar values only, strings cut to 200 characters and the whole
payload capped at 2 KB: prompts, response text, hypotheses and secrets never pass because no
allowed key carries them. Only ``wrappers``, ``build`` and ``service`` emit events.
"""

import json
import threading
from pathlib import Path

from .cockpit import append_jsonl, read_jsonl, utc_text

MAX_PAYLOAD = 2048
MAX_TEXT = 200
TAIL_BYTES = 256 * 1024

ALLOWED = {
    "run_start": ("command", "state", "resumed"),
    "cycle_state": ("state", "previous"),
    "stage_start": ("stage", "role", "detail"),
    "stage_end": ("stage", "role", "outcome", "seconds"),
    "round_start": ("round", "position"),
    "candidate_start": ("attempt", "round"),
    "candidate_end": ("attempt", "outcome"),
    "fold_start": ("bundle", "fold", "folds", "partition"),
    "fold_end": ("bundle", "fold", "folds", "partition", "seconds", "score"),
    "llm_call": ("role", "outcome", "input_tokens", "output_tokens", "seconds", "model"),
    "guidance_recorded": ("round",),
    "waiting": ("question", "state", "round", "plan"),
    "suspended": ("kind", "state"),
    "error": ("kind", "type"),
    "finished": ("reason", "state"),
    "paused": ("state",),
    "stopped": ("state",),
    "decision_applied": ("decision", "kind", "source"),
    "decision_stale": ("decision", "kind", "reason", "source"),
    "limits_applied": ("rounds", "candidates", "stagnation", "pause_every"),
}


def clean_payload(type_: str, payload: dict) -> dict:
    """Keep only the allowed keys with scalar values; shorten text; cap the size."""
    if type_ not in ALLOWED:
        raise ValueError(f"unknown event type {type_!r}")
    out = {}
    for key in ALLOWED[type_]:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            value = round(value, 6)
        elif isinstance(value, str):
            value = " ".join(value.split())[:MAX_TEXT]
        elif not isinstance(value, (bool, int)):
            continue
        out[key] = value
    while len(json.dumps(out)) > MAX_PAYLOAD and out:
        out.pop(next(reversed(out)))
    return out


class EventLog:
    """Appends events with consecutive ids; the next id is found from the last complete line."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._last = self._read_last_id()

    def _read_last_id(self) -> int:
        if not self.path.exists():
            return 0
        size = self.path.stat().st_size
        records, _ = read_jsonl(self.path) if size <= TAIL_BYTES else self._tail_records()
        ids = [r["id"] for r in records if isinstance(r.get("id"), int)]
        return max(ids, default=0)

    def _tail_records(self):
        with open(self.path, "rb") as handle:
            handle.seek(-TAIL_BYTES, 2)
            raw = handle.read().split(b"\n", 1)[1]  # drop the partial first line
        records = []
        for line in raw.split(b"\n")[:-1]:
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
        return records, 0

    def append(self, type_: str, cycle: str | None, payload: dict) -> dict:
        data = clean_payload(type_, payload)
        with self._lock:
            self._last += 1
            event = {"id": self._last, "at": utc_text(), "type": type_, "cycle": cycle, "data": data}
            append_jsonl(self.path, event)
        return event


class Emitter:
    """``emit(type, **payload)``; the cycle id is set once it is known."""

    def __init__(self, log: EventLog | None, cycle_id: str | None = None) -> None:
        self.log = log
        self.cycle_id = cycle_id

    def emit(self, type_: str, **payload) -> None:
        if self.log is None:
            return
        try:
            self.log.append(type_, self.cycle_id, payload)
        except OSError:
            pass  # an event is never worth stopping the run for


NULL = Emitter(None)
