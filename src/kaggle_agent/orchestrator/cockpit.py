"""Files of the cockpit in ``<project>/cockpit/`` as the orchestrator process sees them.

Each file has ONE writer process (see ``web/store.py`` for the full contract). The orchestrator
writes ``events.jsonl``, ``heartbeat.json``, ``exit.json`` and ``decisions-applied.jsonl`` and holds
``run.lock``; it only READS ``pause-request.json``, ``decisions.jsonl``, ``limit-changes.jsonl`` and ``holdout-reads.jsonl``.

JSONL rules: one JSON object per line, at most 32 KB; an append is one ``os.write`` on an
``O_APPEND`` descriptor under an exclusive ``flock``; readers accept only lines that end in a newline
and skip (and count) a line that does not parse, so a torn tail never breaks them.
"""

import dataclasses
import fcntl
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .project import write_atomic

COCKPIT_DIR = "cockpit"
MAX_LINE = 32 * 1024
HEARTBEAT_INTERVAL = 10.0  # seconds; a test script may lower it before calling ``main``

_APPEND_LOCK = threading.Lock()


class PauseRequested(BaseException):
    """The DS asked to pause: raised between calls, never recorded as a failed attempt."""


class Interrupted(BaseException):
    """SIGTERM arrived (stop): lets the ``finally`` blocks run (sandbox group kill, registry close)."""


def utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


class CockpitFiles:
    """The paths of the cockpit folder of one project."""

    def __init__(self, project_folder) -> None:
        self.folder = Path(project_folder) / COCKPIT_DIR
        self.events = self.folder / "events.jsonl"
        self.heartbeat = self.folder / "heartbeat.json"
        self.exit = self.folder / "exit.json"
        self.applied = self.folder / "decisions-applied.jsonl"
        self.run_lock = self.folder / "run.lock"
        self.pause_request = self.folder / "pause-request.json"
        self.decisions = self.folder / "decisions.jsonl"
        self.limit_changes = self.folder / "limit-changes.jsonl"
        self.holdout_reads = self.folder / "holdout-reads.jsonl"

    def ensure(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)

    def pause_pending(self) -> bool:
        return self.pause_request.exists()


def read_jsonl(path) -> tuple[list[dict], int]:
    """The complete, valid records of a JSONL file and the number of lines skipped."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return [], 0
    parts = raw.split(b"\n")
    tail = parts.pop()  # what follows the last newline: empty, or a torn line
    records, skipped = [], 1 if tail.strip() else 0
    for line in parts:
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except ValueError:
            skipped += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            skipped += 1
    return records, skipped


def append_jsonl(path, record: dict, sync: bool = False) -> None:
    """Append one record as one line (single write, exclusive flock, torn tail repaired)."""
    line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(line) > MAX_LINE:
        raise ValueError(f"a record is at most {MAX_LINE} bytes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                line = b"\n" + line  # a crash left a half line: it becomes a skipped line of its own
            os.write(fd, line)
            if sync:
                os.fsync(fd)
        finally:
            os.close(fd)  # closing releases the flock


class RunLock:
    """``flock`` on ``run.lock`` for the whole run; the kernel frees it when the process dies."""

    def __init__(self, files: CockpitFiles) -> None:
        self.files = files
        self._fd: int | None = None

    def acquire(self) -> bool:
        self.files.ensure()
        fd = os.open(self.files.run_lock, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.set_inheritable(fd, False)
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class Heartbeat:
    """A daemon thread that rewrites ``heartbeat.json`` every few seconds while the run lives."""

    def __init__(self, files: CockpitFiles, interval: float | None = None) -> None:
        self.files = files
        self.interval = HEARTBEAT_INTERVAL if interval is None else interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")

    def _beat(self) -> None:
        try:
            write_atomic(self.files.heartbeat, json.dumps({"pid": os.getpid(), "at": utc_text(),
                                                           "monotonic": time.monotonic()}))
        except OSError:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._beat()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._beat()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# -- effective limits ----------------------------------------------------------------------------

_DEDICATED = {"rounds": "iterations", "stagnation": "stagnation_k"}
_STORED_KEYS = ("candidates", "pause_every", "max_consecutive_failures", "max_rejections")
LIMIT_KEYS = ("rounds", "candidates", "stagnation", *_STORED_KEYS[1:])


def _valid(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def limit_overrides(records: list[dict]) -> dict:
    """Base overlaid by the records in file order: the last value wins per key."""
    merged: dict = {}
    for record in records:
        values = record.get("values")
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key in LIMIT_KEYS and _valid(value):
                merged[key] = value
    return merged


def overlay_cycle(cycle, records: list[dict]):
    """The cycle with the DS's limit changes applied; ONLY rounds, stagnation and the stored keys.

    Applied only when the stored parameters exist (so the write-once ``set_cycle_parameters`` is
    unaffected); every other field is returned exactly as stored.
    """
    if cycle.parameters is None:
        return cycle
    merged = limit_overrides(records)
    if not merged:
        return cycle
    changes = {_DEDICATED[k]: v for k, v in merged.items() if k in _DEDICATED}
    parameters = dict(cycle.parameters)
    parameters.update({k: v for k, v in merged.items() if k in _STORED_KEYS})
    return dataclasses.replace(cycle, parameters=parameters, **changes)
