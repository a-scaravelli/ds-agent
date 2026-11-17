"""The files of the cockpit and the ONLY place of the web package that opens, writes or unlinks them.

SINGLE WRITER PRINCIPLE. The orchestrator process is the only writer of ``registry.db``; the web keeps
read-only registry connections. Everything the web writes lives in ``<project>/cockpit/``, next to
``project.json``, in files that each have exactly ONE writer process:

==========================  =====================================================================
``events.jsonl``            orchestrator: the event log (id, at, type, cycle, data)
``heartbeat.json``          orchestrator (``--supervised``): pid and time, every 10 s
``exit.json``               orchestrator: code, meaning, at, pid, written when the run ends
``decisions-applied.jsonl`` orchestrator: which web decision was applied or found stale
``run.lock``                orchestrator: ``flock`` held for the whole run (the kernel frees it)
``run.json``                web controller: pid, pgid, ps start-time string, command, config, ...
``console.log``             the child's stdout and stderr (opened by the controller, inherited)
``controller.lock``         web: a short ``flock`` around check-and-spawn
``audit.jsonl``             web: every DS action (who, when, what, why), written BEFORE it acts
``decisions.jsonl``         web: decisions addressed to a question (plan, round)
``limit-changes.jsonl``     web: append-only limit extensions, overlaid by the orchestrator
``notes.jsonl``            web: notes and tags on a run or an attempt (add, edit, delete, tags)
``settings.json``           web: user settings (auto resume)
``pause-request.json``      web: asks the orchestrator to pause at a safe checkpoint
``jobs/<id>.json``          web: state of a background job (the child writes ``jobs/<id>.out.json``)
``config-history/``         web: the previous ``config.toml`` copies
==========================  =====================================================================

The folder is created lazily by a WRITER; a GET never creates anything, and a project without it opens
normally. ``lead.toml``, ``lead-ledger.json``, ``NOTICES.txt`` and ``last-run.json`` live in the project
folder as before.

JSONL rules: one JSON object per line, at most 32 KB. An append is ONE ``os.write`` on an ``O_APPEND``
descriptor under an exclusive ``flock`` and an in-process lock, followed by ``fsync`` for decisions and
audit. Readers accept only lines that end in a newline, skip a line that does not parse (and count it) and
never fail on a torn tail. A JSON file is written with temp file + ``os.replace`` (the same semantics as
``orchestrator.project.write_atomic``; the web cannot import the orchestrator, so the copy lives here).
Local filesystems only (macOS, Linux): ``flock`` on a network share is not supported.

Every path of the web package is built by ``cockpit_path`` from a fixed set of names.
"""

import fcntl
import getpass
import json
import os
import re
import tempfile
import threading
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

COCKPIT_DIR = "cockpit"
MAX_LINE = 32 * 1024
NAMES = frozenset({
    "events.jsonl", "heartbeat.json", "exit.json", "decisions-applied.jsonl", "run.lock", "run.json",
    "console.log", "controller.lock", "audit.jsonl", "decisions.jsonl", "limit-changes.jsonl",
    "notes.jsonl", "settings.json", "pause-request.json",
})
PROJECT_NAMES = frozenset({"lead.toml", "lead-ledger.json", "NOTICES.txt", "last-run.json", "project.json",
                           "config.toml", "config.candidate.toml"})
JOB_ID = re.compile(r"[a-f0-9]{12,32}")
_APPEND_LOCK = threading.Lock()


def utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def os_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def cockpit_path(project_dir, name: str) -> Path:
    """A path in ``<project>/cockpit/`` from the fixed set of names."""
    if name not in NAMES:
        raise ValueError(f"not a cockpit file: {name!r}")
    return Path(project_dir) / COCKPIT_DIR / name


def project_path(project_dir, name: str) -> Path:
    if name not in PROJECT_NAMES:
        raise ValueError(f"not a project file: {name!r}")
    return Path(project_dir) / name


def job_paths(project_dir, job_id: str) -> tuple[Path, Path]:
    if not JOB_ID.fullmatch(job_id):
        raise ValueError("bad job id")
    folder = Path(project_dir) / COCKPIT_DIR / "jobs"
    return folder / f"{job_id}.json", folder / f"{job_id}.out.json"


# -- JSONL ---------------------------------------------------------------------------------------


def read_jsonl(path) -> tuple[list[dict], int]:
    """The complete valid records and the number of skipped lines; a missing file is empty."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return [], 0
    parts = raw.split(b"\n")
    tail = parts.pop()
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
                line = b"\n" + line
            os.write(fd, line)
            if sync:
                os.fsync(fd)
        finally:
            os.close(fd)


# -- JSON files ----------------------------------------------------------------------------------


def write_atomic(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def write_json(path, value) -> None:
    write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def remove(path) -> None:
    Path(path).unlink(missing_ok=True)


def exists(path) -> bool:
    return Path(path).exists()


def mtime(path) -> float | None:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


# -- locks ---------------------------------------------------------------------------------------


def lock_is_held(path) -> bool:
    """True if another process holds ``flock`` on the file (never creates the file)."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


class FileLock:
    """A blocking exclusive ``flock`` on a file of the cockpit folder (created lazily)."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def console_handle(path):
    """The console log opened for append, to be inherited by the child process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "ab")


def tail_bytes(path, limit: int) -> bytes:
    """The last ``limit`` bytes of a file (empty when missing)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read()
    except OSError:
        return b""


# -- project files read by the pages -------------------------------------------------------------


def read_toml(path) -> dict | None:
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return data


def lead_toml_summary(project_dir) -> dict:
    """Only the table names and addresses of ``lead.toml`` (never free text), or a warning."""
    path = project_path(project_dir, "lead.toml")
    if not path.exists():
        return {"exists": False, "invalid": None, "tables": {}}
    data = read_toml(path)
    if data is None:
        return {"exists": True, "invalid": "lead.toml is not valid TOML", "tables": {}}
    tables = {}
    for name in ("plan", "guidance", "review"):
        table = data.get(name)
        if isinstance(table, dict) and table:
            tables[name] = {k: table.get(k) for k in ("decision", "action", "round", "plan")
                            if isinstance(table.get(k), (str, int)) and not isinstance(table.get(k), bool)}
    return {"exists": True, "invalid": None, "tables": tables}


def read_ledger(project_dir) -> list[dict]:
    data = read_json(project_path(project_dir, "lead-ledger.json"), {})
    items = data.get("revisions") if isinstance(data, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def read_notices(project_dir) -> list[str]:
    try:
        text = project_path(project_dir, "NOTICES.txt").read_text(encoding="utf-8")
    except OSError:
        return []
    return text.splitlines()[-50:]


# -- a small TOML writer (stdlib has only a reader) ---------------------------------------------------


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    raise ValueError(f"cannot write {type(value).__name__} to TOML")


def dumps_toml(data: dict) -> str:
    """Tables of scalars, lists and inline tables: enough for ``config.toml``."""
    lines = []
    for section, table in data.items():
        if not isinstance(table, dict):
            raise ValueError(f"{section}: must be a table")
        lines.append(f"[{section}]")
        for key, value in table.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", str(key)):
                raise ValueError(f"bad key {key!r}")
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def history_path(project_dir, stamp: str) -> Path:
    """Where the previous ``config.toml`` is kept (``cockpit/config-history/``)."""
    if not re.fullmatch(r"[0-9TZ:+.-]+", stamp):
        raise ValueError("bad stamp")
    return Path(project_dir) / COCKPIT_DIR / "config-history" / f"config-{stamp.replace(':', '')}.toml"


def copy_file(source, target) -> None:
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    write_atomic(target, Path(source).read_text(encoding="utf-8"))


def list_jobs(project_dir) -> list[Path]:
    folder = Path(project_dir) / COCKPIT_DIR / "jobs"
    try:
        return sorted(p for p in folder.iterdir() if p.name.endswith(".json") and ".out." not in p.name)
    except OSError:
        return []
