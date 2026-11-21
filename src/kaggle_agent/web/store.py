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
``jobs/<id>.progress.jsonl``  orchestrator job: progress events of a heavy job (read by the web)
==========================  =====================================================================

ANALYTICS (ticket 29). ``<project>/analytics/`` holds files written ONLY by the orchestrator (see
``orchestrator.analytics_record``): content-addressed, immutable files plus ``index.jsonl``, whose line is the commit
point. The web READS them here and nowhere else: a path comes only from ``analytics_path`` (strict regexes for the
bundle id and the sha12, never a request string), a file is opened only if an index line names it, is read with a
byte cap, its sha256 is compared with the index (one retry, then "not readable") and a GET never creates the
folder. Because files are never rewritten, a reader cannot see a half-written file while the orchestrator writes.

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
import hashlib
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


# -- analytics files (read only; written by the orchestrator) ----------------------------------------

ANALYTICS_DIR = "analytics"
BUNDLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA12 = re.compile(r"[a-f0-9]{12}")
ANALYTICS_CAPS = {"oof": 25 * 1024 * 1024, "describe": 80 * 1024, "development": 100 * 1024 * 1024,
                  "importance": 5 * 1024 * 1024}
_SUFFIX = {"oof": ".csv.gz", "describe": ".json"}


def analytics_path(project_dir, kind: str, bundle_id: str | None = None, sha12: str | None = None) -> Path:
    """A path in ``<project>/analytics/`` built from a kind and validated ids (never from a request string)."""
    base = Path(project_dir) / ANALYTICS_DIR
    if kind == "index":
        return base / "index.jsonl"
    if kind == "development":
        if not (sha12 and SHA12.fullmatch(sha12)):
            raise ValueError("bad hash")
        return base / f"development-{sha12}.csv.gz"
    if kind not in ("oof", "describe", "importance") or not (bundle_id and BUNDLE_ID.fullmatch(bundle_id)):
        raise ValueError("bad analytics file")
    if kind == "importance":
        return base / "importance" / f"{bundle_id}.json"
    if not (sha12 and SHA12.fullmatch(sha12)):
        raise ValueError("bad hash")
    return base / kind / f"{bundle_id}-{sha12}{_SUFFIX[kind]}"


def analytics_index(project_dir) -> tuple[dict, int]:
    """The last valid index line per (kind, bundle_id); a torn or invalid line is skipped and counted."""
    records, skipped = read_jsonl(analytics_path(project_dir, "index"))
    latest: dict = {}
    for record in records:
        kind, bundle = record.get("kind"), record.get("bundle_id")
        if record.get("v") != 1 or kind not in ("oof", "describe", "development"):
            skipped += 1
            continue
        latest[(kind, bundle if kind != "development" else None)] = record
    return latest, skipped


def _read_capped(path: Path, cap: int) -> bytes:
    with open(path, "rb") as handle:
        blob = handle.read(cap + 1)
    if len(blob) > cap:
        raise ValueError("the file is over the size cap")
    return blob


def read_analytics_file(project_dir, line: dict) -> tuple[bytes | None, str | None]:
    """The bytes a valid index line names, checked against its sha256; (None, reason) when not readable."""
    kind = line.get("kind")
    digest = line.get("sha256")
    try:
        if kind not in ANALYTICS_CAPS or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("the index line is not valid")
        path = analytics_path(project_dir, kind, line.get("bundle_id"), digest[:12])
        relative = path.relative_to(Path(project_dir) / ANALYTICS_DIR).as_posix()
        if line.get("file") != relative:
            raise ValueError("the index line names a different file")
    except ValueError as exc:
        return None, str(exc)
    reason = "the file is missing"
    for attempt in range(2):  # one retry: a copy of the folder may be half way
        try:
            blob = _read_capped(path, ANALYTICS_CAPS[kind])
        except FileNotFoundError:
            reason = "the file named by the index is missing"
        except (OSError, ValueError) as exc:
            reason = f"the file cannot be read ({exc})"
        else:
            if hashlib.sha256(blob).hexdigest() == digest:
                return blob, None
            reason = "the file does not match the hash of the index"
        if attempt == 0:
            threading.Event().wait(0.05)
    return None, reason


def read_importance(project_dir, bundle_id: str) -> dict | None:
    """The result of the importance job for a bundle (a JSON object), or None."""
    try:
        path = analytics_path(project_dir, "importance", bundle_id)
        value = json.loads(_read_capped(path, ANALYTICS_CAPS["importance"]).decode("utf-8"))
    except (ValueError, OSError):
        return None
    return value if isinstance(value, dict) else None


def job_progress(project_dir, job_id: str) -> dict:
    """The last progress event of a heavy job and its percentage (nothing is created)."""
    if not JOB_ID.fullmatch(job_id):
        raise ValueError("bad job id")
    records, _ = read_jsonl(Path(project_dir) / COCKPIT_DIR / "jobs" / f"{job_id}.progress.jsonl")
    last = records[-1] if records else None
    total = next((r.get("total") for r in records if r.get("type") == "job_start"), None)
    done = next((r.get("done") for r in reversed(records) if r.get("type") == "job_progress"), 0)
    finished = bool(last) and last.get("type") == "job_end"
    percent = 100.0 if finished and last.get("status") == "done" else (
        round(100.0 * done / total, 1) if isinstance(total, int) and total > 0 and isinstance(done, int) else None)
    return {"last": last, "events": len(records), "percent": percent}
