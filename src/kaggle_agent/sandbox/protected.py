"""Protected files: SHA-256 hashes, diff check and read-only copies (design section 8).

Standard library only.
"""

import hashlib
import os
import posixpath
import re
import shutil
import stat
from collections.abc import Iterable
from pathlib import Path

from .errors import SandboxConfigError
from .models import ProtectedPath

MISSING = "<missing>"
UNREADABLE = "<unreadable>"
COPY_PREFIX = "copy:"


def _files(root: Path) -> list[Path]:
    found = (p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(root).parts)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def hash_path(path: Path) -> str:
    """SHA-256 of a file, or of a folder (sorted relative paths and bytes, no ``__pycache__``)."""
    digest = hashlib.sha256()
    if path.is_dir():
        for item in _files(path):
            digest.update(item.relative_to(path).as_posix().encode("utf-8") + b"\0")
            digest.update(item.read_bytes() + b"\0")
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_paths(protected: Iterable[ProtectedPath], missing_ok: bool = False) -> dict[str, str]:
    """SHA-256 per protected entry, keyed by name.

    A missing path raises ``SandboxConfigError``; with ``missing_ok`` (used after a run) a
    missing or unreadable path gets a marker value instead, so it counts as a change.
    """
    result = {}
    for entry in protected:
        if not entry.path.exists():
            if not missing_ok:
                raise SandboxConfigError(f"protected path does not exist: {entry.name}")
            result[entry.name] = MISSING
            continue
        try:
            result[entry.name] = hash_path(entry.path)
        except OSError:
            if not missing_ok:
                raise
            result[entry.name] = UNREADABLE
    return result


def _normalize(raw: str) -> str:
    text = re.sub(r"/+", "/", raw.strip().strip('"').replace("\\", "/"))
    return posixpath.normpath(text)


def _header_paths(diff_text: str) -> list[str]:
    """Raw paths declared in the headers of unified and git diffs."""
    paths: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("--- ", "+++ ")):
            raw = line[4:].split("\t")[0]
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            raw = line.split(" ", 2)[2]
        elif line.startswith("diff --git "):
            rest = line[len("diff --git "):]
            match = re.match(r'^"?(a/.+?)"? "?(b/.+?)"?$', rest)
            paths.extend(match.groups() if match else rest.split(" "))
            continue
        else:
            continue
        if raw.strip() != "/dev/null":
            paths.append(raw)
    return paths


def diff_touched_paths(diff_text: str, protected: Iterable[ProtectedPath]) -> tuple[str, ...]:
    """Names of the protected entries a diff touches, plus suspicious paths.

    A touched path matches an entry when it equals it or lies inside it. ``a/`` and ``b/``
    prefixes, ``.``/``..`` segments and double slashes are normalized; both the raw and the
    prefix-stripped path are checked. A path that escapes the root, or an absolute path that
    is not a protected one, is reported as ``"<escapes root>: <path>"``. Heuristic: only
    paths declared in headers are read.
    """
    entries = list(protected)
    hits: list[str] = []

    def add(item: str) -> None:
        if item not in hits:
            hits.append(item)

    for raw in _header_paths(diff_text):
        candidates = [_normalize(raw)]
        stripped = re.sub(r"^[ab]/", "", raw.strip().strip('"').replace("\\", "/"))
        if stripped != raw.strip().strip('"'):
            candidates.append(_normalize(stripped))
        for cand in candidates:
            if cand.startswith("/"):
                inside = [e for e in entries if _contains(str(e.path.resolve()), cand)]
                for e in inside:
                    add(e.name)
                if not inside:
                    add(f"<escapes root>: {raw.strip()}")
            elif cand == ".." or cand.startswith("../"):
                add(f"<escapes root>: {raw.strip()}")
            else:
                for e in entries:
                    if _contains(e.name, cand):
                        add(e.name)
    return tuple(hits)


def _contains(entry: str, touched: str) -> bool:
    entry, touched = posixpath.normpath(entry).casefold(), touched.casefold()
    return touched == entry or touched.startswith(entry + "/")


def make_readonly_copies(protected: Iterable[ProtectedPath], dest: Path) -> tuple[ProtectedPath, ...]:
    """Copy the entries with ``expose=True`` under ``dest/<name>``: files 0o444, folders 0o555."""
    dest = Path(dest)
    copies = []
    for entry in protected:
        if not entry.expose:
            continue
        target = dest / entry.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.path.is_dir():
            shutil.copytree(entry.path, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(entry.path, target)
        copies.append(ProtectedPath(name=entry.name, path=target, expose=True))
    for root, dirs, files in os.walk(dest, topdown=False):
        for name in files:
            os.chmod(Path(root, name), 0o444)
        for name in dirs:
            os.chmod(Path(root, name), 0o555)
    os.chmod(dest, 0o555)
    return tuple(copies)


def make_writable(path: Path) -> None:
    """Restore owner permissions under ``path`` so it can be removed. Symlinks are not followed."""
    path = Path(path)
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return
    if stat.S_ISLNK(mode):
        return
    if stat.S_ISDIR(mode):
        os.chmod(path, 0o700)
        for child in os.listdir(path):
            make_writable(path / child)
    else:
        os.chmod(path, 0o600)
