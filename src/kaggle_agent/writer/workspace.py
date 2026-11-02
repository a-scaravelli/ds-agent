"""The isolated staging workspace of one writer call and the snapshots that detect escapes.

Standard library only. Detection is best effort: it covers the workspace root, the protected paths
and the request folder, not the whole disk.
"""

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from kaggle_agent.sandbox import ProtectedPath, hash_paths

from .errors import PreviousBundleError, WorkspaceViolationError

BUNDLE_SUBFOLDER = "bundle"
IGNORED_ROOT_ENTRIES = (".claude",)  # state the CLI may keep in its working directory


def tree_state(root: Path, exclude: Path | None = None) -> dict[str, str]:
    """Relative path -> content hash (files), link target (links) or ``<dir>`` for a whole tree."""
    state: dict[str, str] = {}
    if not root.exists():
        return state
    for current, dirs, files in os.walk(root):
        here = Path(current)
        if exclude is not None and here == exclude:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if exclude is None or here / d != exclude]
        for name in dirs:
            path = here / name
            rel = path.relative_to(root).as_posix()
            state[rel] = f"<link {os.readlink(path)}>" if path.is_symlink() else "<dir>"
        for name in files:
            path = here / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                state[rel] = f"<link {os.readlink(path)}>"
            else:
                state[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return state


class Staging:
    """A fresh working directory under ``workspace_root`` with an empty ``bundle/`` folder.

    Use it as a context manager: the directory is always removed on exit. ``check`` compares the
    state taken at the start with the current one and raises ``WorkspaceViolationError``.
    """

    def __init__(self, workspace_root: Path, request_folder: Path,
                 protected: Iterable[ProtectedPath]) -> None:
        self.root = Path(workspace_root).resolve()
        self.request_folder = Path(request_folder)
        self.protected = tuple(protected)
        self.directory: Path | None = None

    @property
    def bundle(self) -> Path:
        return self.directory / BUNDLE_SUBFOLDER

    def __enter__(self) -> "Staging":
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="writer-", dir=self.root)).resolve()
        self.bundle.mkdir()
        self._before = self._snapshot()
        return self

    def __exit__(self, *exc) -> None:
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)

    def _snapshot(self) -> tuple[dict, dict, dict]:
        return (
            hash_paths(self.protected, missing_ok=True),
            tree_state(self.root, exclude=self.directory),
            tree_state(self.request_folder),
        )

    def check(self) -> None:
        """Raise ``WorkspaceViolationError`` if anything changed outside ``bundle/``."""
        extra = sorted(
            p.name for p in self.directory.iterdir()
            if p.name != BUNDLE_SUBFOLDER and p.name not in IGNORED_ROOT_ENTRIES
        )
        if extra:
            raise WorkspaceViolationError(
                f"files written in the working directory outside {BUNDLE_SUBFOLDER}/: {', '.join(extra)}"
            )
        protected, outside, folder = self._snapshot()
        before_protected, before_outside, before_folder = self._before
        changed = sorted(n for n in before_protected if protected.get(n) != before_protected[n])
        if changed:
            raise WorkspaceViolationError(
                f"protected path changed during the Claude call: {', '.join(changed)}"
            )
        if outside != before_outside:
            names = sorted(k for k in outside.keys() | before_outside.keys()
                           if outside.get(k) != before_outside.get(k))
            raise WorkspaceViolationError(
                f"files changed outside the staging directory: {', '.join(names[:5])}"
            )
        if folder != before_folder:
            raise WorkspaceViolationError(
                f"the request folder was written during the Claude call: {self.request_folder}"
            )


def read_previous_bundle(folder: Path, cap: int) -> dict[str, str]:
    """The text of ``bundle.json`` and every ``.py`` file of a previous bundle, capped in size.

    Read only. Raises ``PreviousBundleError`` when the folder or its manifest is missing.
    """
    folder = Path(folder)
    if not folder.is_dir() or not (folder / "bundle.json").is_file():
        raise PreviousBundleError(f"previous bundle not readable: {folder}")
    names = sorted(
        p.relative_to(folder).as_posix() for p in folder.rglob("*")
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
        and p.relative_to(folder).parts[0] != "model"
        and (p.suffix == ".py" or p.name == "bundle.json")
    )
    files, used = {}, 0
    for name in names:
        try:
            text = (folder / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise PreviousBundleError(f"previous bundle not readable: {folder}") from None
        room = max(cap - used, 0)
        files[name] = text if len(text) <= room else text[:room] + "\n... (truncated)"
        used += min(len(text), room)
    return files
