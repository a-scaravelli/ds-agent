"""Environment health cards: Kaggle credentials, Claude login, libraries, OpenMP, Python and disk.

Stdlib only and no subprocess of its own: the Claude check is asked of the ``RunController``. The
Kaggle card reports PRESENCE only, in the lookup order of ``kaggle/credentials.py``, with
``os.environ`` membership and ``os.stat``: a secret is never opened, read, shown or stored. Every card
degrades to ``unknown`` on any failure. The page renders at once with 'checking'; the browser fetches
``/api/environment`` and refreshes it (a result is cached with a time stamp).
"""

import ctypes.util
import os
import platform
import shutil
import stat
import threading
import time
from importlib import metadata
from pathlib import Path

LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "flask")
CACHE_SECONDS = 30.0

KAGGLE_STEPS = (
    "Open kaggle.com, go to Settings > API and create a token.",
    "Save it as the file ~/.kaggle/access_token (or put it in the KAGGLE_API_TOKEN environment variable "
    "of the Terminal that starts the cockpit).",
    "Run `chmod 600 ~/.kaggle/access_token` so only you can read it.",
    "Press Refresh here. The page never asks for the token and never shows it.",
)


def _file_state(path: Path) -> dict | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return {"path": f"~/.kaggle/{path.name}", "too_open": bool(info.st_mode & (stat.S_IRWXG | stat.S_IRWXO))}


def kaggle_card(env=None, home=None) -> dict:
    """Which token source exists (never its content)."""
    try:
        env = os.environ if env is None else env
        folder = (Path.home() if home is None else Path(home)) / ".kaggle"
        source, warnings = None, []
        if "KAGGLE_API_TOKEN" in env:
            source = "the KAGGLE_API_TOKEN environment variable"
        else:
            token = _file_state(folder / "access_token")
            if token:
                source = "the file ~/.kaggle/access_token"
                if token["too_open"]:
                    warnings.append("~/.kaggle/access_token is readable by others: run chmod 600 on it")
            elif "KAGGLE_USERNAME" in env and "KAGGLE_KEY" in env:
                source = "the KAGGLE_USERNAME and KAGGLE_KEY environment variables"
            else:
                legacy = _file_state(folder / "kaggle.json")
                if legacy:
                    source = "the file ~/.kaggle/kaggle.json"
                    if legacy["too_open"]:
                        warnings.append("~/.kaggle/kaggle.json is readable by others: run chmod 600 on it")
        if source:
            return {"state": "present", "source": source, "warnings": warnings, "steps": []}
        return {"state": "missing", "source": None, "warnings": [], "steps": list(KAGGLE_STEPS)}
    except Exception:
        return {"state": "unknown", "source": None, "warnings": [], "steps": []}


def claude_card(controller) -> dict:
    status = controller.claude_status()
    card = {"state": status["state"], "detail": status.get("detail"), "fix": None}
    if status["state"] != "logged_in":
        card["fix"] = "Open a normal Terminal and run: claude auth login"
    return card


def libraries_card() -> dict:
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = "unknown"
    return {"state": "ok", "versions": versions}


def openmp_card() -> dict:
    try:
        found = ctypes.util.find_library("omp") or ctypes.util.find_library("gomp") \
            or ctypes.util.find_library("iomp5")
    except Exception:
        return {"state": "unknown", "library": None}
    return {"state": "present" if found else "missing", "library": found}


def python_card() -> dict:
    try:
        return {"state": "ok", "version": platform.python_version(), "platform": platform.platform()}
    except Exception:
        return {"state": "unknown", "version": None, "platform": None}


def disk_card(project_dir) -> dict:
    try:
        target = Path(project_dir)
        while not target.exists() and target != target.parent:
            target = target.parent
        usage = shutil.disk_usage(target)
        return {"state": "ok", "free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}
    except Exception:
        return {"state": "unknown", "free_gb": None, "total_gb": None}


class Environment:
    """Builds all cards; the slow Claude check runs in a background thread and is cached."""

    def __init__(self, controller, project_dir, env=None, home=None) -> None:
        self.controller, self.project_dir, self.env, self.home = controller, project_dir, env, home
        self._claude: dict | None = None
        self._claude_at = 0.0
        self._running = False
        self._lock = threading.Lock()

    def _refresh_claude(self) -> None:
        try:
            card = claude_card(self.controller)
        except Exception:
            card = {"state": "unknown", "detail": "the check failed", "fix": "Open a normal Terminal and run: claude auth login"}
        with self._lock:
            self._claude, self._claude_at, self._running = card, time.time(), False

    def refresh(self) -> None:
        """Start a new Claude check in the background (at most one at a time)."""
        with self._lock:
            if self._running:
                return
            self._running = True
        threading.Thread(target=self._refresh_claude, daemon=True, name="claude-check").start()

    def cards(self, force: bool = False) -> dict:
        with self._lock:
            stale = self._claude is None or force or time.time() - self._claude_at > CACHE_SECONDS
            claude = self._claude
            checking = self._running
        if stale:
            self.refresh()
            checking = True
        shown = claude if claude is not None else {"state": "checking", "detail": None, "fix": None}
        if checking and claude is None:
            shown = {"state": "checking", "detail": None, "fix": None}
        return {
            "kaggle": kaggle_card(self.env, self.home), "claude": shown, "libraries": libraries_card(),
            "openmp": openmp_card(), "python": python_card(), "disk": disk_card(self.project_dir),
            "checked_at": time.time(),
        }
