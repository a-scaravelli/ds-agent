"""Minimal environment of the child process (design section 8).

The environment is built from scratch, never copied from ``os.environ``, which is
never modified. Standard library only.
"""

from collections.abc import Mapping
from pathlib import Path

from .errors import SandboxConfigError

FIXED_PATH = "/usr/bin:/bin"
PROTECTED_DIR_VARIABLE = "SANDBOX_PROTECTED_DIR"

# Every key the child can get from the sandbox itself, besides ``extra_env``.
MINIMAL_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "HOME",
    "TMPDIR",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    PROTECTED_DIR_VARIABLE,
)

# Substrings (upper case) that make an extra variable look like a secret.
SENSITIVE_FRAGMENTS = (
    "KAGGLE", "ANTHROPIC", "CLAUDE", "TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL",
)
FORBIDDEN_NAMES = ("PYTHONPATH",)


def validate_extra_env(extra_env: Mapping[str, str]) -> None:
    """Raise ``SandboxConfigError`` for a secret-looking name, ``PYTHONPATH`` or a reserved name."""
    for name, value in extra_env.items():
        upper = name.upper()
        if upper in FORBIDDEN_NAMES:
            raise SandboxConfigError(f"extra variable {name} is not allowed: pass paths explicitly")
        if any(fragment in upper for fragment in SENSITIVE_FRAGMENTS):
            raise SandboxConfigError(f"extra variable {name} looks like a credential")
        if name in MINIMAL_KEYS:
            raise SandboxConfigError(f"extra variable {name} is set by the sandbox")
        if not isinstance(value, str) or "=" in name or not name:
            raise SandboxConfigError(f"invalid extra variable {name!r}")


def build_child_environment(
    run_dir: Path | str, extra_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """A new environment dict for the child: fixed PATH, HOME and TMPDIR inside ``run_dir``."""
    extra = dict(extra_env or {})
    validate_extra_env(extra)
    run = Path(run_dir)
    env = {
        "PATH": FIXED_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(run / "home"),
        "TMPDIR": str(run / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        PROTECTED_DIR_VARIABLE: str(run / "protected"),
    }
    env.update(extra)
    return env
