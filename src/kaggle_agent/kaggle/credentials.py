"""Credential resolution for the Kaggle client (design section 10).

Only modules of this subpackage read credentials, and only ``real.py`` ever
asks for the secret. The secret never appears in ``repr``, ``str``, format,
pickles, logs or error messages; the result records the *kind* of source,
which is safe to log.

Lookup order, first match wins (it follows the precedence of the official
client: access token first, then the legacy username and key, where the
environment variables override ``kaggle.json``):

1. ``KAGGLE_API_TOKEN`` environment variable (the token generated with
   "Generate New Token" on the Kaggle account page);
2. ``~/.kaggle/access_token`` file holding the same token;
3. legacy ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables (both);
4. legacy ``~/.kaggle/kaggle.json`` file (``username`` and ``key``).

An empty or whitespace-only value counts as absent. A credentials file that
is readable by group or others (anything looser than ``600``) raises
``InsecureCredentialsError``; the official client only prints a warning.
``KAGGLE_CONFIG_DIR`` is not supported: the files live in ``~/.kaggle``.
"""

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from .errors import CredentialsNotFoundError, InsecureCredentialsError

SOURCE_ENV = "env"
SOURCE_ACCESS_TOKEN = "access_token"
SOURCE_LEGACY_ENV = "legacy_env"
SOURCE_LEGACY_FILE = "legacy_file"

_SEARCHED = (
    "KAGGLE_API_TOKEN, ~/.kaggle/access_token, "
    "KAGGLE_USERNAME + KAGGLE_KEY, ~/.kaggle/kaggle.json"
)


class Credentials:
    """Resolved credentials: a source kind and a secret reachable only through one method."""

    __slots__ = ("_source", "_token", "_username", "_key")

    def __init__(
        self,
        source: str,
        token: str | None = None,
        username: str | None = None,
        key: str | None = None,
    ) -> None:
        self._source = source
        self._token = token
        self._username = username
        self._key = key

    @property
    def source(self) -> str:
        """Kind of source: ``env``, ``access_token``, ``legacy_env`` or ``legacy_file``."""
        return self._source

    def reveal_for_adapter(self) -> tuple[str | None, str | None, str | None]:
        """Return ``(token, username, key)``; called only by the real adapter."""
        return self._token, self._username, self._key

    def __repr__(self) -> str:
        return f"Credentials(source={self._source!r})"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("Credentials cannot be pickled or copied")


def _clean(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _check_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise InsecureCredentialsError(
            f"{path} is readable by group or others: run 'chmod 600 {path}'"
        )


def _read_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    _check_permissions(path)
    return path.read_text(encoding="utf-8")


def load_credentials(
    env: Mapping[str, str] | None = None, home: Path | str | None = None
) -> Credentials:
    """Resolve credentials with the lookup order of the module docstring.

    ``env`` and ``home`` are injectable for tests (default ``os.environ`` and
    the user's home). Raises ``CredentialsNotFoundError`` if nothing usable
    is found and ``InsecureCredentialsError`` for a file with loose permissions.
    """
    env = os.environ if env is None else env
    home_dir = Path.home() if home is None else Path(home)
    config_dir = home_dir / ".kaggle"

    token = _clean(env.get("KAGGLE_API_TOKEN"))
    if token:
        return Credentials(SOURCE_ENV, token=token)

    token = _clean(_read_file(config_dir / "access_token"))
    if token:
        return Credentials(SOURCE_ACCESS_TOKEN, token=token)

    username, key = _clean(env.get("KAGGLE_USERNAME")), _clean(env.get("KAGGLE_KEY"))
    if username and key:
        return Credentials(SOURCE_LEGACY_ENV, username=username, key=key)

    text = _read_file(config_dir / "kaggle.json")
    if text is not None and text.strip():
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        username = _clean(data.get("username")) if isinstance(data, dict) else None
        key = _clean(data.get("key")) if isinstance(data, dict) else None
        if not (username and key):
            raise CredentialsNotFoundError(
                "~/.kaggle/kaggle.json is malformed: it must be a JSON object "
                "with 'username' and 'key'"
            )
        return Credentials(SOURCE_LEGACY_FILE, username=username, key=key)

    raise CredentialsNotFoundError(f"no Kaggle credentials found; searched: {_SEARCHED}")
