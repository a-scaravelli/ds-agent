"""Adapter to real Kaggle on top of the official ``kaggle`` package (design section 10).

This is the only module of the subpackage that touches the third-party
library and the network, and it is never exercised by the tests: they use
``FakeKaggleClient`` or a stub in place of the library.

Importing the package ``kaggle`` runs ``api.authenticate()`` in its
``__init__``, which reads credentials with its own lookup and calls the
network. That would bypass the documented order of ``credentials.py``, so
the adapter loads only the module with the ``KaggleApi`` class, with the
package ``__init__`` skipped, and hands it the credentials resolved by
``load_credentials`` directly on the object. ``os.environ`` is never
modified, so child processes (the future sandbox) do not inherit the secret.

Library errors are mapped to ``KaggleError`` subclasses with sanitized
messages: the text of the library exception is never copied, because it may
quote a token.

Skipping the package ``__init__`` has a cost: ``kaggle_api_extended.py`` does
``import kaggle`` (line 59 in the pinned ``kaggle==2.2.4``), so the module
global ``kaggle`` is the placeholder, and the library reads
``kaggle.__version__`` from it in ``_check_response_version`` (about lines 8431
and 9122), the response processor that runs on every HTTP response. Without
that attribute every real call failed after the response arrived. This was
found by the manual smoke test. In 2.2.4 the only attributes read from the package
object are ``__version__`` and ``kaggle.benchmarks.*`` (lines ~9443-9940, used
only by the benchmark-task methods, which this adapter never calls); the other
``kaggle.xxx`` hits are strings, imports or a local variable named ``kaggle``
bound by ``with ... as kaggle``. The placeholder therefore carries only
``__version__``, read from the installed distribution metadata. This depends on the internals of the pinned version and must be
rechecked, with the source-scan test, whenever the pin changes.
"""

import csv
import importlib
import importlib.metadata
import importlib.util
import logging
import sys
import tempfile
import time as _time
import types
import zipfile
from collections.abc import Callable, Mapping
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from .credentials import Credentials, load_credentials
from .errors import (
    CompetitionNotFoundError,
    KaggleError,
    RulesNotAcceptedError,
)
from .interface import DEFAULT_DOWNLOAD_ROOT, KaggleClient, _utc_now, sha256_of_file
from .models import (
    STATUS_PENDING,
    CompetitionInfo,
    DailyLimit,
    DownloadedFile,
    DownloadResult,
    SubmissionRecord,
    SubmissionSpec,
)

_LOG = logging.getLogger(__name__)
_API_MODULE = "kaggle.api.kaggle_api_extended"
_SUBMISSIONS_PAGE_SIZE = 100


def load_api_class():
    """Return the library's ``KaggleApi`` class without running the package ``__init__``.

    Raises ``KaggleError`` if the library is not installed.
    """
    try:
        if "kaggle" in sys.modules and sys.modules["kaggle"] is None:
            raise ImportError("kaggle is blocked")
        if _API_MODULE in sys.modules or "kaggle" in sys.modules:
            module = importlib.import_module(_API_MODULE)
        else:
            spec = importlib.util.find_spec("kaggle")
            if spec is None or not spec.submodule_search_locations:
                raise ImportError("kaggle is not installed")
            # a placeholder parent lets the submodule import without the
            # package __init__ (which would authenticate and call the network)
            placeholder = types.ModuleType("kaggle")
            placeholder.__path__ = list(spec.submodule_search_locations)
            # the only attribute the library reads from the package object;
            # a missing distribution raises PackageNotFoundError, a
            # ModuleNotFoundError, reported below as "not installed"
            placeholder.__version__ = importlib.metadata.version("kaggle")
            sys.modules["kaggle"] = placeholder
            try:
                module = importlib.import_module(_API_MODULE)
            finally:
                del sys.modules["kaggle"]
        return module.KaggleApi
    except ImportError:
        raise KaggleError(
            "the 'kaggle' library is not installed: run pip install -e . to install it"
        ) from None


def _status_code(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class RealKaggleClient(KaggleClient):
    """Client for the real Kaggle API. Needs credentials; see ``load_credentials``."""

    def __init__(
        self,
        credentials: Credentials | None = None,
        *,
        api_class: Callable[[], object] | None = None,
        env: Mapping[str, str] | None = None,
        home: Path | str | None = None,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        super().__init__(clock=clock, sleep=sleep)
        # the library first (no authentication), then the credentials
        factory = api_class if api_class is not None else load_api_class()
        credentials = credentials or load_credentials(env=env, home=home)
        self._api = factory()
        token, username, key = credentials.reveal_for_adapter()
        values = {}
        if token:
            values[self._api.CONFIG_NAME_TOKEN] = token
        else:
            values[self._api.CONFIG_NAME_USER] = username
            values[self._api.CONFIG_NAME_KEY] = key
        self._api.config_values = values
        _LOG.debug("Kaggle client ready, credentials source: %s", credentials.source)

    # -- error mapping -------------------------------------------------------

    def _call(self, competition: str, function: Callable, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KaggleError:
            raise
        except Exception as exc:
            status = _status_code(exc)
            if status == 404:
                raise CompetitionNotFoundError(f"competition '{competition}' not found") from None
            if status == 403:
                raise RulesNotAcceptedError(competition) from None
            if status == 401:
                raise KaggleError("Kaggle rejected the credentials (HTTP 401)") from None
            name = getattr(function, "__name__", "call")
            detail = f"HTTP {status}" if status else type(exc).__name__
            raise KaggleError(f"Kaggle request failed ({name}: {detail})") from None

    # -- reads ---------------------------------------------------------------

    def download(self, competition: str, dest_dir: Path | str | None = None) -> DownloadResult:
        target = Path(dest_dir) if dest_dir is not None else DEFAULT_DOWNLOAD_ROOT / competition
        target.mkdir(parents=True, exist_ok=True)
        before = {p.name for p in target.iterdir()}
        self._call(
            competition,
            self._api.competition_download_files,
            competition,
            path=str(target),
            force=True,
            quiet=True,
        )
        names: list[str] = []
        for archive in sorted(p for p in target.iterdir() if p.suffix == ".zip" and p.name not in before):
            names += self._extract(archive, target)
            archive.unlink()
        names += sorted(p.name for p in target.iterdir() if p.is_file() and p.name not in before and p.name not in names and p.suffix != ".zip")
        files = tuple(
            DownloadedFile(name, (target / name).stat().st_size, sha256_of_file(target / name))
            for name in sorted(set(names))
        )
        return DownloadResult(str(target), files)

    @staticmethod
    def _extract(archive: Path, target: Path) -> list[str]:
        root = target.resolve()
        names = []
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                out = (root / member.filename).resolve()
                if root not in out.parents:
                    raise KaggleError("the downloaded archive contains an unsafe path")
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                names.append(out.relative_to(root).as_posix())
        return names

    def _page(self, competition: str, name: str) -> str:
        pages = self._call(competition, self._api.competition_list_pages, competition, page_name=name)
        for page in pages or []:
            if page.name == name:
                return page.content or ""
        return ""

    def get_competition_info(self, competition: str) -> CompetitionInfo:
        return CompetitionInfo(
            description=self._page(competition, "description"),
            rules=self._page(competition, "rules"),
            evaluation=self._page(competition, "evaluation"),
        )

    def get_submission_spec(self, competition: str) -> SubmissionSpec:
        listing = self._call(
            competition, self._api.competition_list_files, competition, page_size=200
        )
        sample = [
            f.name
            for f in listing.files
            if "sample" in f.name.lower() and "submission" in f.name.lower()
        ]
        if not sample:
            raise KaggleError(f"competition '{competition}' has no sample submission file")
        with tempfile.TemporaryDirectory() as tmp:
            self._call(
                competition,
                self._api.competition_download_file,
                competition,
                sample[0],
                path=tmp,
                force=True,
                quiet=True,
            )
            files = sorted(Path(tmp).glob("*"))
            if not files:
                raise KaggleError("the sample submission could not be downloaded")
            with open(files[0], newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        if len(rows) < 2:
            raise KaggleError("the sample submission is empty")
        columns = tuple(rows[0])
        return SubmissionSpec(columns, columns[0], tuple(row[0] for row in rows[1:]))

    def daily_limit(self, competition: str) -> DailyLimit:
        limits = self._call(
            competition, self._api.competition_get_submission_limits, competition
        )
        now = self._clock().astimezone(timezone.utc)
        midnight = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
        return DailyLimit(
            limit=limits.num_today + limits.num_allowed_now,
            used=limits.num_today,
            remaining=limits.num_allowed_now,
            resets_at=midnight + timedelta(days=1),
        )

    def _record(self, competition: str, item, file_sha256: str = "") -> SubmissionRecord:
        try:
            score = float(item.public_score) if item.public_score not in (None, "") else None
        except ValueError:
            score = None
        status = item.status.name.lower()
        return SubmissionRecord(
            submission_id=str(item.ref),
            competition=competition,
            file_sha256=file_sha256,
            message=item.description or "",
            submitted_at=_as_utc(item.date),
            status=status,
            public_score=score if status == "complete" else None,
        )

    def get_submission(self, competition: str, submission_id: str) -> SubmissionRecord:
        items = self._call(
            competition,
            self._api.competition_submissions,
            competition,
            page_size=_SUBMISSIONS_PAGE_SIZE,
        )
        for item in items or []:
            if str(item.ref) == str(submission_id):
                return self._record(competition, item)
        raise KaggleError(f"submission '{submission_id}' not found in the list of Kaggle")

    # -- submission ----------------------------------------------------------

    def _send(
        self, competition: str, file: Path, message: str, file_sha256: str
    ) -> SubmissionRecord:
        response = self._call(
            competition,
            self._api.competition_submit,
            str(file),
            message,
            competition,
            quiet=True,
        )
        if not getattr(response, "ref", 0):
            raise KaggleError("Kaggle did not confirm the submission: nothing was recorded")
        return SubmissionRecord(
            submission_id=str(response.ref),
            competition=competition,
            file_sha256=file_sha256,
            message=message,
            submitted_at=self._clock(),
            status=STATUS_PENDING,
        )
