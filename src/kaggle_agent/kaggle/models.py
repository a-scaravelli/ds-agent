"""Value types returned by the Kaggle client (design section 10). Standard library only."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType


@dataclass(frozen=True)
class DownloadedFile:
    """One downloaded file with its size in bytes and SHA-256."""

    name: str
    size: int
    sha256: str


def combined_hash(files: tuple[DownloadedFile, ...]) -> str:
    """SHA-256 over the sorted names and file hashes: stable across directories and runs."""
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda f: f.name):
        digest.update(f"{file.name}\0{file.sha256}\0".encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DownloadResult:
    """Where the data went and what it contains; ``data_hash`` goes to execution records."""

    directory: str
    files: tuple[DownloadedFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))

    @property
    def data_hash(self) -> str:
        return combined_hash(self.files)


@dataclass(frozen=True)
class CompetitionInfo:
    """Description, rules, evaluation and data description as text for the model to summarize.

    ``pages`` maps every page name, as Kaggle returns it, to its content, so
    pages such as the FAQ are not lost.
    """

    description: str
    rules: str
    evaluation: str
    data_description: str
    pages: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", MappingProxyType(dict(self.pages)))


@dataclass(frozen=True)
class SubmissionSpec:
    """What a valid submission looks like, taken from the sample submission."""

    columns: tuple[str, ...]
    id_column: str
    expected_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "expected_ids", tuple(self.expected_ids))


@dataclass(frozen=True)
class SubmissionRecord:
    """A submission as Kaggle reports it. ``public_score`` is None until it completes."""

    submission_id: str
    competition: str
    file_sha256: str
    message: str
    submitted_at: datetime
    status: str
    public_score: float | None = None


@dataclass(frozen=True)
class DailyLimit:
    """Submissions allowed per UTC day, used and left; ``resets_at`` is the next UTC midnight."""

    limit: int
    used: int
    remaining: int
    resets_at: datetime


STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"
