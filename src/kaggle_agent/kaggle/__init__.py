"""Kaggle integration: download, rules, submission, score (design section 10).

The real adapter (``kaggle_agent.kaggle.real``) is not imported here, so the
package works without the third-party library and never authenticates on import.
"""

from .credentials import Credentials, load_credentials
from .errors import (
    CompetitionNotFoundError,
    CredentialsNotFoundError,
    DailyLimitReachedError,
    InsecureCredentialsError,
    KaggleError,
    RulesNotAcceptedError,
    ScoreTimeoutError,
    SubmissionFormatError,
)
from .fake import FakeCompetition, FakeKaggleClient
from .interface import KaggleClient
from .models import (
    CompetitionInfo,
    DailyLimit,
    DownloadedFile,
    DownloadResult,
    SubmissionRecord,
    SubmissionSpec,
)
from .recording import record_score, record_submission
from .submission import DailyLimitGuard, validate_submission

__all__ = [
    "CompetitionInfo",
    "CompetitionNotFoundError",
    "Credentials",
    "CredentialsNotFoundError",
    "DailyLimit",
    "DailyLimitGuard",
    "DailyLimitReachedError",
    "DownloadResult",
    "DownloadedFile",
    "FakeCompetition",
    "FakeKaggleClient",
    "InsecureCredentialsError",
    "KaggleClient",
    "KaggleError",
    "RulesNotAcceptedError",
    "ScoreTimeoutError",
    "SubmissionFormatError",
    "SubmissionRecord",
    "SubmissionSpec",
    "load_credentials",
    "record_score",
    "record_submission",
    "validate_submission",
]
