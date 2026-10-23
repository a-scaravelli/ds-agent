"""Errors of the Kaggle client. No message may contain a credential."""


class KaggleError(Exception):
    """Base class of every error raised by the Kaggle client."""


class CredentialsNotFoundError(KaggleError):
    """No usable credentials in any of the places searched."""


class InsecureCredentialsError(KaggleError):
    """A credentials file is readable by group or others (it must be ``600``)."""


class RulesNotAcceptedError(KaggleError):
    """The rules of the competition were not accepted yet.

    Accepting them is a manual step of the human, on the competition website.
    """

    def __init__(self, competition: str) -> None:
        self.competition = competition
        super().__init__(
            f"the rules of competition '{competition}' are not accepted: "
            "accept them on the competition website, then retry"
        )


class SubmissionFormatError(KaggleError):
    """The submission file does not match the sample submission; the reason is showable."""


class DailyLimitReachedError(KaggleError):
    """No submissions are left today (UTC); nothing was sent."""


class ScoreTimeoutError(KaggleError):
    """The score was still pending when the timeout elapsed."""


class CompetitionNotFoundError(KaggleError):
    """The competition does not exist (or is not visible to the account)."""
