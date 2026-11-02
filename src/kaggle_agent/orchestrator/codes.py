"""Exit codes of the command line and the result type of the commands."""

from dataclasses import dataclass

FINISHED = 0          # the cycle finished (or the submit succeeded / the dry run ended)
ERROR = 1             # any other error
USAGE = 2             # usage or configuration error, or a project that does not match
WAITING = 10          # the cycle waits for the lead
SUSPENDED = 11        # quota or rate limit: run resume later
LEAD_INVALID = 12     # the lead file is not valid
CRASHED = 13          # unexpected crash, the registry is intact: run resume
NOT_ACCEPTED = 20     # submit: the review was not accepted yet
INVALID_FILE = 21     # submit: the submission file is not valid
DAILY_LIMIT = 22      # submit: no submissions left today
SCORE_PENDING = 23    # submit: sent, the score is still pending
MARKER = 24           # submit: a previous send may have reached Kaggle

ALL = (FINISHED, ERROR, USAGE, WAITING, SUSPENDED, LEAD_INVALID, CRASHED, NOT_ACCEPTED, INVALID_FILE,
       DAILY_LIMIT, SCORE_PENDING, MARKER)


@dataclass(frozen=True)
class CommandResult:
    """Exit code and the lines printed by a command."""

    code: int
    lines: tuple[str, ...]
