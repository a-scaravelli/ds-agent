"""The reply type of the writer: the files are the real output, the final message is a note."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class WriterReport:
    """Optional summary of what Claude wrote.

    ``from_json`` accepts ANY text (JSON, prose, prose with a code fence) and never raises, so an
    unparseable final message cannot become an ``InvalidResponseError`` that would wrongly suspend
    the cycle or trigger a validation retry. The reply text is otherwise ignored.
    """

    summary: str | None = None

    @classmethod
    def from_json(cls, text: object) -> "WriterReport":
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return cls()
        summary = data.get("summary") if isinstance(data, dict) else None
        return cls(summary if isinstance(summary, str) and summary.strip() else None)
