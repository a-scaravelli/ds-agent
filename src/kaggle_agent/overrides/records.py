"""The decision kinds of the DS overrides, their payload validation and the risky-action table."""

import re

from kaggle_agent.proposals import Steering, TryThis, ValidationError
from kaggle_agent.proposals.errors import require_text

KINDS = ("choose_model", "bar_override", "steering", "try_this", "rerun")
# actions for which the DS must give a reason (nothing is ever blocked for a technical-free reason)
REASON_REQUIRED = ("choose_model", "bar_override", "holdout_early", "holdout_second", "fork_protocol",
                   "recover_not_sent")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_REASON = 4000
MAX_SEED_OFFSET = 1000


def attempt_id(value: object, field: str = "attempt") -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValidationError(field, "must be the id of an attempt")
    return value


def reason_text(body: dict, required: bool) -> str:
    value = body.get("reason")
    if value is None or value == "":
        if required:
            raise ValidationError("reason", "is required: say why you override the program")
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("reason", "must be a non-empty text")
    if len(value) > MAX_REASON:
        raise ValidationError("reason", f"must be at most {MAX_REASON} characters")
    return value.strip()


def _count(body: dict, key: str) -> int:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(key, "must be a whole number of at least 1")
    return value


def normalize_payload(kind: str, body: dict) -> tuple[dict, object, str]:
    """(extra fields of the record, address, reason) of a decision body; ``ValidationError`` when invalid."""
    if kind not in KINDS:
        raise ValidationError("kind", f"unknown decision kind (known: {', '.join(KINDS)})")
    reason = reason_text(body, kind in REASON_REQUIRED)
    if kind == "choose_model":
        return {"attempt": attempt_id(body.get("attempt")), "reason": reason}, None, reason
    if kind == "bar_override":
        scope = body.get("scope", "round")
        if scope not in ("round", "cycle"):
            raise ValidationError("scope", "must be round or cycle")
        number = _count(body, "round")
        extra = {"scope": scope, "round": number, "reason": reason}
        if scope == "round":
            extra["attempt"] = attempt_id(body.get("attempt"))
        return extra, number, reason
    if kind == "steering":
        if body.get("clear") is True:
            return {"clear": True, "steering": Steering().to_dict()}, None, reason
        raw = body.get("steering")
        if not isinstance(raw, dict):
            raise ValidationError("steering", "is required (or clear: true)")
        steering = Steering.from_dict(raw)
        if steering.empty:
            raise ValidationError("steering", "is empty: set something, or clear the steering")
        return {"clear": False, "steering": steering.to_dict()}, None, reason
    if kind == "try_this":
        raw = body.get("request")
        if not isinstance(raw, dict):
            raise ValidationError("request", "is required: the hypothesis and optional constraints")
        number = _count(body, "from_round")
        return {"request": TryThis.from_dict(raw).to_dict(), "from_round": number}, number, reason
    offset = body.get("seed_offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= MAX_SEED_OFFSET:
        raise ValidationError("seed_offset", f"must be a whole number from 0 to {MAX_SEED_OFFSET}")
    return {"attempt": attempt_id(body.get("attempt")), "seed_offset": offset}, None, reason


def records_for_cycle(records: list[dict], cycle_id: str, first_cycle_id: str) -> list[dict]:
    """Records of one cycle; a record without ``cycle`` (an older file) belongs to the first cycle."""
    return [r for r in records if r.get("cycle", first_cycle_id) == cycle_id]


def require_text_value(name: str, value: object) -> str:
    return require_text(name, value)
