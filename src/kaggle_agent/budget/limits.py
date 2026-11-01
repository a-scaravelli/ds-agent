"""The parameters of one cycle (design section 11), validated and serialisable."""

from dataclasses import dataclass, fields

from kaggle_agent.proposals import ValidationError

DEFAULT_ROUNDS = 5
DEFAULT_CANDIDATES = 3
DEFAULT_STAGNATION_ROUNDS = 2
DEFAULT_PAUSE_EVERY = 1
DEFAULT_FIX_RETRIES = 3
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_MAX_REJECTIONS = 3

# stored in ``Cycle.parameters``; N, K and the fix cap have their own Cycle fields
_STORED = ("candidates", "pause_every", "max_consecutive_failures", "max_rejections")


def _positive(field: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(field, "must be an integer")
    if value < minimum:
        raise ValidationError(field, f"must be at least {minimum}")
    return value


@dataclass(frozen=True)
class CycleLimits:
    """Caps of the level 1 cycle. Every default is provisional and configurable.

    - ``rounds`` (N, default 5): rounds of the iteration phase, the setup round excluded.
    - ``candidates`` (C, default 3): candidates per round.
    - ``stagnation_rounds`` (K, default 2): rounds without real improvement that stop the cycle.
    - ``pause_every`` (M, default 1): the cycle pauses for guidance after every M-th round.
    - ``fix_retries`` (default 3): code-fix retries per attempt.
    - ``max_consecutive_failures`` (default 3): attempts closed as failed in a row that stop the cycle.
    - ``max_rejections`` (default 3): consecutive rejected proposals of one candidate before it is
      recorded as inconclusive.
    """

    rounds: int = DEFAULT_ROUNDS
    candidates: int = DEFAULT_CANDIDATES
    stagnation_rounds: int = DEFAULT_STAGNATION_ROUNDS
    pause_every: int = DEFAULT_PAUSE_EVERY
    fix_retries: int = DEFAULT_FIX_RETRIES
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_rejections: int = DEFAULT_MAX_REJECTIONS

    def __post_init__(self) -> None:
        for field in fields(self):
            minimum = 0 if field.name == "fix_retries" else 1
            _positive(field.name, getattr(self, field.name), minimum)

    def to_dict(self) -> dict:
        """The values kept in ``Cycle.parameters`` (the others live in dedicated Cycle fields)."""
        return {name: getattr(self, name) for name in _STORED}

    @classmethod
    def from_dict(cls, data: dict, **dedicated) -> "CycleLimits":
        """Build limits from ``to_dict`` output plus the values of the dedicated Cycle fields."""
        if not isinstance(data, dict):
            raise ValidationError("parameters", "must be an object")
        unknown = sorted(set(data) - set(_STORED))
        if unknown:
            raise ValidationError(unknown[0], f"unknown parameter (allowed: {', '.join(_STORED)})")
        return cls(**{**data, **dedicated})

    @classmethod
    def resolve(cls, cycle, injected: "CycleLimits | None" = None) -> "CycleLimits":
        """The limits a cycle runs with.

        N, K and the fix cap always come from the cycle's own fields. The rest comes from
        ``cycle.parameters`` when set; the injected limits are only a fallback for a cycle that
        has none (an older cycle, or the first start).
        """
        injected = injected or cls()
        dedicated = {
            "rounds": cycle.iterations,
            "stagnation_rounds": cycle.stagnation_k,
            "fix_retries": cycle.max_fix_retries,
        }
        stored = cycle.parameters if cycle.parameters is not None else injected.to_dict()
        return cls.from_dict(stored, **dedicated)
