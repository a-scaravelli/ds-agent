"""Paired comparison against the significance threshold (design section 6).

An improvement is real only if the mean paired difference, oriented by the
direction of the protocol, exceeds ``threshold_k`` times the dispersion of the
baseline across partitions. ``threshold_k`` always comes from the protocol.
Raising the threshold with the number of comparisons is not implemented: the
engine only counts comparisons.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from kaggle_agent.proposals import ValidationError

from .protocol import Protocol


@dataclass(frozen=True)
class Comparison:
    """Outcome of one paired comparison; positive ``difference`` means candidate is better."""

    difference: float
    threshold: float
    is_real: bool
    per_partition: dict[str, float]


def dispersion(scores: Mapping[str, float] | Sequence[float]) -> float:
    """Sample standard deviation (ddof=1) of the baseline scores over its partitions."""
    values = list(scores.values()) if isinstance(scores, Mapping) else list(scores)
    if len(values) < 2:
        raise ValidationError("scores", "need at least 2 partitions to estimate the dispersion")
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def compare(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    baseline_dispersion: float,
    protocol: Protocol,
) -> Comparison:
    """Compare candidate and reference scores on the same partitions."""
    if set(candidate) != set(reference):
        raise ValidationError("scores", "candidate and reference must cover the same partitions")
    if not candidate:
        raise ValidationError("scores", "must not be empty")
    if baseline_dispersion < 0 or not np.isfinite(baseline_dispersion):
        raise ValidationError("baseline_dispersion", "must be a non-negative number")
    sign = 1.0 if protocol.direction == "maximize" else -1.0
    per_partition = {p: sign * (candidate[p] - reference[p]) for p in sorted(candidate)}
    difference = float(np.mean(list(per_partition.values())))
    threshold = float(protocol.threshold_k * baseline_dispersion)
    return Comparison(difference, threshold, bool(difference > threshold), per_partition)
