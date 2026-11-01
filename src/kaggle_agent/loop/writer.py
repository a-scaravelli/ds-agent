"""The open interface that turns role outputs into a bundle folder.

Bundle production is deliberately not part of the typed output of the roles (they return
hypotheses, not code). A ``BundleWriter`` is given the attempt and the outputs of the roles and
produces a folder that satisfies the bundle contract. A Claude-driven writer needs tools and a
workspace and is a later step; here there is only the interface and a fake for tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.agents import DataFindings
from kaggle_agent.proposals import Attempt


@dataclass(frozen=True)
class RoleOutputs:
    """What the roles produced for one candidate, passed on by the loop."""

    findings: DataFindings
    feature_engineering: Attempt
    modeling: Attempt


@dataclass(frozen=True)
class FixRequest:
    """Why a new bundle is asked for the same attempt: the failure of the previous run."""

    retry: int
    reason: str
    previous_bundle_id: str


@dataclass(frozen=True)
class BundleRequest:
    """One request to the writer.

    ``folder`` is a new, empty-or-absent folder the writer must fill; ``bundle_id`` is the id the
    manifest must carry. ``fix`` is set when the previous bundle of the attempt crashed.
    """

    attempt: Attempt
    outputs: RoleOutputs
    bundle_id: str
    folder: Path
    fix: FixRequest | None = None


class BundleWriter(ABC):
    """Produces a bundle folder for an attempt."""

    @abstractmethod
    def write(self, request: BundleRequest) -> Path:
        """Write the bundle under ``request.folder`` (or elsewhere) and return its folder."""
