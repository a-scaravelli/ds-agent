"""Replaceable storage interface of the registry and its read models (design section 7).

Invariants every implementation must keep:

- Retries do not consume iterations: an attempt gets one iteration number,
  however many execution records it holds.
- The outcome of an attempt is the outcome of its last execution. An open
  attempt has no outcome; only closed attempts appear in outcome and failure
  queries.
- ``add_execution`` accepts a new execution only on an open attempt that has
  no execution yet or whose last execution is ``failed`` with ``code_error``
  (a fix retry), and only while the count stays within 1 + ``max_fix_retries``
  of its cycle.
- ``close_attempt`` raises ``RegistryError`` if the attempt has no
  executions, is already closed, or its last execution is ``failed/code_error``
  and the attempt holds fewer than 1 + ``max_fix_retries`` executions. So an
  attempt-level ``code_error`` always means "still failing after the cap",
  while on an intermediate execution it only means "this run crashed".
- The fingerprint is unique across all attempts, whatever cycle and outcome.
  It is checked only in ``add_attempt``, never between executions.
- The protocol hash of a cycle is written once; a different second value
  raises ``RegistryError``.
- A cycle has at most one holdout session, never reopened (even after it is
  closed). A holdout read needs an open session. Reads and comparisons are
  append-only. Which attempt may be read and what is compared is policy of
  the evaluation engine, not of the registry.
- A rejected call leaves the registry unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kaggle_agent.proposals import Attempt, Suggestion

from .records import ComparisonRecord, Cycle, ExecutionRecord, HoldoutRead, HoldoutSession


@dataclass(frozen=True)
class AttemptSummary:
    """One row of the list of attempts."""

    attempt_id: str
    cycle_id: str
    iteration: int
    parent_id: str | None
    phase: str
    outcome: str | None
    failure_kind: str | None


@dataclass(frozen=True)
class AttemptDetail:
    """Full traceability of an attempt; outcome fields come from the last execution."""

    attempt: Attempt
    cycle_id: str
    iteration: int
    fingerprint: str
    closed: bool
    conclusion: str | None
    executions: tuple[ExecutionRecord, ...]
    outcome: str | None
    failure_kind: str | None
    reason: str | None


@dataclass(frozen=True)
class MetricPoint:
    """Metric of one attempt on a partition, for the series per iteration."""

    iteration: int
    attempt_id: str
    mean: float
    dispersion: float


@dataclass(frozen=True)
class FailureRecord:
    """An item of the memory of failures: what was tried, outcome, why."""

    attempt_id: str
    what_changed: str
    outcome: str
    failure_kind: str
    reason: str


class Registry(ABC):
    """Storage of cycles, attempts, execution records and suggestions."""

    @abstractmethod
    def add_cycle(self, cycle: Cycle) -> None:
        """Store a cycle; a duplicate id raises ``RegistryError``."""

    @abstractmethod
    def get_cycle(self, cycle_id: str) -> Cycle:
        """Return a cycle; unknown id raises ``NotFoundError``."""

    @abstractmethod
    def set_cycle_state(self, cycle_id: str, state: str) -> None:
        """Persist a new non-empty state of the cycle."""

    @abstractmethod
    def add_attempt(self, cycle_id: str, attempt: Attempt, fingerprint: str) -> int:
        """Store a proposal and return its iteration number in the cycle.

        Raises ``NotFoundError`` for an unknown cycle or ``parent_id``,
        ``RegistryError`` for a reused attempt id and ``DuplicateAttemptError``
        when the fingerprint belongs to any earlier attempt.
        """

    @abstractmethod
    def add_execution(self, execution: ExecutionRecord) -> int:
        """Store an execution and return its sequence number (from 1).

        Enforces the fix-retry rules stated in the module docstring.
        """

    @abstractmethod
    def close_attempt(self, attempt_id: str, conclusion: str) -> None:
        """Close an attempt with the conclusion of the analysis (what was learned)."""

    @abstractmethod
    def list_attempts(
        self, cycle_id: str | None = None, outcome: str | None = None
    ) -> list[AttemptSummary]:
        """Attempts ordered by iteration and then insertion, optionally filtered.

        Filtering by ``outcome`` returns closed attempts only.
        """

    @abstractmethod
    def get_attempt(self, attempt_id: str) -> AttemptDetail:
        """Detail of an attempt with its ordered executions."""

    @abstractmethod
    def metric_series(self, cycle_id: str, partition: str) -> list[MetricPoint]:
        """Metric per iteration, from the last execution that has the partition."""

    @abstractmethod
    def list_failures(self, failure_kind: str | None = None) -> list[FailureRecord]:
        """Closed attempts that did not succeed, optionally of one ``failure_kind``."""

    @abstractmethod
    def add_suggestion(self, suggestion: Suggestion) -> None:
        """Store a suggestion; its origin and spawned cycles must exist."""

    @abstractmethod
    def link_cycle(self, suggestion_id: str, cycle_id: str) -> None:
        """Record that a cycle was spawned from a suggestion."""

    @abstractmethod
    def get_suggestion(self, suggestion_id: str) -> Suggestion:
        """Return a suggestion with its spawned cycles."""

    @abstractmethod
    def list_suggestions(self) -> list[Suggestion]:
        """All suggestions in insertion order."""

    @abstractmethod
    def cycles_of_suggestion(self, suggestion_id: str) -> list[Cycle]:
        """Cycles spawned from a suggestion, in link order."""

    @abstractmethod
    def set_protocol_hash(self, cycle_id: str, protocol_hash: str) -> None:
        """Store the frozen protocol hash; writing a different value again raises."""

    @abstractmethod
    def open_holdout_session(self, cycle_id: str, opened_at: str) -> None:
        """Open the single holdout session of a cycle; a second opening raises."""

    @abstractmethod
    def close_holdout_session(self, cycle_id: str, closed_at: str) -> None:
        """Close the open session; no session or already closed raises ``RegistryError``."""

    @abstractmethod
    def get_holdout_session(self, cycle_id: str) -> HoldoutSession | None:
        """The session of a cycle, or None if it was never opened."""

    @abstractmethod
    def add_holdout_read(self, read: HoldoutRead) -> None:
        """Log a holdout read; requires an open session and an existing attempt."""

    @abstractmethod
    def list_holdout_reads(self, cycle_id: str) -> list[HoldoutRead]:
        """Holdout reads of a cycle in insertion order."""

    @abstractmethod
    def add_comparison(self, comparison: ComparisonRecord) -> None:
        """Store a comparison; cycle and both attempts must exist."""

    @abstractmethod
    def list_comparisons(self, cycle_id: str) -> list[ComparisonRecord]:
        """Comparisons of a cycle in insertion order."""

    @abstractmethod
    def count_comparisons(self, cycle_id: str) -> int:
        """Number of comparisons made in a cycle."""
