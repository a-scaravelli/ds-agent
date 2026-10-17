"""SQLite implementation of the registry, standard library only (design section 7)."""

import sqlite3
from contextlib import contextmanager

from kaggle_agent.proposals import Attempt, ShortPlan, Suggestion, ValidationError
from kaggle_agent.proposals.errors import require_text

from .errors import DuplicateAttemptError, NotFoundError, RegistryError
from .interface import (
    AttemptDetail,
    AttemptSummary,
    FailureRecord,
    MetricPoint,
    Registry,
)
from .records import FAILURE_KINDS, OUTCOMES, Cycle, ExecutionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id TEXT PRIMARY KEY,
    plan TEXT NOT NULL,
    iterations INTEGER NOT NULL,
    stagnation_k INTEGER NOT NULL,
    max_fix_retries INTEGER NOT NULL,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    iteration INTEGER NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    closed INTEGER NOT NULL DEFAULT 0,
    conclusion TEXT,
    proposal TEXT NOT NULL,
    UNIQUE (cycle_id, iteration)
);
CREATE TABLE IF NOT EXISTS executions (
    id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL REFERENCES attempts(id),
    sequence INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    failure_kind TEXT,
    record TEXT NOT NULL,
    PRIMARY KEY (attempt_id, sequence)
);
CREATE INDEX IF NOT EXISTS executions_failure_kind ON executions(failure_kind);
CREATE TABLE IF NOT EXISTS suggestions (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    origin_cycle TEXT NOT NULL REFERENCES cycles(id)
);
CREATE TABLE IF NOT EXISTS suggestion_cycles (
    suggestion_id TEXT NOT NULL REFERENCES suggestions(id),
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (suggestion_id, cycle_id)
);
"""

# the last execution of every attempt, joined to the attempt row
_LAST_EXECUTION = (
    "LEFT JOIN executions e ON e.attempt_id = a.id AND e.sequence = "
    "(SELECT MAX(sequence) FROM executions WHERE attempt_id = a.id)"
)


class SqliteRegistry(Registry):
    """Registry on a SQLite file, or in memory with the default ``':memory:'``."""

    def __init__(self, path: str = ":memory:") -> None:
        # autocommit mode: transactions are explicit, see ``_transaction``
        self._db = sqlite3.connect(str(path), isolation_level=None)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "SqliteRegistry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # -- helpers -----------------------------------------------------------

    def _one(self, sql: str, args: tuple = ()):
        return self._db.execute(sql, args).fetchone()

    def _require_cycle_row(self, cycle_id: str):
        row = self._one(
            "SELECT id, plan, iterations, stagnation_k, max_fix_retries, state "
            "FROM cycles WHERE id = ?",
            (cycle_id,),
        )
        if row is None:
            raise NotFoundError(f"cycle {cycle_id!r} does not exist")
        return row

    @staticmethod
    def _cycle(row) -> Cycle:
        return Cycle(
            id=row[0],
            plan=ShortPlan.from_json(row[1]),
            iterations=row[2],
            stagnation_k=row[3],
            max_fix_retries=row[4],
            state=row[5],
        )

    def _require_attempt_row(self, attempt_id: str):
        row = self._one(
            "SELECT id, cycle_id, iteration, fingerprint, closed, conclusion, proposal "
            "FROM attempts WHERE id = ?",
            (attempt_id,),
        )
        if row is None:
            raise NotFoundError(f"attempt {attempt_id!r} does not exist")
        return row

    def _executions(self, attempt_id: str) -> list[ExecutionRecord]:
        rows = self._db.execute(
            "SELECT record FROM executions WHERE attempt_id = ? ORDER BY sequence",
            (attempt_id,),
        ).fetchall()
        return [ExecutionRecord.from_json(r[0]) for r in rows]

    # -- cycles ------------------------------------------------------------

    def add_cycle(self, cycle: Cycle) -> None:
        with self._transaction():
            if self._one("SELECT 1 FROM cycles WHERE id = ?", (cycle.id,)):
                raise RegistryError(f"cycle {cycle.id!r} already exists")
            self._db.execute(
                "INSERT INTO cycles VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cycle.id,
                    cycle.plan.to_json(),
                    cycle.iterations,
                    cycle.stagnation_k,
                    cycle.max_fix_retries,
                    cycle.state,
                ),
            )

    def get_cycle(self, cycle_id: str) -> Cycle:
        return self._cycle(self._require_cycle_row(cycle_id))

    def set_cycle_state(self, cycle_id: str, state: str) -> None:
        require_text("state", state)
        with self._transaction():
            self._require_cycle_row(cycle_id)
            self._db.execute("UPDATE cycles SET state = ? WHERE id = ?", (state, cycle_id))

    # -- attempts ----------------------------------------------------------

    def add_attempt(self, cycle_id: str, attempt: Attempt, fingerprint: str) -> int:
        require_text("fingerprint", fingerprint)
        with self._transaction():
            self._require_cycle_row(cycle_id)
            if self._one("SELECT 1 FROM attempts WHERE id = ?", (attempt.id,)):
                raise RegistryError(f"attempt {attempt.id!r} already exists")
            if attempt.parent_id is not None and not self._one(
                "SELECT 1 FROM attempts WHERE id = ?", (attempt.parent_id,)
            ):
                raise NotFoundError(f"parent attempt {attempt.parent_id!r} does not exist")
            existing = self._one(
                "SELECT id FROM attempts WHERE fingerprint = ?", (fingerprint,)
            )
            if existing is not None:
                raise DuplicateAttemptError(existing[0], fingerprint)
            iteration = self._one(
                "SELECT COALESCE(MAX(iteration), 0) + 1 FROM attempts WHERE cycle_id = ?",
                (cycle_id,),
            )[0]
            self._db.execute(
                "INSERT INTO attempts (id, cycle_id, iteration, fingerprint, proposal) "
                "VALUES (?, ?, ?, ?, ?)",
                (attempt.id, cycle_id, iteration, fingerprint, attempt.to_json()),
            )
            return iteration

    def add_execution(self, execution: ExecutionRecord) -> int:
        with self._transaction():
            attempt = self._require_attempt_row(execution.attempt_id)
            if attempt[4]:
                raise RegistryError(f"attempt {execution.attempt_id!r} is closed")
            if self._one("SELECT 1 FROM executions WHERE id = ?", (execution.id,)):
                raise RegistryError(f"execution {execution.id!r} already exists")
            last = self._one(
                "SELECT sequence, outcome, failure_kind FROM executions "
                "WHERE attempt_id = ? ORDER BY sequence DESC LIMIT 1",
                (execution.attempt_id,),
            )
            count = last[0] if last else 0
            if last and (last[1], last[2]) != ("failed", "code_error"):
                raise RegistryError(
                    "a new execution is accepted only after a failed code_error execution"
                )
            cap = self._require_cycle_row(attempt[1])[4]
            if count + 1 > 1 + cap:
                raise RegistryError(
                    f"fix-retry cap reached: at most {1 + cap} executions per attempt"
                )
            sequence = count + 1
            self._db.execute(
                "INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    execution.id,
                    execution.attempt_id,
                    sequence,
                    execution.outcome,
                    execution.failure_kind,
                    execution.to_json(),
                ),
            )
            return sequence

    def close_attempt(self, attempt_id: str, conclusion: str) -> None:
        require_text("conclusion", conclusion)
        with self._transaction():
            attempt = self._require_attempt_row(attempt_id)
            if attempt[4]:
                raise RegistryError(f"attempt {attempt_id!r} is already closed")
            last = self._one(
                "SELECT sequence, outcome, failure_kind FROM executions "
                "WHERE attempt_id = ? ORDER BY sequence DESC LIMIT 1",
                (attempt_id,),
            )
            if last is None:
                raise RegistryError(f"attempt {attempt_id!r} has no executions")
            cap = self._require_cycle_row(attempt[1])[4]
            if (last[1], last[2]) == ("failed", "code_error") and last[0] < 1 + cap:
                raise RegistryError(
                    "code_error is final only after the fix-retry cap: "
                    f"{last[0]} of {1 + cap} executions used"
                )
            self._db.execute(
                "UPDATE attempts SET closed = 1, conclusion = ? WHERE id = ?",
                (conclusion, attempt_id),
            )

    # -- queries -----------------------------------------------------------

    def list_attempts(
        self, cycle_id: str | None = None, outcome: str | None = None
    ) -> list[AttemptSummary]:
        if outcome is not None and outcome not in OUTCOMES:
            raise ValidationError("outcome", f"must be one of {', '.join(OUTCOMES)}")
        sql = (
            "SELECT a.id, a.cycle_id, a.iteration, a.proposal, a.closed, e.outcome, "
            f"e.failure_kind FROM attempts a {_LAST_EXECUTION} WHERE 1 = 1"
        )
        args: list = []
        if cycle_id is not None:
            sql += " AND a.cycle_id = ?"
            args.append(cycle_id)
        if outcome is not None:
            sql += " AND a.closed = 1 AND e.outcome = ?"
            args.append(outcome)
        sql += " ORDER BY a.iteration, a.rowid"
        out = []
        for row in self._db.execute(sql, args).fetchall():
            proposal = Attempt.from_json(row[3])
            closed = bool(row[4])
            out.append(
                AttemptSummary(
                    attempt_id=row[0],
                    cycle_id=row[1],
                    iteration=row[2],
                    parent_id=proposal.parent_id,
                    phase=proposal.phase,
                    outcome=row[5] if closed else None,
                    failure_kind=row[6] if closed else None,
                )
            )
        return out

    def get_attempt(self, attempt_id: str) -> AttemptDetail:
        row = self._require_attempt_row(attempt_id)
        executions = tuple(self._executions(attempt_id))
        closed = bool(row[4])
        last = executions[-1] if executions and closed else None
        return AttemptDetail(
            attempt=Attempt.from_json(row[6]),
            cycle_id=row[1],
            iteration=row[2],
            fingerprint=row[3],
            closed=closed,
            conclusion=row[5],
            executions=executions,
            outcome=last.outcome if last else None,
            failure_kind=last.failure_kind if last else None,
            reason=last.reason if last else None,
        )

    def metric_series(self, cycle_id: str, partition: str) -> list[MetricPoint]:
        self._require_cycle_row(cycle_id)
        rows = self._db.execute(
            "SELECT id, iteration FROM attempts WHERE cycle_id = ? ORDER BY iteration",
            (cycle_id,),
        ).fetchall()
        points = []
        for attempt_id, iteration in rows:
            for execution in reversed(self._executions(attempt_id)):
                metric = next((m for m in execution.metrics if m.partition == partition), None)
                if metric is not None:
                    points.append(
                        MetricPoint(iteration, attempt_id, metric.mean, metric.dispersion)
                    )
                    break
        return points

    def list_failures(self, failure_kind: str | None = None) -> list[FailureRecord]:
        if failure_kind is not None and failure_kind not in FAILURE_KINDS:
            raise ValidationError(
                "failure_kind", f"must be one of {', '.join(FAILURE_KINDS)}"
            )
        sql = (
            "SELECT a.id, a.proposal, e.record FROM attempts a "
            f"{_LAST_EXECUTION} WHERE a.closed = 1 AND e.outcome != 'succeeded'"
        )
        args: list = []
        if failure_kind is not None:
            sql += " AND e.failure_kind = ?"
            args.append(failure_kind)
        sql += " ORDER BY a.iteration, a.rowid"
        out = []
        for attempt_id, proposal, record in self._db.execute(sql, args).fetchall():
            execution = ExecutionRecord.from_json(record)
            out.append(
                FailureRecord(
                    attempt_id=attempt_id,
                    what_changed=Attempt.from_json(proposal).what_changed,
                    outcome=execution.outcome,
                    failure_kind=execution.failure_kind,
                    reason=execution.reason,
                )
            )
        return out

    # -- suggestions -------------------------------------------------------

    def add_suggestion(self, suggestion: Suggestion) -> None:
        with self._transaction():
            if self._one("SELECT 1 FROM suggestions WHERE id = ?", (suggestion.id,)):
                raise RegistryError(f"suggestion {suggestion.id!r} already exists")
            for cycle_id in (suggestion.origin_cycle, *suggestion.spawned_cycles):
                self._require_cycle_row(cycle_id)
            self._db.execute(
                "INSERT INTO suggestions VALUES (?, ?, ?)",
                (suggestion.id, suggestion.text, suggestion.origin_cycle),
            )
            for position, cycle_id in enumerate(suggestion.spawned_cycles):
                self._db.execute(
                    "INSERT INTO suggestion_cycles VALUES (?, ?, ?)",
                    (suggestion.id, cycle_id, position),
                )

    def link_cycle(self, suggestion_id: str, cycle_id: str) -> None:
        with self._transaction():
            current = self.get_suggestion(suggestion_id)
            self._require_cycle_row(cycle_id)
            # rebuilding the Suggestion applies its rules (no origin, no duplicates)
            Suggestion(
                current.id,
                current.text,
                current.origin_cycle,
                current.spawned_cycles + (cycle_id,),
            )
            self._db.execute(
                "INSERT INTO suggestion_cycles VALUES (?, ?, ?)",
                (suggestion_id, cycle_id, len(current.spawned_cycles)),
            )

    def get_suggestion(self, suggestion_id: str) -> Suggestion:
        row = self._one(
            "SELECT id, text, origin_cycle FROM suggestions WHERE id = ?", (suggestion_id,)
        )
        if row is None:
            raise NotFoundError(f"suggestion {suggestion_id!r} does not exist")
        spawned = self._db.execute(
            "SELECT cycle_id FROM suggestion_cycles WHERE suggestion_id = ? ORDER BY position",
            (suggestion_id,),
        ).fetchall()
        return Suggestion(row[0], row[1], row[2], tuple(r[0] for r in spawned))

    def list_suggestions(self) -> list[Suggestion]:
        rows = self._db.execute("SELECT id FROM suggestions ORDER BY rowid").fetchall()
        return [self.get_suggestion(r[0]) for r in rows]

    def cycles_of_suggestion(self, suggestion_id: str) -> list[Cycle]:
        suggestion = self.get_suggestion(suggestion_id)
        return [self.get_cycle(c) for c in suggestion.spawned_cycles]
