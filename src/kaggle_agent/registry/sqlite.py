"""SQLite implementation of the registry, standard library only (design section 7)."""

import json
import sqlite3
from contextlib import contextmanager

from kaggle_agent.proposals import (
    Attempt,
    Guidance,
    ReviewerReport,
    Round,
    ShortPlan,
    Suggestion,
    ValidationError,
    validate_role,
)
from kaggle_agent.proposals.errors import require_text

from .errors import DuplicateAttemptError, NotFoundError, RegistryError
from .interface import (
    AttemptDetail,
    AttemptSummary,
    FailureRecord,
    MetricPoint,
    Registry,
    RoundDetail,
    RoundMetrics,
)
from .records import (
    FAILURE_KINDS,
    OUTCOMES,
    BundleRecord,
    ComparisonRecord,
    Cycle,
    ExecutionRecord,
    HoldoutRead,
    HoldoutSession,
    LlmCallEntry,
    RoleTrace,
    SubmissionEntry,
    _finite,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id TEXT PRIMARY KEY,
    plan TEXT NOT NULL,
    iterations INTEGER NOT NULL,
    stagnation_k INTEGER NOT NULL,
    max_fix_retries INTEGER NOT NULL,
    state TEXT NOT NULL,
    protocol_hash TEXT
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
CREATE TABLE IF NOT EXISTS holdout_sessions (
    cycle_id TEXT PRIMARY KEY REFERENCES cycles(id),
    opened_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE TABLE IF NOT EXISTS holdout_reads (
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    attempt_id TEXT NOT NULL REFERENCES attempts(id),
    rows INTEGER NOT NULL,
    read_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comparisons (
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    candidate_attempt_id TEXT NOT NULL REFERENCES attempts(id),
    reference_attempt_id TEXT NOT NULL REFERENCES attempts(id),
    difference REAL NOT NULL,
    threshold REAL NOT NULL,
    is_real INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    competition TEXT NOT NULL,
    submission_id TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    message TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    status TEXT NOT NULL,
    public_score REAL,
    completed_at TEXT,
    PRIMARY KEY (competition, submission_id)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    purpose TEXT NOT NULL,
    outcome TEXT NOT NULL,
    started_at TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    attempt INTEGER NOT NULL,
    model TEXT,
    rejection_reason TEXT
);
CREATE TABLE IF NOT EXISTS rounds (
    id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    position INTEGER NOT NULL,
    tasks TEXT NOT NULL,
    UNIQUE (cycle_id, position)
);
CREATE TABLE IF NOT EXISTS bundles (
    id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    folder_hash TEXT NOT NULL,
    reference TEXT NOT NULL,
    schema_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviewer_reports (
    id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES rounds(id),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lead_guidance (
    id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    round_id TEXT NOT NULL REFERENCES rounds(id),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS role_traces (
    cycle_id TEXT NOT NULL REFERENCES cycles(id),
    role TEXT NOT NULL,
    round_id TEXT NOT NULL REFERENCES rounds(id),
    candidate_id TEXT REFERENCES attempts(id),
    input_type TEXT NOT NULL,
    input TEXT NOT NULL,
    output_type TEXT,
    output TEXT,
    outcome TEXT NOT NULL,
    llm_call_id INTEGER,
    created_at TEXT NOT NULL
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
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first schema to files created earlier."""
        columns = [r[1] for r in self._db.execute("PRAGMA table_info(cycles)")]
        if "protocol_hash" not in columns:
            self._db.execute("ALTER TABLE cycles ADD COLUMN protocol_hash TEXT")
        llm_columns = [r[1] for r in self._db.execute("PRAGMA table_info(llm_calls)")]
        if "rejection_reason" not in llm_columns:
            self._db.execute("ALTER TABLE llm_calls ADD COLUMN rejection_reason TEXT")
        # nullable columns without REFERENCES: the code checks the references
        for table, column in (
            ("cycles", "project_id"),
            ("attempts", "round_id"),
            ("llm_calls", "role"),
        ):
            present = [r[1] for r in self._db.execute(f"PRAGMA table_info({table})")]
            if column not in present:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

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
            "SELECT id, plan, iterations, stagnation_k, max_fix_retries, state, "
            "protocol_hash, project_id FROM cycles WHERE id = ?",
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
            protocol_hash=row[6],
            project_id=row[7],
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

    def _require_round_in_cycle(self, round_id: str, cycle_id: str):
        row = self._one("SELECT id, position, tasks FROM rounds WHERE id = ? AND cycle_id = ?",
                        (round_id, cycle_id))
        if row is None:
            raise NotFoundError(f"round {round_id!r} does not exist in cycle {cycle_id!r}")
        return row

    def _point(self, attempt_id: str, iteration: int, partition: str) -> MetricPoint | None:
        for execution in reversed(self._executions(attempt_id)):
            metric = next((m for m in execution.metrics if m.partition == partition), None)
            if metric is not None:
                return MetricPoint(iteration, attempt_id, metric.mean, metric.dispersion)
        return None

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
                "INSERT INTO cycles (id, plan, iterations, stagnation_k, max_fix_retries, "
                "state, protocol_hash, project_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cycle.id,
                    cycle.plan.to_json(),
                    cycle.iterations,
                    cycle.stagnation_k,
                    cycle.max_fix_retries,
                    cycle.state,
                    cycle.protocol_hash,
                    cycle.project_id,
                ),
            )

    def get_cycle(self, cycle_id: str) -> Cycle:
        return self._cycle(self._require_cycle_row(cycle_id))

    def set_cycle_state(self, cycle_id: str, state: str) -> None:
        require_text("state", state)
        with self._transaction():
            self._require_cycle_row(cycle_id)
            self._db.execute("UPDATE cycles SET state = ? WHERE id = ?", (state, cycle_id))

    def set_protocol_hash(self, cycle_id: str, protocol_hash: str) -> None:
        require_text("protocol_hash", protocol_hash)
        with self._transaction():
            current = self._require_cycle_row(cycle_id)[6]
            if current is not None and current != protocol_hash:
                raise RegistryError(f"cycle {cycle_id!r} already has a different protocol hash")
            self._db.execute(
                "UPDATE cycles SET protocol_hash = ? WHERE id = ?", (protocol_hash, cycle_id)
            )

    # -- attempts ----------------------------------------------------------

    def add_attempt(
        self,
        cycle_id: str,
        attempt: Attempt,
        fingerprint: str,
        round_id: str | None = None,
    ) -> int:
        require_text("fingerprint", fingerprint)
        with self._transaction():
            self._require_cycle_row(cycle_id)
            if round_id is not None:
                require_text("round_id", round_id)
                self._require_round_in_cycle(round_id, cycle_id)
            if attempt.bundle_id is not None and not self._one(
                "SELECT 1 FROM bundles WHERE id = ?", (attempt.bundle_id,)
            ):
                raise NotFoundError(f"bundle {attempt.bundle_id!r} does not exist")
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
                "INSERT INTO attempts (id, cycle_id, iteration, fingerprint, proposal, round_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (attempt.id, cycle_id, iteration, fingerprint, attempt.to_json(), round_id),
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
            point = self._point(attempt_id, iteration, partition)
            if point is not None:
                points.append(point)
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
            self._insert_suggestion(suggestion)

    def _insert_suggestion(self, suggestion: Suggestion) -> None:
        """Insert a suggestion inside a transaction the caller already opened."""
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

    # -- holdout sessions and comparisons ------------------------------------

    def open_holdout_session(self, cycle_id: str, opened_at: str) -> None:
        session = HoldoutSession(cycle_id, opened_at)
        with self._transaction():
            self._require_cycle_row(cycle_id)
            if self._one("SELECT 1 FROM holdout_sessions WHERE cycle_id = ?", (cycle_id,)):
                raise RegistryError(f"cycle {cycle_id!r} already had its holdout session")
            self._db.execute(
                "INSERT INTO holdout_sessions (cycle_id, opened_at) VALUES (?, ?)",
                (session.cycle_id, session.opened_at),
            )

    def close_holdout_session(self, cycle_id: str, closed_at: str) -> None:
        require_text("closed_at", closed_at)
        with self._transaction():
            self._require_cycle_row(cycle_id)
            row = self._one(
                "SELECT closed_at FROM holdout_sessions WHERE cycle_id = ?", (cycle_id,)
            )
            if row is None:
                raise RegistryError(f"cycle {cycle_id!r} has no holdout session")
            if row[0] is not None:
                raise RegistryError(f"holdout session of cycle {cycle_id!r} is already closed")
            self._db.execute(
                "UPDATE holdout_sessions SET closed_at = ? WHERE cycle_id = ?",
                (closed_at, cycle_id),
            )

    def get_holdout_session(self, cycle_id: str) -> HoldoutSession | None:
        self._require_cycle_row(cycle_id)
        row = self._one(
            "SELECT cycle_id, opened_at, closed_at FROM holdout_sessions WHERE cycle_id = ?",
            (cycle_id,),
        )
        return None if row is None else HoldoutSession(*row)

    def add_holdout_read(self, read: HoldoutRead) -> None:
        with self._transaction():
            self._require_cycle_row(read.cycle_id)
            self._require_attempt_row(read.attempt_id)
            row = self._one(
                "SELECT closed_at FROM holdout_sessions WHERE cycle_id = ?", (read.cycle_id,)
            )
            if row is None or row[0] is not None:
                raise RegistryError(
                    f"cycle {read.cycle_id!r} has no open holdout session: read refused"
                )
            self._db.execute(
                "INSERT INTO holdout_reads (cycle_id, attempt_id, rows, read_at) "
                "VALUES (?, ?, ?, ?)",
                (read.cycle_id, read.attempt_id, read.rows, read.read_at),
            )

    def list_holdout_reads(self, cycle_id: str) -> list[HoldoutRead]:
        self._require_cycle_row(cycle_id)
        rows = self._db.execute(
            "SELECT cycle_id, attempt_id, rows, read_at FROM holdout_reads "
            "WHERE cycle_id = ? ORDER BY rowid",
            (cycle_id,),
        ).fetchall()
        return [HoldoutRead(*r) for r in rows]

    def add_comparison(self, comparison: ComparisonRecord) -> None:
        with self._transaction():
            self._require_cycle_row(comparison.cycle_id)
            self._require_attempt_row(comparison.candidate_attempt_id)
            self._require_attempt_row(comparison.reference_attempt_id)
            self._db.execute(
                "INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, ?)",
                (
                    comparison.cycle_id,
                    comparison.candidate_attempt_id,
                    comparison.reference_attempt_id,
                    comparison.difference,
                    comparison.threshold,
                    int(comparison.is_real),
                ),
            )

    def list_comparisons(self, cycle_id: str) -> list[ComparisonRecord]:
        self._require_cycle_row(cycle_id)
        rows = self._db.execute(
            "SELECT cycle_id, candidate_attempt_id, reference_attempt_id, difference, "
            "threshold, is_real FROM comparisons WHERE cycle_id = ? ORDER BY rowid",
            (cycle_id,),
        ).fetchall()
        return [ComparisonRecord(*r[:5], bool(r[5])) for r in rows]

    def count_comparisons(self, cycle_id: str) -> int:
        self._require_cycle_row(cycle_id)
        return self._one("SELECT COUNT(*) FROM comparisons WHERE cycle_id = ?", (cycle_id,))[0]

    # -- submissions to Kaggle ------------------------------------------------

    def add_submission(self, entry: SubmissionEntry) -> None:
        with self._transaction():
            self._require_cycle_row(entry.cycle_id)
            if self._one(
                "SELECT 1 FROM submissions WHERE competition = ? AND submission_id = ?",
                (entry.competition, entry.submission_id),
            ):
                raise RegistryError(
                    f"submission {entry.submission_id} of {entry.competition} already recorded"
                )
            self._db.execute(
                "INSERT INTO submissions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.cycle_id,
                    entry.competition,
                    entry.submission_id,
                    entry.file_sha256,
                    entry.message,
                    entry.submitted_at,
                    entry.status,
                    entry.public_score,
                    entry.completed_at,
                ),
            )

    def update_submission_score(
        self,
        competition: str,
        submission_id: str,
        status: str,
        public_score: float | None,
        completed_at: str,
    ) -> None:
        require_text("status", status)
        require_text("completed_at", completed_at)
        if status == "pending":
            raise RegistryError("a pending submission has no result to record")
        if public_score is not None:
            public_score = _finite("public_score", public_score)
        with self._transaction():
            row = self._one(
                "SELECT completed_at FROM submissions WHERE competition = ? AND submission_id = ?",
                (competition, submission_id),
            )
            if row is None:
                raise NotFoundError(f"unknown submission {submission_id} of {competition}")
            if row[0] is not None:
                raise RegistryError(
                    f"score of submission {submission_id} of {competition} already recorded"
                )
            self._db.execute(
                "UPDATE submissions SET status = ?, public_score = ?, completed_at = ? "
                "WHERE competition = ? AND submission_id = ?",
                (status, public_score, completed_at, competition, submission_id),
            )

    def list_submissions(self, cycle_id: str) -> list[SubmissionEntry]:
        self._require_cycle_row(cycle_id)
        rows = self._db.execute(
            "SELECT cycle_id, competition, submission_id, file_sha256, message, "
            "submitted_at, status, public_score, completed_at FROM submissions "
            "WHERE cycle_id = ? ORDER BY rowid",
            (cycle_id,),
        ).fetchall()
        return [SubmissionEntry(*row) for row in rows]

    # -- language model calls -------------------------------------------------

    def add_llm_call(self, entry: LlmCallEntry) -> int:
        with self._transaction():
            self._require_cycle_row(entry.cycle_id)
            cursor = self._db.execute(
                "INSERT INTO llm_calls (cycle_id, purpose, outcome, started_at, "
                "duration_seconds, input_tokens, output_tokens, attempt, model, "
                "rejection_reason, role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.cycle_id,
                    entry.purpose,
                    entry.outcome,
                    entry.started_at,
                    entry.duration_seconds,
                    entry.input_tokens,
                    entry.output_tokens,
                    entry.attempt,
                    entry.model,
                    entry.rejection_reason,
                    entry.role,
                ),
            )
            return cursor.lastrowid

    def list_llm_calls(self, cycle_id: str, role: str | None = None) -> list[LlmCallEntry]:
        if role is not None:
            validate_role(role)
        self._require_cycle_row(cycle_id)
        sql = (
            "SELECT cycle_id, purpose, outcome, started_at, duration_seconds, "
            "input_tokens, output_tokens, attempt, model, rejection_reason, role "
            "FROM llm_calls WHERE cycle_id = ?"
        )
        args: list = [cycle_id]
        if role is not None:
            sql += " AND role = ?"
            args.append(role)
        rows = self._db.execute(sql + " ORDER BY rowid", args).fetchall()
        return [LlmCallEntry(*row) for row in rows]

    # -- rounds ----------------------------------------------------------------

    def _candidate_ids(self, round_id: str) -> tuple[str, ...]:
        rows = self._db.execute(
            "SELECT id FROM attempts WHERE round_id = ? ORDER BY iteration", (round_id,)
        ).fetchall()
        return tuple(r[0] for r in rows)

    def add_round(self, cycle_id: str, round: Round) -> int:
        with self._transaction():
            self._require_cycle_row(cycle_id)
            if self._one("SELECT 1 FROM rounds WHERE id = ?", (round.id,)):
                raise RegistryError(f"round {round.id!r} already exists")
            for candidate_id in round.candidates:
                row = self._one(
                    "SELECT cycle_id, round_id FROM attempts WHERE id = ?", (candidate_id,)
                )
                if row is None or row[0] != cycle_id:
                    raise NotFoundError(
                        f"candidate {candidate_id!r} is not an attempt of cycle {cycle_id!r}"
                    )
                if row[1] is not None:
                    raise RegistryError(f"candidate {candidate_id!r} already belongs to a round")
            position = self._one(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM rounds WHERE cycle_id = ?",
                (cycle_id,),
            )[0]
            self._db.execute(
                "INSERT INTO rounds (id, cycle_id, position, tasks) VALUES (?, ?, ?, ?)",
                (round.id, cycle_id, position, json.dumps(list(round.tasks), ensure_ascii=False)),
            )
            for candidate_id in round.candidates:
                self._db.execute(
                    "UPDATE attempts SET round_id = ? WHERE id = ?", (round.id, candidate_id)
                )
            return position

    def _round(self, round_id: str, tasks: str) -> Round:
        return Round(round_id, tuple(json.loads(tasks)), self._candidate_ids(round_id))

    def get_round(self, round_id: str) -> Round:
        row = self._one("SELECT id, tasks FROM rounds WHERE id = ?", (round_id,))
        if row is None:
            raise NotFoundError(f"round {round_id!r} does not exist")
        return self._round(*row)

    def list_rounds_with_candidates(self, cycle_id: str) -> list[RoundDetail]:
        self._require_cycle_row(cycle_id)
        rows = self._db.execute(
            "SELECT id, position, tasks FROM rounds WHERE cycle_id = ? ORDER BY position",
            (cycle_id,),
        ).fetchall()
        summaries = {s.attempt_id: s for s in self.list_attempts(cycle_id)}
        return [
            RoundDetail(
                round=(round_ := self._round(round_id, tasks)),
                cycle_id=cycle_id,
                position=position,
                candidates=tuple(summaries[c] for c in round_.candidates),
            )
            for round_id, position, tasks in rows
        ]

    def metric_series_by_round(self, cycle_id: str, partition: str) -> list[RoundMetrics]:
        self._require_cycle_row(cycle_id)
        rounds = self._db.execute(
            "SELECT id, position FROM rounds WHERE cycle_id = ? ORDER BY position", (cycle_id,)
        ).fetchall()
        out = []
        for round_id, position in rounds:
            rows = self._db.execute(
                "SELECT id, iteration FROM attempts WHERE round_id = ? ORDER BY iteration",
                (round_id,),
            ).fetchall()
            points = (self._point(attempt_id, it, partition) for attempt_id, it in rows)
            out.append(RoundMetrics(round_id, position, tuple(p for p in points if p)))
        return out

    # -- bundles ---------------------------------------------------------------

    def add_bundle(self, bundle: BundleRecord) -> None:
        with self._transaction():
            self._require_cycle_row(bundle.cycle_id)
            if self._one("SELECT 1 FROM bundles WHERE id = ?", (bundle.id,)):
                raise RegistryError(f"bundle {bundle.id!r} already exists")
            self._db.execute(
                "INSERT INTO bundles (id, cycle_id, folder_hash, reference, schema_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    bundle.id,
                    bundle.cycle_id,
                    bundle.folder_hash,
                    bundle.reference,
                    json.dumps(bundle.schema, ensure_ascii=False, sort_keys=True),
                ),
            )

    @staticmethod
    def _bundle(row) -> BundleRecord:
        return BundleRecord(row[0], row[1], row[2], row[3], json.loads(row[4]))

    def get_bundle(self, bundle_id: str) -> BundleRecord:
        row = self._one(
            "SELECT id, cycle_id, folder_hash, reference, schema_json FROM bundles WHERE id = ?",
            (bundle_id,),
        )
        if row is None:
            raise NotFoundError(f"bundle {bundle_id!r} does not exist")
        return self._bundle(row)

    def list_bundles(self, cycle_id: str) -> list[BundleRecord]:
        self._require_cycle_row(cycle_id)
        rows = self._db.execute(
            "SELECT id, cycle_id, folder_hash, reference, schema_json FROM bundles "
            "WHERE cycle_id = ? ORDER BY rowid",
            (cycle_id,),
        ).fetchall()
        return [self._bundle(r) for r in rows]

    # -- reviewer reports, guidance and role traces ------------------------------

    def add_reviewer_report(self, report: ReviewerReport) -> None:
        with self._transaction():
            if not self._one("SELECT 1 FROM rounds WHERE id = ?", (report.round_id,)):
                raise NotFoundError(f"round {report.round_id!r} does not exist")
            if self._one("SELECT 1 FROM reviewer_reports WHERE id = ?", (report.id,)):
                raise RegistryError(f"reviewer report {report.id!r} already exists")
            self._db.execute(
                "INSERT INTO reviewer_reports (id, round_id, text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (report.id, report.round_id, report.text, report.created_at),
            )

    def list_reviewer_reports(
        self, cycle_id: str, round_id: str | None = None
    ) -> list[ReviewerReport]:
        self._require_cycle_row(cycle_id)
        if round_id is not None:
            self._require_round_in_cycle(round_id, cycle_id)
        sql = (
            "SELECT p.id, p.round_id, p.text, p.created_at FROM reviewer_reports p "
            "JOIN rounds r ON r.id = p.round_id WHERE r.cycle_id = ?"
        )
        args: list = [cycle_id]
        if round_id is not None:
            sql += " AND p.round_id = ?"
            args.append(round_id)
        rows = self._db.execute(sql + " ORDER BY p.rowid", args).fetchall()
        return [ReviewerReport(*r) for r in rows]

    def add_guidance(self, cycle_id: str, guidance: Guidance) -> None:
        with self._transaction():
            self._require_cycle_row(cycle_id)
            self._require_round_in_cycle(guidance.round_id, cycle_id)
            if self._one("SELECT 1 FROM lead_guidance WHERE id = ?", (guidance.id,)):
                raise RegistryError(f"guidance {guidance.id!r} already exists")
            # raises when the suggestion id is taken, before anything is written
            self._insert_suggestion(Suggestion(guidance.id, guidance.text, cycle_id))
            self._db.execute(
                "INSERT INTO lead_guidance (id, cycle_id, round_id, text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (guidance.id, cycle_id, guidance.round_id, guidance.text, guidance.created_at),
            )

    def list_guidance(self, cycle_id: str) -> list[Guidance]:
        self._require_cycle_row(cycle_id)
        rows = self._db.execute(
            "SELECT id, round_id, text, created_at FROM lead_guidance "
            "WHERE cycle_id = ? ORDER BY rowid",
            (cycle_id,),
        ).fetchall()
        return [Guidance(*r) for r in rows]

    def add_role_trace(self, trace: RoleTrace) -> None:
        with self._transaction():
            self._require_cycle_row(trace.cycle_id)
            self._require_round_in_cycle(trace.round_id, trace.cycle_id)
            if trace.candidate_id is not None:
                row = self._one("SELECT cycle_id FROM attempts WHERE id = ?", (trace.candidate_id,))
                if row is None:
                    raise NotFoundError(f"candidate {trace.candidate_id!r} does not exist")
                if row[0] != trace.cycle_id:
                    raise RegistryError(
                        f"candidate {trace.candidate_id!r} is not an attempt of cycle "
                        f"{trace.cycle_id!r}"
                    )
            if trace.llm_call_id is not None:
                row = self._one("SELECT cycle_id FROM llm_calls WHERE rowid = ?", (trace.llm_call_id,))
                if row is None:
                    raise NotFoundError(f"llm call {trace.llm_call_id} does not exist")
                if row[0] != trace.cycle_id:
                    raise RegistryError(
                        f"llm call {trace.llm_call_id} belongs to another cycle"
                    )
            dump = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)  # noqa: E731
            self._db.execute(
                "INSERT INTO role_traces (cycle_id, role, round_id, candidate_id, input_type, "
                "input, output_type, output, outcome, llm_call_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.cycle_id,
                    trace.role,
                    trace.round_id,
                    trace.candidate_id,
                    trace.input_type,
                    dump(trace.input),
                    trace.output_type,
                    None if trace.output is None else dump(trace.output),
                    trace.outcome,
                    trace.llm_call_id,
                    trace.created_at,
                ),
            )

    def list_role_traces(
        self, cycle_id: str, role: str | None = None, round_id: str | None = None
    ) -> list[RoleTrace]:
        if role is not None:
            validate_role(role)
        self._require_cycle_row(cycle_id)
        sql = (
            "SELECT cycle_id, role, round_id, candidate_id, input_type, input, output_type, "
            "output, outcome, llm_call_id, created_at FROM role_traces WHERE cycle_id = ?"
        )
        args: list = [cycle_id]
        for column, value in (("role", role), ("round_id", round_id)):
            if value is not None:
                sql += f" AND {column} = ?"
                args.append(value)
        out = []
        for r in self._db.execute(sql + " ORDER BY rowid", args).fetchall():
            out.append(
                RoleTrace(
                    cycle_id=r[0], role=r[1], round_id=r[2], candidate_id=r[3],
                    input_type=r[4], input=json.loads(r[5]), output_type=r[6],
                    output=None if r[7] is None else json.loads(r[7]),
                    outcome=r[8], llm_call_id=r[9], created_at=r[10],
                )
            )
        return out
