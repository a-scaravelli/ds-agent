"""SQLite registry: traceability, deduplication, fix retries, queries (no network)."""

import inspect

import pytest
from registry_helpers import (
    close_after_cap,
    crash,
    make_attempt,
    make_cycle,
    make_execution,
    new_attempt,
)

from kaggle_agent.proposals import Suggestion, ValidationError
from kaggle_agent.registry import (
    DuplicateAttemptError,
    NotFoundError,
    PartitionMetric,
    Registry,
    RegistryError,
    SqliteRegistry,
    compute_fingerprint,
)


@pytest.fixture
def reg():
    with SqliteRegistry() as registry:
        registry.add_cycle(make_cycle("C1"))
        yield registry


def _counts(reg):
    attempts = reg.list_attempts()
    return (
        [(s.attempt_id, s.iteration) for s in attempts],
        [len(reg.get_attempt(s.attempt_id).executions) for s in attempts],
        [reg.get_attempt(s.attempt_id).closed for s in attempts],
    )


def test_all_section_7_fields_saved_and_reread(reg):
    attempt = make_attempt(
        "A1", evidence=("e1",), exploratory=False, parent_id=None, artifacts=("diff",),
        expected_effect="+1", estimated_cost="low", risks=("overfit",),
    )
    reg.add_attempt("C1", attempt, "fp1")
    execution = make_execution(
        "A1", config={"lr": 0.1, "layers": [1, 2]}, seed=42, resources={"cpu": 2, "mem_mb": 512},
        metrics=[PartitionMetric("validation", 0.8, 0.02), PartitionMetric("train", 0.9, 0.01)],
        reason="improved", artifacts=["preds.csv"],
    )
    reg.add_execution(execution)
    reg.close_attempt("A1", "learned that lr 0.1 works")
    detail = reg.get_attempt("A1")
    assert detail.attempt == attempt
    saved = detail.executions[0]
    for name in execution.to_dict():
        assert getattr(saved, name) == getattr(execution, name), name
    assert (detail.conclusion, detail.iteration, detail.closed) == ("learned that lr 0.1 works", 1, True)
    assert detail.outcome == "succeeded" and detail.fingerprint == "fp1"


def test_traceability_survives_restart(tmp_path):
    path = tmp_path / "registry.db"
    reg = SqliteRegistry(path)
    reg.add_cycle(make_cycle("C1"))
    new_attempt(reg, "A1")
    reg.add_execution(make_execution("A1"))
    reg.close_attempt("A1", "ok")
    new_attempt(reg, "A2")
    close_after_cap(reg, "A2")
    new_attempt(reg, "A3")
    reg.add_execution(make_execution("A3", outcome="failed", failure_kind="timeout", reason="slow"))
    reg.close_attempt("A3", "too slow")
    reg.add_suggestion(Suggestion("S1", "try more", "C1"))
    reg.add_cycle(make_cycle("C2"))
    reg.link_cycle("S1", "C2")

    def snapshot(r):
        return (
            r.list_attempts(), r.get_attempt("A2"), r.metric_series("C1", "validation"),
            r.list_failures(), r.list_suggestions(), r.get_cycle("C2"),
        )

    before = snapshot(reg)
    reg.close()
    with SqliteRegistry(path) as again:
        assert snapshot(again) == before
        assert len(before[3]) == 2


def test_duplicate_rejected_pointing_to_existing(reg):
    new_attempt(reg, "A1", fingerprint="same")
    before = _counts(reg)
    with pytest.raises(DuplicateAttemptError) as exc:
        reg.add_attempt("C1", make_attempt("A2"), "same")
    assert exc.value.existing_attempt_id == "A1" and exc.value.fingerprint == "same"
    assert _counts(reg) == before
    with pytest.raises(NotFoundError):
        reg.get_attempt("A2")
    assert reg.add_attempt("C1", make_attempt("A2"), "other") == 2


def test_duplicate_checked_against_all_outcomes_and_cycles(reg):
    reg.add_cycle(make_cycle("B"))
    new_attempt(reg, "F", fingerprint="fp-f")
    reg.add_execution(make_execution("F", outcome="failed", failure_kind="hypothesis_falsified", reason="no"))
    reg.close_attempt("F", "falsified")
    new_attempt(reg, "S", "B", fingerprint="fp-s")
    reg.add_execution(make_execution("S"))
    reg.close_attempt("S", "ok")
    for cycle in ("C1", "B"):
        for fingerprint, existing in (("fp-f", "F"), ("fp-s", "S")):
            with pytest.raises(DuplicateAttemptError) as exc:
                reg.add_attempt(cycle, make_attempt(f"N-{cycle}-{existing}"), fingerprint)
            assert exc.value.existing_attempt_id == existing


def test_same_config_different_code_is_accepted(reg):
    config = {"lr": 0.1}
    f1 = compute_fingerprint(config, "code1", "d", "p")
    f2 = compute_fingerprint(config, "code2", "d", "p")
    assert reg.add_attempt("C1", make_attempt("A1"), f1) == 1
    assert reg.add_attempt("C1", make_attempt("A2"), f2) == 2


def test_fix_retries_are_executions_not_attempts(reg):
    fingerprint = compute_fingerprint({"lr": 0.1}, "code1", "d", "p")
    reg.add_attempt("C1", make_attempt("A1"), fingerprint)
    assert reg.add_execution(crash("A1", 1)) == 1
    assert reg.add_execution(crash("A1", 2)) == 2
    assert reg.add_execution(make_execution("A1", code_hash="fixed")) == 3
    reg.close_attempt("A1", "fixed after two crashes")
    detail = reg.get_attempt("A1")
    assert [e.id for e in detail.executions] == ["A1-crash1", "A1-crash2", "A1-E"]
    assert detail.outcome == "succeeded" and detail.failure_kind is None
    assert len(reg.list_attempts()) == 1
    assert reg.add_attempt("C1", make_attempt("A2"), "other") == 2


def test_fix_retry_cap_is_three_and_configurable(reg):
    reg.add_cycle(make_cycle("C2", max_fix_retries=1))
    new_attempt(reg, "D", "C1")
    new_attempt(reg, "E", "C2")
    for n in range(4):
        reg.add_execution(crash("D", n))
    with pytest.raises(RegistryError):
        reg.add_execution(crash("D", 4))
    reg.add_execution(crash("E", 0))
    reg.add_execution(crash("E", 1))
    with pytest.raises(RegistryError):
        reg.add_execution(crash("E", 2))
    assert len(reg.get_attempt("D").executions) == 4
    assert len(reg.get_attempt("E").executions) == 2
    for bad in (-1, 1.5, "3"):
        with pytest.raises(ValidationError):
            make_cycle("C3", max_fix_retries=bad)


def test_cap_exceeded_closes_failed_with_code_error(reg):
    new_attempt(reg, "A1")
    for n in range(4):
        reg.add_execution(crash("A1", n))
    reg.close_attempt("A1", "still failing")
    detail = reg.get_attempt("A1")
    assert (detail.outcome, detail.failure_kind) == ("failed", "code_error")
    assert detail.reason == "Traceback: boom"
    with pytest.raises(RegistryError):
        reg.add_execution(crash("A1", 9))
    assert [f.attempt_id for f in reg.list_failures(failure_kind="code_error")] == ["A1"]


def test_close_before_cap_refused_for_code_error(reg):
    new_attempt(reg, "A1")
    for n in range(1, 4):
        reg.add_execution(crash("A1", n))
        with pytest.raises(RegistryError):
            reg.close_attempt("A1", "too early")
        detail = reg.get_attempt("A1")
        assert detail.closed is False and detail.outcome is None
        assert reg.list_failures() == [] and reg.list_failures("code_error") == []
    reg.add_execution(crash("A1", 4))
    reg.close_attempt("A1", "now final")
    assert reg.get_attempt("A1").closed


def test_attempt_without_executions_cannot_be_closed(reg):
    new_attempt(reg, "A1")
    with pytest.raises(RegistryError):
        reg.close_attempt("A1", "nothing ran")
    assert reg.get_attempt("A1").closed is False
    reg.add_execution(make_execution("A1"))
    reg.close_attempt("A1", "ok")
    with pytest.raises(RegistryError):
        reg.close_attempt("A1", "again")
    with pytest.raises(RegistryError):
        reg.add_execution(make_execution("A1", exec_id="late"))


@pytest.mark.parametrize(
    "outcome,kind",
    [
        ("succeeded", None),
        ("failed", "timeout"),
        ("failed", "hypothesis_falsified"),
        ("inconclusive", "below_threshold"),
    ],
)
def test_fix_retry_only_after_code_crash(reg, outcome, kind):
    new_attempt(reg, "A1")
    first = make_execution("A1", "first", outcome, failure_kind=kind) if kind else make_execution("A1", "first")
    reg.add_execution(first)
    with pytest.raises(RegistryError):
        reg.add_execution(make_execution("A1", "second"))
    assert len(reg.get_attempt("A1").executions) == 1


def test_fix_retry_accepted_after_code_error(reg):
    new_attempt(reg, "A1")
    reg.add_execution(crash("A1", 1))
    assert reg.add_execution(make_execution("A1", "second")) == 2


def test_open_attempt_has_no_outcome_and_is_excluded(reg):
    new_attempt(reg, "A1")
    reg.add_execution(crash("A1"))
    assert reg.list_failures() == []
    assert reg.list_attempts(outcome="failed") == []
    detail = reg.get_attempt("A1")
    assert detail.closed is False and detail.outcome is None and detail.failure_kind is None
    assert reg.list_attempts()[0].outcome is None


def test_failures_by_failure_kind(reg):
    new_attempt(reg, "CE")
    close_after_cap(reg, "CE")
    for name, kind in (("HF", "hypothesis_falsified"), ("TO", "timeout"), ("BT", "below_threshold")):
        new_attempt(reg, name)
        reg.add_execution(make_execution(name, outcome="failed", failure_kind=kind, reason=f"r-{kind}"))
        reg.close_attempt(name, "done")
    new_attempt(reg, "OK")
    reg.add_execution(make_execution("OK"))
    reg.close_attempt("OK", "done")
    everything = reg.list_failures()
    assert [f.attempt_id for f in everything] == ["CE", "HF", "TO", "BT"]
    for kind, expected in (
        ("code_error", "CE"), ("hypothesis_falsified", "HF"), ("timeout", "TO"), ("below_threshold", "BT"),
    ):
        (item,) = reg.list_failures(kind)
        assert item.attempt_id == expected and item.failure_kind == kind
        assert item.outcome == "failed" and item.what_changed == f"change {expected}"
    assert reg.list_failures("timeout")[0].reason == "r-timeout"
    with pytest.raises(ValidationError):
        reg.list_failures("oops")


def test_list_attempts_filters_and_order(reg):
    reg.add_cycle(make_cycle("B"))
    new_attempt(reg, "A1")
    new_attempt(reg, "A2", parent_id="A1", phase="baseline")
    new_attempt(reg, "A3")
    new_attempt(reg, "B1", "B")
    reg.add_execution(make_execution("A1"))
    reg.close_attempt("A1", "c")
    reg.add_execution(make_execution("A2", outcome="inconclusive", failure_kind="below_threshold"))
    reg.close_attempt("A2", "c")
    reg.add_execution(make_execution("B1"))
    reg.close_attempt("B1", "c")
    assert [s.attempt_id for s in reg.list_attempts()] == ["A1", "B1", "A2", "A3"]
    assert [s.attempt_id for s in reg.list_attempts(cycle_id="C1")] == ["A1", "A2", "A3"]
    assert [s.attempt_id for s in reg.list_attempts(outcome="succeeded")] == ["A1", "B1"]
    (second,) = reg.list_attempts(outcome="inconclusive")
    assert (second.cycle_id, second.iteration, second.parent_id, second.phase) == ("C1", 2, "A1", "baseline")
    assert (second.outcome, second.failure_kind) == ("inconclusive", "below_threshold")
    with pytest.raises(ValidationError):
        reg.list_attempts(outcome="done")


def test_metric_series_per_iteration(reg):
    def metric(mean, partition="validation"):
        return [PartitionMetric(partition, mean, mean / 10)]

    new_attempt(reg, "base", phase="baseline")
    reg.add_execution(make_execution("base", metrics=metric(0.7)))
    new_attempt(reg, "A1")
    reg.add_execution(make_execution("A1", metrics=metric(0.75)))
    new_attempt(reg, "A2")
    reg.add_execution(make_execution("A2", metrics=metric(0.5, "holdout")))
    new_attempt(reg, "A3")
    reg.add_execution(crash("A3", 1))
    reg.add_execution(make_execution("A3", metrics=metric(0.8)))
    series = reg.metric_series("C1", "validation")
    assert [(p.iteration, p.attempt_id, p.mean) for p in series] == [
        (1, "base", 0.7), (2, "A1", 0.75), (4, "A3", 0.8),
    ]
    assert series[0].dispersion == pytest.approx(0.07)
    with pytest.raises(NotFoundError):
        reg.metric_series("nope", "validation")


def test_attempt_detail_and_unknown_id(reg):
    new_attempt(reg, "P")
    new_attempt(reg, "A1", parent_id="P")
    reg.add_execution(crash("A1", 1))
    reg.add_execution(make_execution("A1", outcome="failed", failure_kind="timeout", reason="slow"))
    reg.close_attempt("A1", "timeouts on this data")
    detail = reg.get_attempt("A1")
    assert detail.attempt.parent_id == "P" and len(detail.executions) == 2
    assert (detail.outcome, detail.failure_kind, detail.reason) == ("failed", "timeout", "slow")
    assert detail.conclusion == "timeouts on this data"
    with pytest.raises(NotFoundError):
        reg.get_attempt("unknown")


def test_attempt_referential_integrity(reg):
    new_attempt(reg, "A1")
    before = _counts(reg)
    with pytest.raises(NotFoundError):
        reg.add_attempt("nope", make_attempt("A2"), "fp2")
    with pytest.raises(NotFoundError):
        reg.add_attempt("C1", make_attempt("A2", parent_id="ghost"), "fp2")
    with pytest.raises(RegistryError):
        reg.add_attempt("C1", make_attempt("A1"), "fp-new")
    assert _counts(reg) == before
    assert reg.add_attempt("C1", make_attempt("A2"), "fp2") == 2


def test_cycle_saved_reread_and_state_updated(reg):
    cycle = make_cycle("C9", iterations=10, stagnation_k=3, max_fix_retries=3, state="ITERATING")
    reg.add_cycle(cycle)
    assert reg.get_cycle("C9") == cycle
    reg.set_cycle_state("C9", "REPORTING")
    assert reg.get_cycle("C9").state == "REPORTING"
    with pytest.raises(RegistryError):
        reg.add_cycle(cycle)
    with pytest.raises(NotFoundError):
        reg.get_cycle("ghost")
    with pytest.raises(NotFoundError):
        reg.set_cycle_state("ghost", "X")
    for kwargs in ({"iterations": 0}, {"stagnation_k": 0}):
        with pytest.raises(ValidationError):
            make_cycle("C10", **kwargs)


def test_suggestion_saved_with_spawned_cycles(reg):
    suggestion = Suggestion("S1", "try target encoding", "C1")
    reg.add_suggestion(suggestion)
    assert reg.get_suggestion("S1") == suggestion
    reg.add_cycle(make_cycle("B"))
    reg.link_cycle("S1", "B")
    assert reg.get_suggestion("S1") == Suggestion("S1", "try target encoding", "C1", ("B",))
    assert [c.id for c in reg.cycles_of_suggestion("S1")] == ["B"]


def test_suggestion_queries_and_integrity(reg):
    reg.add_cycle(make_cycle("B"))
    reg.add_cycle(make_cycle("D"))
    reg.add_suggestion(Suggestion("S1", "one", "C1"))
    reg.add_suggestion(Suggestion("S2", "two", "C1"))
    reg.link_cycle("S1", "D")
    reg.link_cycle("S1", "B")
    assert [s.id for s in reg.list_suggestions()] == ["S1", "S2"]
    assert reg.get_suggestion("S1").spawned_cycles == ("D", "B")
    assert [c.id for c in reg.cycles_of_suggestion("S1")] == ["D", "B"]
    assert reg.cycles_of_suggestion("S2") == []
    with pytest.raises(NotFoundError):
        reg.add_suggestion(Suggestion("S3", "x", "ghost"))
    with pytest.raises(NotFoundError):
        reg.add_suggestion(Suggestion("S3", "x", "C1", ("ghost",)))
    with pytest.raises(NotFoundError):
        reg.link_cycle("S1", "ghost")
    with pytest.raises(NotFoundError):
        reg.link_cycle("ghost", "B")
    with pytest.raises(NotFoundError):
        reg.cycles_of_suggestion("ghost")
    with pytest.raises(ValidationError):
        reg.link_cycle("S1", "C1")
    with pytest.raises(ValidationError):
        reg.link_cycle("S1", "B")
    assert reg.get_suggestion("S1").spawned_cycles == ("D", "B")
    with pytest.raises(RegistryError):
        reg.add_suggestion(Suggestion("S1", "again", "C1"))


def test_storage_is_a_replaceable_interface():
    with pytest.raises(TypeError):
        Registry()
    assert issubclass(SqliteRegistry, Registry)
    assert not inspect.isabstract(SqliteRegistry)
    assert Registry.__abstractmethods__

    stub = type("Stub", (Registry,), {name: (lambda self, *a, **k: None) for name in Registry.__abstractmethods__})
    assert isinstance(stub(), Registry)


def test_rejected_operations_leave_registry_unchanged(reg):
    new_attempt(reg, "A1")
    reg.add_execution(crash("A1", 1))
    new_attempt(reg, "A2")
    for n in range(4):
        reg.add_execution(crash("A2", n))
    before = _counts(reg)
    with pytest.raises(DuplicateAttemptError):
        reg.add_attempt("C1", make_attempt("A3"), "fp-A1")
    with pytest.raises(RegistryError):
        reg.add_execution(crash("A2", 9))
    with pytest.raises(RegistryError):
        reg.add_execution(crash("A1", 1))  # reused execution id
    with pytest.raises(RegistryError):
        reg.close_attempt("A1", "too early")
    with pytest.raises(ValidationError):
        reg.close_attempt("A2", "")
    assert _counts(reg) == before
    assert reg.add_attempt("C1", make_attempt("A3"), "fp-A3") == 3


def test_protocol_hash_written_once_and_persisted(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        assert r.get_cycle("C1").protocol_hash is None
        r.set_protocol_hash("C1", "h1")
        r.set_protocol_hash("C1", "h1")
        with pytest.raises(RegistryError):
            r.set_protocol_hash("C1", "h2")
        with pytest.raises(NotFoundError):
            r.set_protocol_hash("nope", "h")
    with SqliteRegistry(path) as r:
        assert r.get_cycle("C1").protocol_hash == "h1"


def test_holdout_session_rules_and_log(reg):
    from kaggle_agent.registry import HoldoutRead

    new_attempt(reg, "A1")
    assert reg.get_holdout_session("C1") is None
    with pytest.raises(RegistryError):
        reg.add_holdout_read(HoldoutRead("C1", "A1", 5, "t"))
    with pytest.raises(RegistryError):
        reg.close_holdout_session("C1", "t")
    reg.open_holdout_session("C1", "t0")
    with pytest.raises(RegistryError):
        reg.open_holdout_session("C1", "t0b")
    reg.add_holdout_read(HoldoutRead("C1", "A1", 5, "t1"))
    with pytest.raises(NotFoundError):
        reg.add_holdout_read(HoldoutRead("C1", "ghost", 5, "t1"))
    reg.close_holdout_session("C1", "t2")
    assert not reg.get_holdout_session("C1").is_open
    with pytest.raises(RegistryError):
        reg.close_holdout_session("C1", "t3")
    with pytest.raises(RegistryError):
        reg.open_holdout_session("C1", "t4")
    with pytest.raises(RegistryError):
        reg.add_holdout_read(HoldoutRead("C1", "A1", 5, "t5"))
    assert [(r.attempt_id, r.rows) for r in reg.list_holdout_reads("C1")] == [("A1", 5)]


def test_comparisons_stored_and_counted(reg):
    from kaggle_agent.registry import ComparisonRecord

    new_attempt(reg, "A1")
    new_attempt(reg, "A2")
    assert reg.count_comparisons("C1") == 0
    reg.add_comparison(ComparisonRecord("C1", "A2", "A1", 0.5, 0.2, True))
    reg.add_comparison(ComparisonRecord("C1", "A1", "A2", -0.5, 0.2, False))
    with pytest.raises(NotFoundError):
        reg.add_comparison(ComparisonRecord("C1", "A9", "A1", 0.5, 0.2, True))
    assert reg.count_comparisons("C1") == 2
    assert [c.is_real for c in reg.list_comparisons("C1")] == [True, False]


def test_ticket_2_sqlite_file_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE cycles (id TEXT PRIMARY KEY, plan TEXT NOT NULL, iterations INTEGER NOT NULL,"
        " stagnation_k INTEGER NOT NULL, max_fix_retries INTEGER NOT NULL, state TEXT NOT NULL);"
    )
    cycle = make_cycle("OLD")
    db.execute(
        "INSERT INTO cycles VALUES (?, ?, ?, ?, ?, ?)",
        (cycle.id, cycle.plan.to_json(), 10, 3, 3, "ITERATING"),
    )
    db.commit()
    db.close()
    with SqliteRegistry(path) as r:
        assert r.get_cycle("OLD") == cycle
        assert r.get_cycle("OLD").protocol_hash is None
        r.set_protocol_hash("OLD", "h")
    with SqliteRegistry(path) as r:  # the migration is idempotent
        assert r.get_cycle("OLD").protocol_hash == "h"


def _entry(**kwargs):
    from kaggle_agent.registry import SubmissionEntry

    data = {
        "cycle_id": "C1", "competition": "comp", "submission_id": "s1",
        "file_sha256": "f" * 64, "message": "msg",
        "submitted_at": "2026-03-10T12:00:00+00:00", "status": "pending",
    }
    return SubmissionEntry(**{**data, **kwargs})


def test_submission_entry_store_update_once_and_list(reg):
    reg.add_submission(_entry())
    reg.add_submission(_entry(submission_id="s2"))
    assert [e.submission_id for e in reg.list_submissions("C1")] == ["s1", "s2"]
    reg.update_submission_score("comp", "s1", "complete", 0.83, "2026-03-10T12:05:00+00:00")
    first = reg.list_submissions("C1")[0]
    assert (first.status, first.public_score, first.completed_at) == (
        "complete", 0.83, "2026-03-10T12:05:00+00:00",
    )
    assert first.file_sha256 == "f" * 64 and first.message == "msg"
    with pytest.raises(RegistryError):
        reg.update_submission_score("comp", "s1", "complete", 0.9, "later")
    assert reg.list_submissions("C1")[0].public_score == 0.83
    reg.update_submission_score("comp", "s2", "error", None, "t")
    assert reg.list_submissions("C1")[1].public_score is None


def test_submission_entry_refusals(reg):
    with pytest.raises(NotFoundError):
        reg.add_submission(_entry(cycle_id="nope"))
    reg.add_submission(_entry())
    with pytest.raises(RegistryError):
        reg.add_submission(_entry())  # same competition and submission id
    with pytest.raises(NotFoundError):
        reg.update_submission_score("comp", "missing", "complete", 0.5, "t")
    with pytest.raises(RegistryError):
        reg.update_submission_score("comp", "s1", "pending", None, "t")
    with pytest.raises(NotFoundError):
        reg.list_submissions("nope")
    assert reg.list_submissions("C1")[0].completed_at is None


def test_submission_entries_persist_and_old_files_gain_the_table(tmp_path):
    import sqlite3

    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        r.add_submission(_entry())
        r.update_submission_score("comp", "s1", "complete", 0.7, "t")
    with SqliteRegistry(path) as r:
        assert r.list_submissions("C1")[0].public_score == 0.7
    db = sqlite3.connect(path)
    db.execute("DROP TABLE submissions")  # a file written before this table existed
    db.commit()
    db.close()
    with SqliteRegistry(path) as r:
        assert r.get_cycle("C1").id == "C1" and r.list_submissions("C1") == []
        r.add_submission(_entry())


# -- llm_calls: additive table for the consumption of the language model (case 25) --

def _call(**kwargs):
    from kaggle_agent.registry import LlmCallEntry

    data = {
        "cycle_id": "C1", "purpose": "plan", "outcome": "ok", "started_at": "2026-03-10T12:00:00+00:00",
        "duration_seconds": 2.5, "input_tokens": 10, "output_tokens": 5, "attempt": 1, "model": "m",
        **kwargs,
    }
    return LlmCallEntry(**data)


def test_llm_calls_round_trip_in_insertion_order_including_model(reg):
    first, second = _call(), _call(outcome="rate_limited", input_tokens=None, output_tokens=None,
                                   attempt=2, model=None)
    reg.add_llm_call(first)
    reg.add_llm_call(second)
    assert reg.list_llm_calls("C1") == [first, second]


def test_llm_calls_refusals_leave_the_registry_unchanged(reg):
    with pytest.raises(NotFoundError):
        reg.add_llm_call(_call(cycle_id="nope"))
    with pytest.raises(NotFoundError):
        reg.list_llm_calls("nope")
    for bad in ({"outcome": "weird"}, {"input_tokens": -1}, {"output_tokens": -3},
                {"duration_seconds": -1.0}, {"attempt": 0}, {"purpose": ""}, {"model": ""}):
        with pytest.raises(ValidationError):
            _call(**bad)
    assert reg.list_llm_calls("C1") == []


def test_llm_calls_persist_and_old_files_gain_the_table(tmp_path):
    import sqlite3

    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        r.add_llm_call(_call())
    with SqliteRegistry(path) as r:
        assert r.list_llm_calls("C1") == [_call()]
    db = sqlite3.connect(path)
    db.execute("DROP TABLE llm_calls")  # a file written before this table existed
    db.commit()
    db.close()
    with SqliteRegistry(path) as r:
        assert r.get_cycle("C1").id == "C1" and r.list_llm_calls("C1") == []
        r.add_llm_call(_call())
        assert len(r.list_llm_calls("C1")) == 1


def test_llm_calls_without_rejection_reason_column_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        r.add_llm_call(_call())
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE llm_calls DROP COLUMN rejection_reason")  # schema of the previous round
    db.commit()
    db.close()
    with SqliteRegistry(path) as r:
        assert [c.rejection_reason for c in r.list_llm_calls("C1")] == [None]
        r.add_llm_call(_call(outcome="invalid_output", rejection_reason="models: must not be empty"))
        assert [c.rejection_reason for c in r.list_llm_calls("C1")] == [None, "models: must not be empty"]
    with SqliteRegistry(path) as r:  # reopening a migrated file is a no-op
        assert len(r.list_llm_calls("C1")) == 2


# -- three-level design: rounds, bundles, reports, guidance, traces ---------------------------

import sqlite3  # noqa: E402

from registry_helpers import T0, add_round, crash, make_trace  # noqa: E402

from kaggle_agent.proposals import Guidance, ReviewerReport, Round  # noqa: E402
from kaggle_agent.registry import BundleRecord, Cycle, PartitionMetric  # noqa: E402

NEW_TABLES = ("rounds", "bundles", "reviewer_reports", "lead_guidance", "role_traces")


def _rows(path, table):
    db = sqlite3.connect(path)
    try:
        return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        db.close()


def _columns(path, table):
    db = sqlite3.connect(path)
    try:
        return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
    finally:
        db.close()


def _tables(path):
    db = sqlite3.connect(path)
    try:
        return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        db.close()


def _write_old_file(path):
    """A file with the schema of tickets 1-5: no new columns, no new tables."""
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("OLD"))
        new_attempt(r, "A1", "OLD")
        r.add_llm_call(_call(cycle_id="OLD"))
    db = sqlite3.connect(path)
    for table in NEW_TABLES:
        db.execute(f"DROP TABLE {table}")
    db.execute("ALTER TABLE cycles DROP COLUMN project_id")
    db.execute("ALTER TABLE attempts DROP COLUMN round_id")
    db.execute("ALTER TABLE llm_calls DROP COLUMN role")
    db.commit()
    db.close()


def test_old_files_open_and_gain_tables_and_columns(tmp_path):
    path = tmp_path / "old.db"
    _write_old_file(path)
    assert not set(NEW_TABLES) & _tables(path)
    for _ in range(2):  # the second opening is a no-op
        with SqliteRegistry(path) as r:
            assert r.get_cycle("OLD").project_id is None
            assert r.list_llm_calls("OLD")[0].role is None
            assert r.list_rounds_with_candidates("OLD") == []
            assert r.get_attempt("A1").attempt.bundle_id is None
    assert set(NEW_TABLES) <= _tables(path)
    assert "project_id" in _columns(path, "cycles")
    assert "round_id" in _columns(path, "attempts")
    assert "role" in _columns(path, "llm_calls")
    with SqliteRegistry(path) as r:
        add_round(r, "R1", "OLD", candidates=["A1"])
        assert r.list_rounds_with_candidates("OLD")[0].candidates[0].attempt_id == "A1"


def test_project_id_and_awaiting_guidance_state_persist(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1", project_id="P1"))
        r.add_cycle(make_cycle("C2"))
        r.set_cycle_state("C2", "AWAITING_GUIDANCE")
    with SqliteRegistry(path) as r:
        assert r.get_cycle("C1").project_id == "P1"
        assert r.get_cycle("C2").project_id is None
        assert r.get_cycle("C2").state == "AWAITING_GUIDANCE"
        assert Cycle.from_json(r.get_cycle("C1").to_json()) == r.get_cycle("C1")


def test_rounds_list_their_own_candidates(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        add_round(r, "R1", tasks=("profile", "baseline"))
        add_round(r, "R2", tasks=("trees",))
        r.add_attempt("C1", make_attempt("A1"), "f1", round_id="R1")
        r.add_attempt("C1", make_attempt("A2"), "f2", round_id="R2")
        r.add_attempt("C1", make_attempt("A3"), "f3", round_id="R1")
        r.add_attempt("C1", make_attempt("A4"), "f4")
        before = _counts(r)
        with pytest.raises(NotFoundError):
            r.add_attempt("C1", make_attempt("A5"), "f5", round_id="ghost")
        assert _counts(r) == before
    with SqliteRegistry(path) as r:
        details = r.list_rounds_with_candidates("C1")
        assert [d.round.id for d in details] == ["R1", "R2"]
        assert [d.position for d in details] == [1, 2]
        assert [[c.attempt_id for c in d.candidates] for d in details] == [["A1", "A3"], ["A2"]]
        assert details[0].round.tasks == ("profile", "baseline")
        assert details[0].round.candidates == ("A1", "A3")
        assert r.get_round("R2") == Round("R2", ("trees",), ("A2",))
        with pytest.raises(NotFoundError):
            r.get_round("ghost")
        with pytest.raises(NotFoundError):
            r.list_rounds_with_candidates("ghost")


def test_add_round_rules(reg):
    new_attempt(reg, "A1")
    reg.add_cycle(make_cycle("C2"))
    new_attempt(reg, "B1", "C2")
    assert add_round(reg, "R1", candidates=["A1"]) == 1
    for round_ in (Round("R1", ("t",)), Round("R9", ("t",), ("ghost",)), Round("R9", ("t",), ("B1",)),
                   Round("R9", ("t",), ("A1",))):
        with pytest.raises((NotFoundError, RegistryError)):
            reg.add_round("C1", round_)
    with pytest.raises(NotFoundError):
        reg.add_round("ghost", Round("R9", ("t",)))
    assert [d.round.id for d in reg.list_rounds_with_candidates("C1")] == ["R1"]
    with pytest.raises(NotFoundError):  # a round of another cycle is not usable
        reg.add_attempt("C2", make_attempt("B2"), "fb2", round_id="R1")


def test_bundles_and_attempt_reference(tmp_path):
    path = tmp_path / "r.db"
    bundle = BundleRecord("B1", "C1", "hash1", "bundles/B1", {"inputs": ["a"], "n": 2})
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        with pytest.raises(NotFoundError):
            r.add_attempt("C1", make_attempt("A0", bundle_id="B1"), "f0")
        r.add_bundle(bundle)
        with pytest.raises(RegistryError):
            r.add_bundle(bundle)
        with pytest.raises(NotFoundError):
            r.add_bundle(BundleRecord("B2", "ghost", "h", "r", {}))
        assert r.add_attempt("C1", make_attempt("A1", bundle_id="B1"), "f1") == 1
        with pytest.raises(NotFoundError):
            r.add_attempt("C1", make_attempt("A2", bundle_id="B9"), "f2")
        assert r.list_attempts("C1")[0].attempt_id == "A1" and len(r.list_attempts("C1")) == 1
    with SqliteRegistry(path) as r:
        assert r.get_bundle("B1") == bundle
        assert r.get_bundle("B1").to_json() == bundle.to_json()
        assert r.list_bundles("C1") == [bundle]
        assert r.get_attempt("A1").attempt.bundle_id == "B1"
        with pytest.raises(NotFoundError):
            r.get_bundle("B9")


def test_reviewer_reports_per_round(tmp_path):
    path = tmp_path / "r.db"
    reports = [
        ReviewerReport("P1", "R1", "first", T0),
        ReviewerReport("P2", "R2", "second", "2026-10-30T11:00:00+00:00"),
        ReviewerReport("P3", "R1", "third", "2026-10-30T12:00:00+00:00"),
    ]
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        add_round(r, "R1")
        add_round(r, "R2")
        for report in reports:
            r.add_reviewer_report(report)
        before = _rows(path, "reviewer_reports")
        with pytest.raises(NotFoundError):
            r.add_reviewer_report(ReviewerReport("P9", "ghost", "x", T0))
        with pytest.raises(RegistryError):
            r.add_reviewer_report(ReviewerReport("P1", "R2", "x", T0))
        with pytest.raises(NotFoundError):
            r.list_reviewer_reports("C1", "ghost")
        assert _rows(path, "reviewer_reports") == before == 3
    with SqliteRegistry(path) as r:
        assert r.list_reviewer_reports("C1") == reports
        assert r.list_reviewer_reports("C1", "R1") == [reports[0], reports[2]]
        assert [x.to_json() for x in r.list_reviewer_reports("C1", "R2")] == [reports[1].to_json()]


def test_guidance_roundtrip_and_creates_suggestion(tmp_path):
    path = tmp_path / "r.db"
    guidance = Guidance("G1", "R1", "try target encoding", T0)
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1", state="AWAITING_GUIDANCE"))
        add_round(r, "R1")
        r.add_guidance("C1", guidance)
    with SqliteRegistry(path) as r:
        assert r.list_guidance("C1") == [guidance]
        assert r.list_guidance("C1")[0].to_json() == guidance.to_json()
        suggestion = r.get_suggestion("G1")
        assert (suggestion.text, suggestion.origin_cycle, suggestion.spawned_cycles) == (
            "try target encoding", "C1", ())
        assert r.list_suggestions() == [suggestion]


def test_rejected_guidance_writes_nothing(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        r.add_cycle(make_cycle("C2"))
        add_round(r, "R1")
        add_round(r, "R2", cycle_id="C2")
        r.add_suggestion(Suggestion("S1", "old", "C1"))
        r.add_guidance("C1", Guidance("G1", "R1", "ok", T0))
        counts = (_rows(path, "lead_guidance"), _rows(path, "suggestions"))
        cases = [
            Guidance("G2", "ghost", "x", T0),
            Guidance("G3", "R2", "x", T0),
            Guidance("S1", "R1", "x", T0),
            Guidance("G1", "R1", "x", T0),
        ]
        for guidance in cases:
            with pytest.raises((NotFoundError, RegistryError)):
                r.add_guidance("C1", guidance)
            assert (_rows(path, "lead_guidance"), _rows(path, "suggestions")) == counts
        with pytest.raises(NotFoundError):
            r.add_guidance("ghost", Guidance("G4", "R1", "x", T0))
        assert (_rows(path, "lead_guidance"), _rows(path, "suggestions")) == counts


def test_role_traces_are_append_only_and_roundtrip(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        new_attempt(r, "A1")
        add_round(r, "R1", candidates=["A1"])
        call_id = r.add_llm_call(_call(role="modeling"))
        first = make_trace(candidate_id="A1", llm_call_id=call_id)
        second = make_trace(role="coordinator", input_type="Plan", input={"goal": "x"},
                            output_type="Round", output={"tasks": ["a"]})
        r.add_role_trace(first)
        r.add_role_trace(second)
    with SqliteRegistry(path) as r:
        got = r.list_role_traces("C1")
        assert got == [first, second]
        assert [t.to_json() for t in got] == [first.to_json(), second.to_json()]
        assert got[0].llm_call_id == call_id and got[1].candidate_id is None
        names = [n for n in dir(Registry) if "trace" in n]
        assert sorted(names) == ["add_role_trace", "list_role_traces"]


def test_role_traces_filtered_by_role_and_round(reg):
    reg.add_cycle(make_cycle("C2"))
    add_round(reg, "R1")
    add_round(reg, "R2")
    add_round(reg, "R3", cycle_id="C2")
    traces = [
        make_trace(role="modeling", round_id="R1"),
        make_trace(role="reviewer", round_id="R1"),
        make_trace(role="reviewer", round_id="R2"),
        make_trace(role="data", round_id="R2"),
    ]
    for trace in traces:
        reg.add_role_trace(trace)
    other = make_trace("C2", role="reviewer", round_id="R3")
    reg.add_role_trace(other)
    assert reg.list_role_traces("C1") == traces
    assert reg.list_role_traces("C1", role="reviewer") == traces[1:3]
    assert reg.list_role_traces("C1", round_id="R2") == traces[2:]
    assert reg.list_role_traces("C1", role="reviewer", round_id="R2") == [traces[2]]
    assert reg.list_role_traces("C2") == [other]
    with pytest.raises(ValidationError):
        reg.list_role_traces("C1", role="planner")


def test_trace_llm_call_link_is_checked(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        r.add_cycle(make_cycle("C2"))
        add_round(r, "R1")
        mine = r.add_llm_call(_call(role="modeling"))
        theirs = r.add_llm_call(_call(cycle_id="C2"))
        assert isinstance(mine, int) and mine != theirs
        r.add_role_trace(make_trace(llm_call_id=mine))
        for call_id in (9999, theirs):
            with pytest.raises((NotFoundError, RegistryError)):
                r.add_role_trace(make_trace(llm_call_id=call_id))
        assert _rows(path, "role_traces") == 1
        assert r.list_role_traces("C1")[0].llm_call_id == mine


def test_trace_references_are_checked(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        r.add_cycle(make_cycle("C2"))
        new_attempt(r, "B1", "C2")
        add_round(r, "R1")
        add_round(r, "R2", cycle_id="C2")
        bad = [
            make_trace(round_id="ghost"),
            make_trace(round_id="R2"),
            make_trace(candidate_id="ghost"),
            make_trace(candidate_id="B1"),
            make_trace(cycle_id="ghost"),
        ]
        for trace in bad:
            with pytest.raises((NotFoundError, RegistryError)):
                r.add_role_trace(trace)
        for kwargs in ({"role": "planner"}, {"outcome": "done"}, {"input": "text"},
                       {"input": {"x": float("nan")}}):
            with pytest.raises(ValidationError):
                make_trace(**kwargs)
        assert _rows(path, "role_traces") == 0
        r.add_role_trace(make_trace())
        assert _rows(path, "role_traces") == 1


def test_llm_calls_filtered_by_role(reg):
    reg.add_llm_call(_call(role="modeling"))
    reg.add_llm_call(_call(role="reviewer"))
    reg.add_llm_call(_call())
    assert [c.role for c in reg.list_llm_calls("C1")] == ["modeling", "reviewer", None]
    assert [c.role for c in reg.list_llm_calls("C1", role="modeling")] == ["modeling"]
    with pytest.raises(ValidationError):
        reg.list_llm_calls("C1", role="planner")


def test_llm_call_role_survives_reopening(tmp_path):
    path = tmp_path / "r.db"
    entries = [_call(role="data"), _call()]
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        for entry in entries:
            r.add_llm_call(entry)
    with SqliteRegistry(path) as r:
        assert r.list_llm_calls("C1") == entries


def test_metric_series_by_round(reg):
    add_round(reg, "R1")
    add_round(reg, "R2")
    add_round(reg, "R3")
    for name, rnd, mean in (("A1", "R1", 0.7), ("A2", "R1", 0.75), ("A3", "R2", 0.8)):
        reg.add_attempt("C1", make_attempt(name), f"fp-{name}", round_id=rnd)
        reg.add_execution(make_execution(name, metrics=[PartitionMetric("validation", mean, 0.01)]))
    reg.add_attempt("C1", make_attempt("A4"), "fp-A4", round_id="R2")
    reg.add_execution(crash("A4"))
    series = reg.metric_series_by_round("C1", "validation")
    assert [(s.round_id, s.position) for s in series] == [("R1", 1), ("R2", 2), ("R3", 3)]
    assert [[(p.attempt_id, p.mean, p.dispersion) for p in s.points] for s in series] == [
        [("A1", 0.7, 0.01), ("A2", 0.75, 0.01)],
        [("A3", 0.8, 0.01)],
        [],
    ]
    assert reg.metric_series_by_round("C1", "holdout")[0].points == ()
    with pytest.raises(NotFoundError):
        reg.metric_series_by_round("ghost", "validation")


def test_new_records_are_unchanged_after_reopening(tmp_path):
    path = tmp_path / "r.db"
    round_ = Round("R1", ("a", "b"), ("A1",))
    bundle = BundleRecord("B1", "C1", "h", "ref", {"k": [1, 2]})
    report = ReviewerReport("P1", "R1", "text", T0)
    guidance = Guidance("G1", "R1", "go", T0)
    trace = make_trace(candidate_id="A1")
    with SqliteRegistry(path) as r:
        r.add_cycle(make_cycle("C1"))
        new_attempt(r, "A1")
        r.add_round("C1", round_)
        r.add_bundle(bundle)
        r.add_reviewer_report(report)
        r.add_guidance("C1", guidance)
        r.add_role_trace(trace)
        r.add_llm_call(_call(role="reviewer"))
    with SqliteRegistry(path) as r:
        assert r.get_round("R1") == round_ and r.get_round("R1").to_json() == round_.to_json()
        assert r.get_bundle("B1").to_json() == bundle.to_json()
        assert r.list_reviewer_reports("C1")[0].to_json() == report.to_json()
        assert r.list_guidance("C1")[0].to_json() == guidance.to_json()
        assert r.list_role_traces("C1")[0].to_json() == trace.to_json()
        assert r.list_llm_calls("C1")[0] == _call(role="reviewer")


# --- list_cycles and read-only opening (web ticket) ---


def test_list_cycles_in_insertion_order_with_all_fields():
    with SqliteRegistry() as registry:
        assert registry.list_cycles() == []
        registry.add_cycle(make_cycle("C2", project_id="P1", protocol_hash="h"))
        registry.add_cycle(make_cycle("C1"))
        registry.add_cycle(make_cycle("C3", state="DONE"))
        new_attempt(registry, "A1", cycle_id="C2")
        cycles = registry.list_cycles()
        assert [c.id for c in cycles] == ["C2", "C1", "C3"]
        assert cycles == [registry.get_cycle(c) for c in ("C2", "C1", "C3")]
        assert cycles[0].project_id == "P1" and cycles[0].protocol_hash == "h"


def test_read_only_opening_reads_the_same_data_and_never_writes(tmp_path):
    path = tmp_path / "registry.db"
    with SqliteRegistry(path) as writable:
        writable.add_cycle(make_cycle("C1"))
        new_attempt(writable, "A1")
        writable.add_execution(make_execution("A1"))
        writable.close_attempt("A1", "ok")
        expected = (writable.list_cycles(), writable.list_attempts(), writable.get_attempt("A1"))
    before = path.read_bytes()
    with SqliteRegistry(path, read_only=True) as ro:
        assert (ro.list_cycles(), ro.list_attempts(), ro.get_attempt("A1")) == expected
        with pytest.raises(RegistryError, match="read-only"):
            ro.add_cycle(make_cycle("C9"))
        with pytest.raises(RegistryError, match="read-only"):
            ro.set_cycle_state("C1", "DONE")
        with pytest.raises(RegistryError, match="read-only"):
            new_attempt(ro, "A2")
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["registry.db"]


def test_read_only_rejects_memory_and_missing_file_without_creating_it(tmp_path):
    with pytest.raises(RegistryError):
        SqliteRegistry(":memory:", read_only=True)
    missing = tmp_path / "missing.db"
    with pytest.raises(RegistryError, match="does not exist"):
        SqliteRegistry(missing, read_only=True)
    assert not missing.exists()
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    with pytest.raises(RegistryError):
        SqliteRegistry(empty, read_only=True)
