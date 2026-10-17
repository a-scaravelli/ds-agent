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
