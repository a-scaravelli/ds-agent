"""Registering a bundle and linking an attempt to it (case 21)."""

import pytest
from bundle_helpers import FRAME, constant_bundle, engine_and_registry, make_runner, raiser_bundle
from registry_helpers import make_attempt, make_cycle

from kaggle_agent.bundle import attempt_with_bundle, load_bundle, register_bundle, schema_of, score_bundle
from kaggle_agent.registry import NotFoundError, SqliteRegistry


def test_register_bundle_and_set_attempt_bundle_id(tmp_path):
    registry = SqliteRegistry()
    registry.add_cycle(make_cycle("C1"))
    loaded = load_bundle(constant_bundle(tmp_path))
    attempt = make_attempt("A1")
    with pytest.raises(NotFoundError):
        registry.add_attempt("C1", attempt_with_bundle(attempt, loaded.bundle_id), "fp-1")
    record = register_bundle(registry, loaded, "C1", reference=str(loaded.folder))
    stored = registry.get_bundle("B1")
    assert stored == record
    assert (stored.id, stored.folder_hash, stored.reference) == ("B1", loaded.folder_hash, str(loaded.folder))
    assert stored.schema == schema_of(loaded.manifest)
    registry.add_attempt("C1", attempt_with_bundle(attempt, "B1"), "fp-1")
    assert registry.get_attempt("A1").attempt.bundle_id == "B1"
    assert attempt.bundle_id is None  # the original attempt is not changed


def test_failed_execution_record_is_accepted_by_the_registry(tmp_path):
    engine, registry = engine_and_registry()
    _, runner = make_runner(raiser_bundle(tmp_path, "predict"))
    outcome = score_bundle(engine, runner, FRAME, attempt_id="A1", execution_id="E1", config={}, data_hash="d")
    registry.add_execution(outcome.execution)
    (stored,) = registry.get_attempt("A1").executions
    assert stored.failure_kind == "code_error" and stored.resources["sandbox_status"] == "crashed"
