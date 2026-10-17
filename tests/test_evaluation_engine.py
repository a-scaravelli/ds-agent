"""Evaluation engine: validation, holdout session, freezing, comparisons (no network)."""

import numpy as np
import pandas as pd
import pytest
from evaluation_helpers import (
    FailingCandidate,
    MajorityCandidate,
    MeanCandidate,
    OffsetCandidate,
    SeedNoiseCandidate,
    chrono_protocol,
    classification_frame,
    evaluate,
    frozen_engine,
    make_protocol,
    new_registry,
    regression_frame,
)
from registry_helpers import make_cycle, new_attempt

from kaggle_agent.evaluation import (
    EvaluationEngine,
    FrozenSplit,
    HoldoutAccessor,
    HoldoutSessionError,
    ProtocolHashMismatchError,
    ProtocolNotFrozenError,
    dispersion,
)
from kaggle_agent.registry import HoldoutRead, RegistryError, SqliteRegistry

FRAME = classification_frame()
REG_FRAME = regression_frame()


def _majority_scores(frame, protocol, score):
    """Expected per-partition scores computed by hand from the split."""
    split = FrozenSplit.create(frame, protocol)
    out = {}
    for part in split.partitions():
        _, y_train = split.data(part.train_rows)
        _, y_valid = split.data(part.valid_rows)
        out[part.id] = score(y_valid.to_numpy(), y_train.mode().iloc[0])
    return out


def test_metric_is_computed_from_the_protocol_not_hardcoded():
    accuracy = make_protocol(metric="accuracy", direction="maximize")
    error = make_protocol(metric="mae", direction="minimize")
    got_acc = evaluate(frozen_engine(accuracy)[0], MajorityCandidate(), FRAME).scores
    got_err = evaluate(frozen_engine(error)[0], MajorityCandidate(), FRAME).scores
    want_acc = _majority_scores(FRAME, accuracy, lambda y, m: float(np.mean(y == m)))
    want_err = _majority_scores(FRAME, error, lambda y, m: float(np.mean(np.abs(y - m))))
    assert got_acc == pytest.approx(want_acc)
    assert got_err == pytest.approx(want_err)
    assert got_acc != pytest.approx(got_err)


def test_deterministic_scheme_repetitions_change_only_the_candidate_seed():
    protocol = chrono_protocol(repetitions=3, metric="rmse", direction="minimize")
    engine, _ = frozen_engine(protocol)
    spy = MajorityCandidate()
    result = evaluate(engine, spy, REG_FRAME.assign(y=REG_FRAME["y"].round()).rename(columns={}))
    ids = list(result.scores)
    assert ids == [f"r{r}/f{f}" for r in range(3) for f in range(3)]
    seeds = [call[3] for call in spy.calls]
    assert len({seeds[0], seeds[3], seeds[6]}) == 3
    assert seeds[0] == seeds[1] == seeds[2]
    for fold in range(3):
        valid_index = [spy.calls[r * 3 + fold][2].index.tolist() for r in range(3)]
        assert valid_index[0] == valid_index[1] == valid_index[2]
    again = MajorityCandidate()
    evaluate(frozen_engine(protocol)[0], again, REG_FRAME.assign(y=REG_FRAME["y"].round()))
    assert [c[3] for c in again.calls] == seeds


def test_baseline_dispersion_captures_model_seed_noise():
    protocol = chrono_protocol(
        repetitions=4, metric="rmse", direction="minimize",
        validation_params={"time_column": "t", "n_folds": 1},
    )
    noisy = evaluate(frozen_engine(protocol)[0], SeedNoiseCandidate(), REG_FRAME)
    by_rep = [noisy.scores[f"r{r}/f0"] for r in range(4)]
    assert noisy.dispersion > 0
    assert len(set(by_rep)) == 4
    assert noisy.dispersion == pytest.approx(np.std(list(noisy.scores.values()), ddof=1))
    steady = evaluate(frozen_engine(protocol)[0], MeanCandidate(), REG_FRAME)
    assert steady.dispersion == pytest.approx(0.0)
    assert steady.scores["r0/f0"] == pytest.approx(steady.scores["r3/f0"])


def test_dispersion_of_two_repetitions_by_three_folds_is_reported():
    protocol = make_protocol(repetitions=2)
    engine, _ = frozen_engine(protocol)
    result = evaluate(engine, MajorityCandidate(), FRAME)
    assert len(result.scores) == 6
    assert result.dispersion == pytest.approx(np.std(list(result.scores.values()), ddof=1))
    summary = next(m for m in result.execution.metrics if m.partition == "validation")
    assert summary.dispersion == pytest.approx(result.dispersion)
    assert summary.mean == pytest.approx(np.mean(list(result.scores.values())))


def test_hash_frozen_on_the_cycle_once_and_persisted(tmp_path):
    path = tmp_path / "r.db"
    with SqliteRegistry(path) as reg:
        reg.add_cycle(make_cycle("C1"))
        engine = EvaluationEngine(reg, "C1", make_protocol())
        assert reg.get_cycle("C1").protocol_hash is None
        value = engine.freeze()
        assert reg.get_cycle("C1").protocol_hash == value
        assert engine.freeze() == value
        with pytest.raises(RegistryError):
            EvaluationEngine(reg, "C1", make_protocol(threshold_k=9)).freeze()
    with SqliteRegistry(path) as reg:
        assert reg.get_cycle("C1").protocol_hash == value


def test_unfrozen_cycle_is_refused():
    reg = new_registry()
    engine = EvaluationEngine(reg, "C1", make_protocol())
    candidate = MajorityCandidate()
    with pytest.raises(ProtocolNotFrozenError):
        evaluate(engine, candidate, FRAME)
    with pytest.raises(ProtocolNotFrozenError):
        with engine.holdout_session(FRAME):
            pass
    assert candidate.calls == [] and reg.get_holdout_session("C1") is None


@pytest.mark.parametrize("change", [{"threshold_k": 3.0}, {"seed": 12}])
def test_run_blocked_when_protocol_differs_from_frozen(change):
    _, reg = frozen_engine()
    other = EvaluationEngine(reg, "C1", make_protocol(**change))
    candidate = MajorityCandidate()
    with pytest.raises(ProtocolHashMismatchError):
        evaluate(other, candidate, FRAME)
    assert candidate.calls == []


def test_hash_checked_before_every_holdout_read():
    engine, reg = frozen_engine()
    with engine.holdout_session(FRAME) as session:
        engine.protocol = make_protocol(threshold_k=7.0)
        candidate = MajorityCandidate()
        with pytest.raises(ProtocolHashMismatchError):
            session.evaluate_holdout(candidate, "A1")
    assert reg.list_holdout_reads("C1") == [] and candidate.calls == []


def test_holdout_read_logged_and_candidate_never_sees_holdout_labels():
    engine, reg = frozen_engine()
    spy = MajorityCandidate()
    with engine.holdout_session(FRAME) as session:
        result = session.evaluate_holdout(spy, "A1")
    split = FrozenSplit.create(FRAME, engine.protocol)
    holdout = set(split._holdout_rows.tolist())
    (read,) = reg.list_holdout_reads("C1")
    assert (read.cycle_id, read.attempt_id, read.rows) == ("C1", "A1", len(holdout))
    assert read.read_at and result.rows == len(holdout)
    (x_train, y_train, x_valid, _), = spy.calls
    assert set(x_valid.index) == holdout
    assert not set(y_train.index) & holdout and not set(x_train.index) & holdout
    assert all("y" not in arg.columns for arg in (x_train, x_valid))


def test_one_session_with_best_attempt_and_requested_variants():
    engine, reg = frozen_engine()
    with engine.holdout_session(FRAME) as session:
        results = [session.evaluate_holdout(MajorityCandidate(), a) for a in ("A1", "A2", "A3")]
    assert [r.attempt_id for r in results] == ["A1", "A2", "A3"]
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert [r.attempt_id for r in reg.list_holdout_reads("C1")] == ["A1", "A2", "A3"]
    assert reg.get_holdout_session("C1") is not None


def test_no_read_after_the_session_is_closed():
    engine, reg = frozen_engine()
    with engine.holdout_session(FRAME) as session:
        session.evaluate_holdout(MajorityCandidate(), "A1")
    log = reg.list_holdout_reads("C1")
    candidate = MajorityCandidate()
    with pytest.raises(HoldoutSessionError):
        session.evaluate_holdout(candidate, "A2")
    with pytest.raises(HoldoutSessionError):
        with engine.holdout_session(FRAME):
            pass
    assert candidate.calls == [] and reg.list_holdout_reads("C1") == log
    reg.add_cycle(make_cycle("C2"))
    new_attempt(reg, "B1", "C2")
    other = EvaluationEngine(reg, "C2", make_protocol())
    other.freeze()
    with other.holdout_session(FRAME) as session2:
        assert session2.evaluate_holdout(MajorityCandidate(), "B1").rows > 0


def test_read_without_open_session_refused():
    engine, reg = frozen_engine()
    with pytest.raises(RegistryError):
        reg.add_holdout_read(HoldoutRead("C1", "A1", 5, "now"))
    accessor = HoldoutAccessor(FrozenSplit.create(FRAME, engine.protocol), reg, "C1", engine.verify)
    with pytest.raises(HoldoutSessionError):
        accessor.read("A1")
    assert reg.list_holdout_reads("C1") == []


def test_attempt_cannot_be_read_twice_in_a_session():
    engine, reg = frozen_engine()
    with engine.holdout_session(FRAME) as session:
        session.evaluate_holdout(MajorityCandidate(), "A1")
        candidate = MajorityCandidate()
        with pytest.raises(HoldoutSessionError):
            session.evaluate_holdout(candidate, "A1")
    assert len(reg.list_holdout_reads("C1")) == 1 and candidate.calls == []


def test_failed_evaluation_still_consumes_the_session():
    engine, reg = frozen_engine()
    with pytest.raises(RuntimeError, match="boom"):
        with engine.holdout_session(FRAME) as session:
            session.evaluate_holdout(FailingCandidate(), "A1")
    assert [r.attempt_id for r in reg.list_holdout_reads("C1")] == ["A1"]
    assert not reg.get_holdout_session("C1").is_open
    with pytest.raises(HoldoutSessionError):
        with engine.holdout_session(FRAME):
            pass


def test_holdout_does_not_change_validation_decisions():
    engine, reg = frozen_engine(make_protocol(metric="mae", direction="minimize"))
    baseline = evaluate(engine, MeanCandidate(), FRAME, "A1")
    better = evaluate(engine, OffsetCandidate(0.0), FRAME, "A2")
    reg.add_execution(baseline.execution)
    reg.add_execution(better.execution)
    engine.compare("A2", "A1", better.scores, baseline.scores, baseline.scores)
    series = reg.metric_series("C1", "validation")
    comparisons = reg.list_comparisons("C1")
    with engine.holdout_session(FRAME) as session:
        session.evaluate_holdout(MeanCandidate(), "A1")
    assert reg.metric_series("C1", "validation") == series
    assert reg.list_comparisons("C1") == comparisons
    assert reg.count_comparisons("C1") == 1


def test_library_versions_recorded_in_resources_and_stable_hash():
    engine, reg = frozen_engine()
    first = evaluate(engine, MajorityCandidate(), FRAME, "A1")
    second = evaluate(engine, MajorityCandidate(), FRAME, "A2")
    versions = first.execution.resources["library_versions"]
    assert set(versions) == {"pandas", "numpy", "scikit-learn"}
    assert versions["pandas"] == pd.__version__ and versions["numpy"] == np.__version__
    assert first.execution.protocol_hash == second.execution.protocol_hash
    assert first.execution.protocol_hash == reg.get_cycle("C1").protocol_hash
    assert versions["pandas"] not in first.execution.protocol_hash


def test_comparisons_counted_and_threshold_does_not_grow_with_count():
    protocol = make_protocol(threshold_k=1.0)
    engine, reg = frozen_engine(protocol)
    baseline = {"p1": 0.0, "p2": 2.0}
    candidate = {"p1": 3.0, "p2": 3.0}
    reference = {"p1": 1.0, "p2": 1.0}
    results = [engine.compare("A2", "A1", candidate, reference, baseline) for _ in range(4)]
    assert reg.count_comparisons("C1") == 4
    records = reg.list_comparisons("C1")
    assert [(r.candidate_attempt_id, r.reference_attempt_id) for r in records] == [("A2", "A1")] * 4
    assert records[0].difference == pytest.approx(2.0)
    assert records[0].threshold == pytest.approx(dispersion(baseline))
    assert records[0].is_real is True
    assert len({r.threshold for r in results}) == 1


def test_execution_record_fits_the_registry_types():
    engine, reg = frozen_engine()
    result = evaluate(engine, MajorityCandidate(), FRAME, "A1")
    assert reg.add_execution(result.execution) == 1
    detail = reg.get_attempt("A1")
    assert detail.executions[0].protocol_hash == reg.get_cycle("C1").protocol_hash
    (point,) = reg.metric_series("C1", "validation")
    assert point.mean == pytest.approx(result.mean)
    assert point.dispersion == pytest.approx(result.dispersion)
    assert engine.scores_from_execution(detail.executions[0]) == pytest.approx(result.scores)
    assert result.execution.seed == engine.protocol.seed


def test_input_dataframe_unchanged_by_evaluation():
    frame = classification_frame()
    before = frame.copy(deep=True)
    engine, _ = frozen_engine()
    spy = MajorityCandidate()
    evaluate(engine, spy, frame)
    with engine.holdout_session(frame) as session:
        session.evaluate_holdout(spy, "A1")
    pd.testing.assert_frame_equal(frame, before)
    assert all(isinstance(c[0], pd.DataFrame) and isinstance(c[1], pd.Series) for c in spy.calls)
