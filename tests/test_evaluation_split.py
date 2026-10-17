"""FrozenSplit: same seed same split, repetitions, nothing of the holdout exposed (no network)."""

import numpy as np
import pandas as pd
import pytest
from evaluation_helpers import chrono_protocol, classification_frame, make_protocol

from kaggle_agent.evaluation import FrozenSplit
from kaggle_agent.proposals import ValidationError

FRAME = classification_frame()


def _snapshot(split):
    return (
        split.development_rows().tolist(),
        [(p.id, p.train_rows.tolist(), p.valid_rows.tolist()) for p in split.partitions()],
    )


def test_same_seed_same_split_different_seed_differs():
    protocol = make_protocol()
    first = FrozenSplit.create(FRAME, protocol)
    second = FrozenSplit.create(FRAME, protocol)
    other = FrozenSplit.create(FRAME, make_protocol(seed=12))
    assert _snapshot(first) == _snapshot(second)
    assert _snapshot(first) != _snapshot(other)
    assert first._holdout_rows.tolist() == second._holdout_rows.tolist()


def test_random_repetitions_differ_but_are_deterministic():
    protocol = make_protocol(repetitions=3)
    split = FrozenSplit.create(FRAME, protocol)
    ids = [p.id for p in split.partitions()]
    assert ids == [f"r{r}/f{f}" for r in range(3) for f in range(3)]
    assert len(set(ids)) == len(ids)
    by_rep = {r: [p.valid_rows.tolist() for p in split.partitions() if p.repetition == r]
              for r in range(3)}
    assert by_rep[0] != by_rep[1] != by_rep[2]
    assert _snapshot(split) == _snapshot(FrozenSplit.create(FRAME, protocol))


def test_deterministic_scheme_repetitions_keep_identical_folds():
    split = FrozenSplit.create(FRAME, chrono_protocol(repetitions=3))
    ids = [p.id for p in split.partitions()]
    assert ids == [f"r{r}/f{f}" for r in range(3) for f in range(3)]
    parts = split.partitions()
    for fold in range(3):
        base = parts[fold]
        for rep in (1, 2):
            other = parts[rep * 3 + fold]
            assert base.train_rows.tolist() == other.train_rows.tolist()
            assert base.valid_rows.tolist() == other.valid_rows.tolist()


def test_split_exposes_only_development_data():
    split = FrozenSplit.create(FRAME, make_protocol())
    holdout = set(split._holdout_rows.tolist())
    development = set(split.development_rows().tolist())
    assert holdout and not holdout & development
    assert len(holdout) + len(development) == len(FRAME)
    for part in split.partitions():
        union = set(part.train_rows.tolist()) | set(part.valid_rows.tolist())
        assert union == development
        assert not set(part.train_rows.tolist()) & set(part.valid_rows.tolist())
    public = [n for n in vars(split) if not n.startswith("_")]
    assert public == []
    text = repr(split) + " ".join(repr(p) for p in split.partitions())
    assert "holdout" not in text
    with pytest.raises(ValidationError):
        split.data(np.array(sorted(holdout)))
    x, y = split.data(split.partitions()[0].valid_rows)
    assert len(x) == len(y) and "y" not in x.columns


def test_input_dataframe_is_not_modified_and_candidates_get_pandas():
    frame = classification_frame()
    before = frame.copy(deep=True)
    split = FrozenSplit.create(frame, make_protocol())
    x, y = split.data(split.partitions()[0].train_rows)
    x["x1"] = 999.0
    y.iloc[:] = -1
    pd.testing.assert_frame_equal(frame, before)
    assert isinstance(x, pd.DataFrame) and isinstance(y, pd.Series)
    again, _ = split.data(split.partitions()[0].train_rows)
    assert (again["x1"] != 999.0).all()
    with pytest.raises(ValidationError):
        FrozenSplit.create(frame.to_numpy(), make_protocol())
    with pytest.raises(ValidationError):
        FrozenSplit.create(frame.drop(columns=["y"]), make_protocol())
