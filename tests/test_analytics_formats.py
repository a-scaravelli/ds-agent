"""Formats and vocabulary of the analytics: OOF file, bounded decoder, task mapping, families (cases 9, 10)."""

import gzip
import io

import numpy as np
import pandas as pd
import pytest
from analytics_helpers import oof_bytes, oof_frame

from kaggle_agent.analytics import oof_format
from kaggle_agent.analytics.families import feature_families
from kaggle_agent.analytics.oof_format import OofError, decode_oof, encode_oof
from kaggle_agent.analytics.tasks import panel_support, task_of


# 10
def test_equal_input_gives_equal_bytes_and_the_table_round_trips():
    frame = oof_frame([0, 1, 1, 0], proba=[0.2, 0.7, 0.6, 0.1], reps=2)
    first, second = oof_bytes(frame), oof_bytes(frame.copy())
    assert first == second
    table = decode_oof(first)
    assert table.matrix and table.classes == (0, 1)
    assert list(table.frame.columns) == ["row", "rep", "fold", "truth", "p0", "p1"]
    assert np.allclose(table.frame["p1"], frame["p1"])


# 10
def test_the_row_cap_and_the_byte_cap_refuse_a_table():
    frame = oof_frame([0, 1] * 5, prediction=[0, 1] * 5)
    with pytest.raises(OofError, match="over the limit"):
        encode_oof(frame["row"], frame["rep"], frame["fold"], frame["truth"], frame["prediction"], max_rows=5)
    with pytest.raises(OofError, match="more than"):
        decode_oof(oof_bytes(frame), max_rows=5)
    with pytest.raises(OofError, match="size limit"):
        decode_oof(oof_bytes(frame), max_bytes=20)


# 10
def test_a_gzip_bomb_and_bad_columns_are_refused_with_a_clear_error():
    bomb = gzip.compress(b"row,rep,fold,truth,prediction\n" + b"0,0,0,0,0\n" * 200_000)
    with pytest.raises(OofError, match="size limit"):
        decode_oof(bomb, max_bytes=1000)
    with pytest.raises(OofError, match="not a gzip"):
        decode_oof(b"not gzip at all")
    wrong = gzip.compress(b"a,b,c\n1,2,3\n")
    with pytest.raises(OofError, match="first columns"):
        decode_oof(wrong)
    odd = gzip.compress(b"row,rep,fold,truth,prediction,extra\n0,0,0,0,0,0\n")
    with pytest.raises(OofError, match="prediction columns"):
        decode_oof(odd)
    text = io.BytesIO()
    with gzip.GzipFile(fileobj=text, mode="wb") as stream:
        stream.write(b"row,rep,fold,truth,prediction\n1.5,0,0,0,0\n")
    with pytest.raises(OofError, match="integers"):
        decode_oof(text.getvalue())
    assert oof_format.MAX_ROWS == 200_000 and oof_format.MAX_COMPRESSED == 25 * 1024 * 1024


# 9
def test_a_labels_table_has_a_prediction_column_and_no_probability_columns():
    table = decode_oof(oof_bytes(oof_frame([0, 1, 1], prediction=[0, 1, 0])), classification=True)
    assert not table.matrix and "prediction" in table.frame.columns
    assert not [c for c in table.frame.columns if c.startswith("p") and c[1:].isdigit()]
    assert table.classes == (0, 1)


def test_task_mapping_and_panel_support():
    assert task_of("rmse") == {"task": "regression", "input_kind": "values"}
    assert task_of("accuracy")["task"] == "classification"
    assert panel_support("labels")["roc"] != True  # noqa: E712  (a reason text)
    assert panel_support("probabilities")["calibration"] is True
    assert panel_support("scores")["calibration"] != True  # noqa: E712


def test_families_group_derived_features_under_their_source_and_stems_together():
    fam = feature_families(["age", "age_squared", "fare", "cabin_A", "cabin_B", "other"],
                           [{"name": "age_squared", "sources": ["age"]}], known_columns=["age", "fare", "cabin"])
    assert fam["age"] == fam["age_squared"] == "age"
    assert fam["cabin_A"] == fam["cabin_B"] == "cabin"
    assert fam["other"] == "other" and fam["fare"] == "fare"
    assert fam == feature_families(["other", "cabin_B", "fare", "cabin_A", "age_squared", "age"],
                                   [{"name": "age_squared", "sources": ["age"]}], known_columns=["age", "fare", "cabin"])
