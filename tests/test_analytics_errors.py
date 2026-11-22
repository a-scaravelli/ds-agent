"""Segments, hardest rows, leakage and sanity flags (cases 21, 22, 23, 24)."""

import numpy as np
import pandas as pd
import pytest

from kaggle_agent.analytics import flags, hardest, segments
from kaggle_agent.analytics.errors import row_errors
from kaggle_agent.proposals import ValidationError
from analytics_helpers import oof_frame


def _frame_and_errors():
    n = 40
    errors = np.zeros(n)
    errors[[0, 1]] = 1                  # category a: 2 of 20
    errors[20:28] = 1                   # category b: 8 of 20
    missing = np.arange(n, dtype=float)
    missing[36:] = np.nan               # four rows without a value, no error among them
    features = pd.DataFrame({"c": ["a"] * 20 + ["b"] * 20, "f": np.arange(1, n + 1, dtype=float), "m": missing})
    return features, pd.DataFrame({"error": errors})


# 21
def test_segments_are_exact_per_category_bin_and_missing_and_ranked_by_lift():
    features, errors = _frame_and_errors()
    table = segments.build_segments(features, errors, "classification", interactions=False)
    assert table["overall_error"] == pytest.approx(0.25) and table["rows"] == 40
    by = {(s["feature"], s["segment"]): s for s in table["segments"]}
    assert by[("c", "a")]["count"] == 20 and by[("c", "a")]["error"] == pytest.approx(0.1)
    assert by[("c", "a")]["lift"] == pytest.approx(0.4)
    assert by[("c", "b")]["lift"] == pytest.approx(1.6)
    assert by[("m", "(missing)")]["count"] == 4 and by[("m", "(missing)")]["error"] == 0.0
    first = table["segments"][0]
    assert first["feature"] == "f" and first["count"] == 5 and first["error"] == 1.0 and first["lift"] == pytest.approx(4.0)
    lifts = [s["lift"] for s in table["segments"]]
    assert lifts == sorted(lifts, reverse=True)
    assert sum(s["count"] for s in table["segments"] if s["feature"] == "f") == 40  # at most 8 bins, all rows


def test_regression_segments_carry_the_signed_error():
    features = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    errors = pd.DataFrame({"error": [1.0, 1.0, 3.0, 3.0], "signed": [1.0, -1.0, 3.0, 3.0]})
    table = segments.build_segments(features, errors, "regression", interactions=False)
    b = next(s for s in table["segments"] if s["segment"] == "b")
    assert b["error"] == 3.0 and b["mean_signed_error"] == 3.0 and b["lift"] == pytest.approx(1.5)


# 22
def test_an_interaction_that_only_the_pair_shows_is_ranked_first_and_min_size_filters():
    n = 80
    a = np.array([0, 1] * 40)
    b = np.array(([0] * 2 + [1] * 2) * 20)
    features = pd.DataFrame({"a": a, "b": b, "noise": np.arange(n) % 5})
    errors = pd.DataFrame({"error": ((a == 1) & (b == 1)).astype(float)})
    table = segments.build_segments(features, errors, "classification")
    top = table["interactions"][0]
    assert top["feature"] == "a x b" and top["segment"] == "1 & 1" and top["count"] == 20 and top["lift"] == pytest.approx(4.0)
    assert max(s["lift"] for s in table["segments"]) < top["lift"]
    full_order = [(s["feature"], s["segment"]) for s in segments.filter_segments(table, 1, 1000)["segments"]]
    for size in (1, 21, 200):
        shown = segments.filter_segments(table, size, 1000)
        assert all(s["count"] >= size for s in shown["segments"] + shown["interactions"])
        kept = [(s["feature"], s["segment"]) for s in shown["segments"]]
        assert kept == [k for k in full_order if k in set(kept)]  # the ranking of the rest is unchanged
    assert segments.filter_segments(table, 21, 10)["interactions"] == []
    assert segments.filter_segments(table, 200, 10)["segments"] == []


def test_the_interaction_search_respects_its_caps():
    n = 60
    features = pd.DataFrame({f"f{i}": np.arange(n) % (i + 2) for i in range(30)})
    errors = pd.DataFrame({"error": (np.arange(n) % 3 == 0).astype(float)})
    table = segments.build_segments(features, errors, "classification")
    pairs = {s["feature"] for s in table["interactions"]}
    assert len(pairs) <= segments.MAX_PAIR_FEATURES * (segments.MAX_PAIR_FEATURES - 1) // 2
    assert all(len(g) <= segments.PAIR_BINS ** 2 for g in [[s for s in table["interactions"] if s["feature"] == p] for p in pairs])


# 23
def test_hardest_rows_come_with_features_sorted_and_capped():
    features = pd.DataFrame({"x": [5, 4, 3, 2, 1], "name": list("abcde")}, index=[10, 11, 12, 13, 14])
    errors = pd.DataFrame({"error": [0.0, 0.9, 0.3, 0.9, 0.1], "truth": [0, 1, 1, 0, 1], "prediction": [0, 0, 1, 1, 1]},
                          index=pd.Index([10, 11, 12, 13, 14], name="row"))
    worst = hardest.hardest_rows(features, errors, limit=3)
    assert [r["row"] for r in worst["rows"]] == [11, 13, 12]
    assert worst["rows"][0]["x"] == 4 and worst["rows"][0]["name"] == "b"
    by_x = hardest.hardest_rows(features, errors, sort="x", order="asc", limit=3)
    assert [r["x"] for r in by_x["rows"]] == [2, 3, 4]
    by_x_desc = hardest.hardest_rows(features, errors, sort="x", order="desc", limit=3)
    assert [r["x"] for r in by_x_desc["rows"]] == [4, 3, 2]
    for bad in ({"sort": "nope"}, {"order": "up"}, {"limit": 201}, {"limit": 0}):
        with pytest.raises(ValidationError):
            hardest.hardest_rows(features, errors, **bad)
    wide = pd.DataFrame({f"c{i}": [1, 2, 3, 4, 5] for i in range(40)}, index=features.index)
    assert len(hardest.hardest_rows(wide, errors)["columns"]) == 4 + hardest.MAX_FEATURE_COLUMNS


def test_row_errors_average_over_repetitions_and_use_log_loss_for_probabilities():
    frame = oof_frame([0, 1], prediction=[0, 0], reps=2)
    err = row_errors(frame, "classification", (0, 1), "labels")
    assert err["error"].tolist() == [0.0, 1.0]
    proba = oof_frame([0, 1], proba=[0.1, 0.8])
    assert row_errors(proba, "classification", (0, 1), "probabilities")["log_loss"].tolist() == pytest.approx(
        [-np.log(0.9), -np.log(0.8)])
    assert row_errors(oof_frame([0, 1], prediction=[0.2, 3.0]), "classification", (0, 1), "scores") is None


# 24
def test_column_flags_find_ids_and_target_copies_and_leave_a_clean_column_alone():
    n = 100
    target = pd.Series(np.arange(n) % 2)
    features = pd.DataFrame({
        "rid": np.arange(n), "leak": target.to_numpy(), "near": target.to_numpy() + 0.05 * np.sin(np.arange(n)),
        "clean": np.arange(n) % 7, "noise": np.cos(np.arange(n) * 1.3)})
    assert abs(np.corrcoef(features["near"], target)[0, 1]) > 0.98
    found = {f["subject"]: f for f in flags.column_flags(features, target)}
    assert set(found) == {"rid", "leak", "near"}
    assert found["rid"]["kind"] == "id_like" and found["leak"]["kind"] == "target_like"
    for item in found.values():
        assert "heuristic, not proof" in item["text"].lower()


def test_score_flags_find_a_jump_and_a_perfect_result_but_not_a_normal_gain():
    history = [{"attempt": "a1", "score": 0.70}, {"attempt": "a2", "score": 0.71}, {"attempt": "a3", "score": 0.81}]
    assert [f["kind"] for f in flags.score_flags(history, "a3", "maximize", 0.01)] == ["jump"]
    assert flags.score_flags(history, "a2", "maximize", 0.01) == []
    assert flags.score_flags(history, "a3", "minimize", 0.01) == []
    perfect = flags.score_flags([], "a1", "maximize", None, perfect=True)
    assert perfect[0]["kind"] == "perfect" and "heuristic" in perfect[0]["text"].lower()
    assert flags.is_perfect(pd.DataFrame({"error": [0.0, 0.0]})) and not flags.is_perfect(pd.DataFrame({"error": [0.0, 1.0]}))
