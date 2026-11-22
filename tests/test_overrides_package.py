"""The neutral overrides package: records, labels, overlay, sealing and search (cases 29, 34, 35, 36, 45, 53)."""

import ast
import sys
from pathlib import Path

import pytest

from kaggle_agent.overrides import (
    KINDS,
    LABELS,
    REASON_REQUIRED,
    apply_bar_overrides,
    classify_read,
    normalize_payload,
    records_for_cycle,
    resolve_choice,
    run_labels,
    search_items,
    seal_state,
    valid_bar_overrides,
)
from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import ComparisonRecord

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "kaggle_agent" / "overrides"


def comparison(candidate, difference, real=False):
    return ComparisonRecord("c", candidate, "ref", difference, 0.05, real)


# 53
def test_the_package_imports_only_the_standard_library_and_the_proposals():
    stdlib = set(sys.stdlib_module_names)
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= stdlib, path.name
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split(".")[0] in stdlib | {"kaggle_agent"}, path.name
                if node.module.startswith("kaggle_agent"):
                    assert node.module.split(".")[1] == "proposals", (path.name, node.module)


# 45
def test_the_risky_kinds_need_a_reason_and_steering_try_this_and_rerun_do_not():
    assert {"choose_model", "bar_override"} <= set(REASON_REQUIRED) <= set(KINDS) | set(REASON_REQUIRED)
    with pytest.raises(ValidationError, match="reason"):
        normalize_payload("choose_model", {"attempt": "a1"})
    with pytest.raises(ValidationError, match="reason"):
        normalize_payload("bar_override", {"scope": "round", "round": 1, "attempt": "a1"})
    assert normalize_payload("choose_model", {"attempt": "a1", "reason": "better on fold 3"})[2] == "better on fold 3"
    normalize_payload("steering", {"steering": {"focus": "model_tuning"}})
    normalize_payload("try_this", {"request": {"hypothesis": "h"}, "from_round": 2})
    assert normalize_payload("rerun", {"attempt": "a1", "seed_offset": 3})[0] == {"attempt": "a1", "seed_offset": 3}
    with pytest.raises(ValidationError):
        normalize_payload("rerun", {"attempt": "../x"})
    with pytest.raises(ValidationError):
        normalize_payload("steering", {"steering": {}})
    with pytest.raises(ValidationError):
        normalize_payload("unknown", {})


# 34
def test_a_round_override_sets_is_real_and_keeps_the_raw_numbers():
    rows = [comparison("a1", 0.01), comparison("a2", 0.2, real=True), comparison("a3", -0.1)]
    out = apply_bar_overrides(rows, [{"id": "d1", "scope": "round", "round": 1, "attempt": "a1"}], {"a1": 1})
    assert [c.is_real for c in out] == [True, True, False]
    assert (out[0].difference, out[0].threshold) == (0.01, 0.05)
    assert rows[0].is_real is False  # the stored record is not rewritten
    assert apply_bar_overrides(rows, [], {}) == rows


# 36
def test_a_cycle_override_counts_positive_sub_bar_rounds_from_round_k_only():
    rows = [comparison("a1", 0.01), comparison("a2", 0.02), comparison("a3", -0.02)]
    out = apply_bar_overrides(rows, [{"id": "d1", "scope": "cycle", "round": 2}], {"a1": 1, "a2": 2, "a3": 3})
    assert [c.is_real for c in out] == [False, True, False]


# 35
def test_a_stale_override_is_split_off_and_a_pending_one_waits():
    records = [{"id": "d1", "scope": "round", "round": 1, "attempt": "a"},
               {"id": "d2", "scope": "round", "round": 2, "attempt": "b"},
               {"id": "d3", "scope": "round", "round": 3, "attempt": "c"},
               {"id": "d4", "scope": "cycle", "round": 1}]
    valid, stale, pending = valid_bar_overrides(records, {}, rounds_started=2, last_complete=2)
    assert [r["id"] for r in valid] == ["d2"] and [r["id"] for r, _ in stale] == ["d1", "d4"]
    assert [r["id"] for r in pending] == ["d3"] and "round 1" in stale[0][1]
    valid, stale, _ = valid_bar_overrides(records[:1], {"d1": "applied"}, rounds_started=3, last_complete=3)
    assert [r["id"] for r in valid] == ["d1"] and not stale
    assert valid_bar_overrides(records[:1], {"d1": "stale"}, 1, 1) == ([], [], [])


# 31
def test_the_ds_choice_wins_when_it_is_usable_and_the_program_pick_is_kept():
    records = [{"id": "d1", "kind": "choose_model", "attempt": "a2"},
               {"id": "d2", "kind": "choose_model", "attempt": "a9"}]
    assert resolve_choice("a1", records, {"a1", "a2"})[0] == "a2"
    assert resolve_choice("a1", records, {"a1"}) == ("a1", None)
    assert resolve_choice("a1", records, {"a1", "a2"}, {"d1": "stale"}) == ("a1", None)


# 4 6 49
def test_the_labels_come_from_the_records_and_stale_ones_are_not_labelled():
    decisions = [{"id": "d1", "kind": "choose_model", "attempt": "a2", "reason": "why"},
                 {"id": "d2", "kind": "bar_override", "scope": "round", "round": 1, "attempt": "a1", "reason": "r"},
                 {"id": "d3", "kind": "steering", "clear": False}, {"id": "d4", "kind": "steering", "clear": True},
                 {"id": "d5", "kind": "rerun", "attempt": "a1", "seed_offset": 2},
                 {"id": "d6", "kind": "choose_model", "attempt": "a3", "reason": "old"}]
    reads = [{"kind": "first"}, {"kind": "second", "reason": "again"}]
    labels = run_labels("c", "c", decisions, {"d1": "applied", "d6": "stale"}, reads, None)
    names = [x["label"] for x in labels]
    assert names == ["chosen by the DS", "bar overridden by the DS", "steered by the DS",
                     "re-run with another seed", "holdout re-read"]
    assert set(names) <= set(LABELS) and labels[0]["status"] == "applied" and labels[1]["status"] == "pending"
    assert labels[-1]["reason"] == "again"
    assert run_labels("c", "c", [], {}, [{"kind": "first"}], None) == []


# 6
def test_a_fork_is_labelled_by_its_protocol_and_the_repeated_look():
    same = {"protocol_label": "same"}
    changed = {"protocol_label": "changed", "reason": "stricter", "changes": {"threshold_k": [1.0, 2.0]}}
    assert [x["label"] for x in run_labels("c-2", "c-1", [], {}, [], same, {"origin_read": True})] == ["holdout re-read"]
    assert run_labels("c-2", "c-1", [], {}, [], same, {"origin_read": False}) == []
    out = run_labels("c-2", "c-1", [], {}, [], changed, {"origin_read": True, "protocol_hash": "abc"})
    assert [x["label"] for x in out] == ["new protocol (set by the DS)"] and out[0]["protocol_hash"] == "abc"


# 1 3 4
def test_the_holdout_seal_states_and_the_kind_of_a_read():
    assert seal_state([], False) == "sealed" and seal_state([], True) == "ready"
    assert seal_state([{"id": "h"}], True) == "unsealed"
    assert classify_read("AWAITING_REVIEW", []) == "first" and classify_read("STOPPED", []) == "first"
    assert classify_read("AWAITING_GUIDANCE", []) == "early"
    assert classify_read("STOPPED", [{"id": "h"}]) == "second" and classify_read("RUNNING", [{"id": "h"}]) == "second"


# 29
def test_the_search_is_case_insensitive_over_text_title_and_tags_and_an_empty_query_lists_all():
    items = [{"text": "Target ENCODING helped", "tags": ["fe"]}, {"text": "n/a", "tags": ["Leakage", "id"]},
             {"title": "Suggestion", "text": "try lightgbm"}]
    assert len(search_items(items, "")) == 3 and len(search_items(items, None)) == 3
    assert search_items(items, "encoding") == [items[0]]
    assert search_items(items, "LEAKAGE") == [items[1]]
    assert search_items(items, "suggestion lightgbm") == [items[2]]
    assert search_items(items, "encoding lightgbm") == []


def test_records_without_a_cycle_belong_to_the_first_cycle():
    records = [{"id": 1}, {"id": 2, "cycle": "c-2"}, {"id": 3, "cycle": "c-1"}]
    assert [r["id"] for r in records_for_cycle(records, "c-1", "c-1")] == [1, 3]
    assert [r["id"] for r in records_for_cycle(records, "c-2", "c-1")] == [2]
