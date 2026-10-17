"""Fingerprint of an attempt (no network)."""

import pytest

from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import compute_fingerprint, normalize_config

BASE = ({"lr": 0.1, "layers": [1, 2], "name": "a"}, "code", "data", "protocol")


def test_fingerprint_is_deterministic_on_normalized_config():
    config, code, data, protocol = BASE
    reordered = {"name": "a", "layers": (1, 2), "lr": 0.1}
    assert compute_fingerprint(config, code, data, protocol) == compute_fingerprint(
        reordered, code, data, protocol
    )
    assert compute_fingerprint({"x": 1}, code, data, protocol) == compute_fingerprint(
        {"x": 1.0}, code, data, protocol
    )
    assert compute_fingerprint({"x": 1}, "c", "d", "p") != compute_fingerprint(
        {"x": 1.5}, "c", "d", "p"
    )
    assert normalize_config({"b": 1, "a": (1.0, None)}) == '{"a":[1,null],"b":1}'
    for bad in ({"x": object()}, {"x": float("nan")}, {"x": float("inf")}, {1: "a"}, {"x": {1, 2}}):
        with pytest.raises(ValidationError):
            compute_fingerprint(bad, code, data, protocol)


def test_fingerprint_includes_code_data_protocol_and_config():
    base = compute_fingerprint(*BASE)
    variants = [
        ({"lr": 0.2, "layers": [1, 2], "name": "a"}, *BASE[1:]),
        (BASE[0], "code2", *BASE[2:]),
        (BASE[0], BASE[1], "data2", BASE[3]),
        (*BASE[:3], "protocol2"),
    ]
    prints = {compute_fingerprint(*v) for v in variants}
    assert base not in prints and len(prints) == 4
    # parts are separated unambiguously
    assert compute_fingerprint({}, "ab", "c", "d") != compute_fingerprint({}, "a", "bc", "d")
