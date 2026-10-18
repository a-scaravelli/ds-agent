"""Protocol: separate from the plan, every field mandatory, no defaults (no network)."""

import inspect

import pytest
from evaluation_helpers import make_protocol, protocol_dict

from kaggle_agent.evaluation import Protocol, schemes
from kaggle_agent.proposals import ShortPlan, ValidationError

PLAN = {"metric": "m", "validation": "v", "feature_engineering": ["f"], "models": ["x"],
        "baseline": "b"}


def test_protocol_is_separate_from_short_plan_and_round_trips():
    before = ShortPlan.from_dict(PLAN)
    protocol = Protocol.from_dict(protocol_dict(metric_params={}, holdout_share=0.25))
    assert Protocol.from_json(protocol.to_json()) == protocol
    assert Protocol.from_dict(protocol.to_dict()) == protocol
    assert set(protocol.to_dict()) == {
        "metric", "metric_params", "direction", "target", "validation_scheme",
        "validation_params", "holdout_scheme", "holdout_params", "holdout_share", "seed",
        "repetitions", "threshold_k",
    }
    assert ShortPlan.from_dict(PLAN) == before
    assert not isinstance(protocol, ShortPlan)
    with pytest.raises(ValidationError):
        Protocol.from_dict({**protocol.to_dict(), "extra": 1})
    with pytest.raises(ValidationError):
        Protocol.from_json("{not json")


@pytest.mark.parametrize("field", ["validation_scheme", "holdout_scheme"])
def test_protocol_without_scheme_is_rejected_and_code_has_no_default(field):
    data = protocol_dict()
    del data[field]
    with pytest.raises(ValidationError) as exc:
        Protocol.from_dict(data)
    assert exc.value.field == field
    for name in ("validation_scheme", "holdout_scheme"):
        param = inspect.signature(Protocol).parameters[name]
        assert param.default is inspect.Parameter.empty
    assert "DEFAULT_SCHEME" not in inspect.getsource(schemes).upper().replace(" ", "")


def test_protocol_without_metric_is_rejected():
    data = protocol_dict()
    del data["metric"]
    with pytest.raises(ValidationError) as exc:
        Protocol.from_dict(data)
    assert exc.value.field == "metric"
    assert inspect.signature(Protocol).parameters["metric"].default is inspect.Parameter.empty


def test_protocol_without_direction_is_rejected():
    data = protocol_dict()
    del data["direction"]
    with pytest.raises(ValidationError) as exc:
        Protocol.from_dict(data)
    assert exc.value.field == "direction"
    assert inspect.signature(Protocol).parameters["direction"].default is inspect.Parameter.empty
    with pytest.raises(ValidationError):
        make_protocol(direction="sideways")


@pytest.mark.parametrize("field", ["seed", "repetitions", "threshold_k", "holdout_share", "target"])
def test_every_other_field_is_mandatory(field):
    data = protocol_dict()
    del data[field]
    with pytest.raises(ValidationError) as exc:
        Protocol.from_dict(data)
    assert exc.value.field == field


def test_threshold_k_missing_or_negative_is_rejected():
    for bad in (-0.1, "2", None, float("nan")):
        with pytest.raises(ValidationError) as exc:
            make_protocol(threshold_k=bad)
        assert exc.value.field == "threshold_k"
    assert make_protocol(threshold_k=0).threshold_k == 0


def test_holdout_share_and_numeric_fields_validated():
    for share in (0, 1, -0.2, 1.5, "0.2", True):
        with pytest.raises(ValidationError):
            make_protocol(holdout_share=share)
    for field, value in (("seed", -1), ("seed", 1.5), ("repetitions", 0), ("repetitions", True)):
        with pytest.raises(ValidationError):
            make_protocol(**{field: value})


def test_fewer_than_two_partitions_is_rejected():
    with pytest.raises(ValidationError) as exc:
        make_protocol(validation_params={"n_folds": 1})
    assert exc.value.field == "n_folds"

    class OneFold(schemes.RandomScheme):
        name = "one_fold_test"

        def validate_params(self, params, role):
            pass

        def n_folds(self, params):
            return 1

    scheme = OneFold()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(schemes, "SCHEMES", {**schemes.SCHEMES, scheme.name: scheme})
        with pytest.raises(ValidationError) as exc:
            make_protocol(validation_scheme="one_fold_test", validation_params={})
        assert exc.value.field == "repetitions"
        assert make_protocol(
            validation_scheme="one_fold_test", validation_params={}, repetitions=2
        ).repetitions == 2


def test_two_repetitions_times_three_folds_are_accepted():
    assert make_protocol(repetitions=2).repetitions == 2
