"""SVG metric chart written by hand (no network)."""

import re

from web_helpers import SCRIPT, no_network  # noqa: F401  (fixture)

from kaggle_agent.registry import MetricPoint, RoundMetrics
from kaggle_agent.web.chart import render_chart


def _rounds():
    return [
        RoundMetrics("R1", 1, (MetricPoint(1, "A1", 0.70, 0.02), MetricPoint(2, "A2", 0.75, 0.01))),
        RoundMetrics("R2", 2, (MetricPoint(4, "B1", 0.80, 0.03),)),
        RoundMetrics("R3", 3, ()),
    ]


def test_one_circle_per_point_and_a_best_line_per_round_with_data():
    svg = render_chart(_rounds())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<circle") == 3
    assert svg.count('class="dispersion"') == 3
    assert svg.count('class="round-best"') == 2
    assert "round 3" not in svg


def test_overall_best_follows_the_direction():
    def best(direction):
        return re.findall(r'class="point best"[^>]*><title>(\w+):', render_chart(_rounds(), direction))

    assert best("max") == ["B1"]
    assert best("min") == ["A1"]


def test_no_data_gives_no_chart():
    assert render_chart([]) is None
    assert render_chart([RoundMetrics("R1", 1, ())]) is None


def test_single_value_does_not_divide_by_zero():
    svg = render_chart([RoundMetrics("R1", 1, (MetricPoint(1, "A1", 0.5, 0.0),))])
    assert svg.count("<circle") == 1 and "nan" not in svg


def test_texts_in_the_svg_are_escaped():
    svg = render_chart([RoundMetrics("R<1>", 1, (MetricPoint(1, SCRIPT, 0.5, 0.1),))])
    assert SCRIPT not in svg and "&lt;script&gt;" in svg
