"""Metric chart as hand-written SVG: no JavaScript and no library.

One ``<circle>`` per candidate of each round (x = round, y = mean), a
dispersion bar per point, a line on the best value of each round and the
overall best highlighted. Which extreme is best is a parameter because the
registry does not store the direction of the metric. Every text put into the
SVG goes through ``html.escape``.
"""

from html import escape

WIDTH = 720
HEIGHT = 320
MARGIN_LEFT = 60
MARGIN_RIGHT = 20
MARGIN_TOP = 20
MARGIN_BOTTOM = 40


def _pick(values, best: str):
    return max(values) if best == "max" else min(values)


def render_chart(by_round, best: str = "max") -> str | None:
    """SVG for the metric points of the rounds, or None when there is no point.

    ``by_round`` is a list of ``RoundMetrics``; ``best`` is ``max`` or ``min``.
    """
    rounds = [r for r in by_round if r.points]
    if not rounds:
        return None
    points = [p for r in rounds for p in r.points]
    low = min(p.mean - p.dispersion for p in points)
    high = max(p.mean + p.dispersion for p in points)
    if high == low:
        low, high = low - 0.5, high + 0.5
    pad = (high - low) * 0.05
    low, high = low - pad, high + pad
    plot_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    column = plot_w / len(rounds)

    def y_of(value: float) -> float:
        return MARGIN_TOP + (high - value) / (high - low) * plot_h

    overall = _pick([p.mean for p in points], best)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="Metric per round">',
        f'<rect x="{MARGIN_LEFT}" y="{MARGIN_TOP}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#999"/>',
    ]
    for value in (low + (high - low) * f for f in (0.0, 0.25, 0.5, 0.75, 1.0)):
        y = y_of(value)
        parts.append(
            f'<text x="{MARGIN_LEFT - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11">'
            f"{escape(f'{value:.4g}')}</text>"
        )
    for index, r in enumerate(rounds):
        left = MARGIN_LEFT + index * column
        centre = left + column / 2
        label = f"round {r.position}"
        parts.append(
            f'<text x="{centre:.1f}" y="{HEIGHT - 18}" text-anchor="middle" font-size="12">'
            f"{escape(label)}</text>"
        )
        best_mean = _pick([p.mean for p in r.points], best)
        y_best = y_of(best_mean)
        parts.append(
            f'<line class="round-best" x1="{left + 4:.1f}" x2="{left + column - 4:.1f}" '
            f'y1="{y_best:.1f}" y2="{y_best:.1f}" stroke="#2a7" stroke-dasharray="4 3">'
            f"<title>{escape(f'best of round {r.position}: {best_mean:.6g}')}</title></line>"
        )
        step = column / (len(r.points) + 1)
        for k, p in enumerate(r.points, start=1):
            x = left + step * k
            top, bottom = y_of(p.mean + p.dispersion), y_of(p.mean - p.dispersion)
            parts.append(
                f'<line class="dispersion" x1="{x:.1f}" x2="{x:.1f}" y1="{top:.1f}" '
                f'y2="{bottom:.1f}" stroke="#666"/>'
            )
            is_best = p.mean == overall
            css = "point best" if is_best else "point"
            fill, radius = ("#d33", 7) if is_best else ("#36c", 4)
            title = f"{p.attempt_id}: mean {p.mean:.6g}, dispersion {p.dispersion:.6g}"
            parts.append(
                f'<circle class="{css}" cx="{x:.1f}" cy="{y_of(p.mean):.1f}" r="{radius}" '
                f'fill="{fill}"><title>{escape(title)}</title></circle>'
            )
    parts.append("</svg>")
    return "\n".join(parts)
