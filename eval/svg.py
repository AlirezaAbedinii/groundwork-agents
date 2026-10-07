"""A small SVG step chart: precision and recall of the refusal gate against its threshold.

One panel per split (dev, where the threshold is chosen, and test, where it's
measured), sharing both axes, with the chosen threshold drawn on each. The gate's
precision and recall only change at observed scores, so they're drawn as steps. Colors
are the first two slots of a CVD-validated categorical palette, with dark-mode steps
selected by ``prefers-color-scheme``; text uses ink tokens, never the series colors.
"""

from __future__ import annotations

import math
from html import escape

# (threshold, precision, recall); +inf is the "refuse everything" end, None is undefined.
Point = tuple[float, float | None, float | None]

STYLE = """
.surface{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.muted{fill:#898781}
.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}.rule{stroke:#52514e}
.p{stroke:#2a78d6}.r{stroke:#eb6834}.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.halo{paint-order:stroke;stroke:#fcfcfb;stroke-width:4px}
@media (prefers-color-scheme:dark){.surface{fill:#1a1a19}.ink{fill:#fff}.ink2{fill:#c3c2b7}
.grid{stroke:#2c2c2a}.axis{stroke:#383835}.rule{stroke:#c3c2b7}.p{stroke:#3987e5}.r{stroke:#d95926}
.halo{stroke:#1a1a19}}
text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif}
"""
W, H, PANEL_W, PANEL_H, LEFT, TOP, GAP = 680, 300, 280, 180, 48, 72, 40


def _steps(points: list[Point], series: int, x, y, lo: float, hi: float) -> str:
    """A path where point j's value holds on (t_{j-1}, t_j]: the gate refuses s < t."""
    d, prev, open_ = [], lo, False
    for t, *values in points:
        v, end = values[series], (hi if math.isinf(t) else t)
        if v is None:
            open_ = False
        else:
            d.append(f"{'L' if open_ else 'M'}{x(prev):.1f},{y(v):.1f}H{x(end):.1f}")
            open_ = True
        prev = end
    return "".join(d)


def refusal_chart(title: str, panels: list[tuple[str, list[Point]]], chosen: float | None) -> str:
    # The x-axis spans the observed scores (from the tenth below the lowest) up to 1, so
    # the curves fill the panel; everything left of the lowest score refuses nothing.
    finite = [t for _, pts in panels for t, *_ in pts if not math.isinf(t)]
    finite += [] if chosen is None else [chosen]
    lo = math.floor(min(finite, default=0.0) * 10) / 10
    hi = max([1.0, *finite])
    lo = min(lo, hi - 0.1)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"'
        f' role="img" aria-label="{escape(title)}"><style>{STYLE}</style>',
        f'<rect class="surface" width="{W}" height="{H}" rx="8"/>',
        f'<text class="ink" x="{LEFT}" y="24" font-weight="600">{escape(title)}</text>',
    ]
    for i, (name, cls) in enumerate([("precision", "p"), ("recall", "r")]):
        lx = LEFT + i * 110
        out.append(f'<path class="line {cls}" d="M{lx},44H{lx + 20}"/>')
        out.append(f'<text class="ink2" x="{lx + 26}" y="48">{name}</text>')

    def y(v):
        return TOP + (1 - v) * PANEL_H

    for k, (label, points) in enumerate(panels):
        x0 = LEFT + k * (PANEL_W + GAP)

        def x(t, x0=x0):
            return x0 + (t - lo) / (hi - lo) * PANEL_W

        out.append(f'<text class="ink2" x="{x0}" y="{TOP - 8}">{escape(label)}</text>')
        for v in (0, 0.5, 1):
            cls = "axis" if v == 0 else "grid"
            out.append(f'<path class="{cls}" d="M{x0},{y(v):.1f}H{x0 + PANEL_W}"/>')
            if k == 0:
                out.append(
                    f'<text class="muted" x="{x0 - 8}" y="{y(v) + 4:.1f}"'
                    f' text-anchor="end">{v:g}</text>'
                )
        for t in (lo, (lo + hi) / 2, hi):
            out.append(
                f'<text class="muted" x="{x(t):.1f}" y="{TOP + PANEL_H + 18}"'
                f' text-anchor="middle">{t:.2g}</text>'
            )
        for series, cls in ((0, "p"), (1, "r")):
            out.append(f'<path class="line {cls}" d="{_steps(points, series, x, y, lo, hi)}"/>')
        if chosen is not None:
            out.append(f'<path class="rule" d="M{x(chosen):.1f},{TOP}V{TOP + PANEL_H}"/>')
            # Below the y = 1 gridline (the halo would cut a line drawn there), and on the
            # rule's left when the label wouldn't fit inside the panel on its right.
            right = x(chosen) + 84 <= x0 + PANEL_W
            lx, anchor = (x(chosen) + 4, "start") if right else (x(chosen) - 4, "end")
            value = f"{chosen:.12f}".rstrip("0").rstrip(".")  # every decimal it has
            out.append(
                f'<text class="ink2 halo" x="{lx:.1f}" y="{TOP + 18}" text-anchor="{anchor}">'
                f"t = {value}</text>"
            )
    out.append(
        f'<text class="muted" x="{LEFT + PANEL_W + GAP // 2}" y="{H - 12}" text-anchor="middle">'
        "threshold: refuse when the top hit's score is below it</text></svg>"
    )
    return "\n".join(out) + "\n"
