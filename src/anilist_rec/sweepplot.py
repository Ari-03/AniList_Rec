"""Static SVG for the §5 dial-sweep curve: NDCG@10 vs popularity lift.

A connected scatterplot per finalist — each point one dial setting, dial
increasing right-to-left as the re-rank trades popularity lift away. Committed
under reports/ and embedded from eval.md, so it must be self-contained: own
surface, no scripts, palette baked in (validated slots 1-3 of the reference
palette, light mode).
"""

import math

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots, fixed order
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif"'

W, H = 720, 440
ML, MR, MT, MB = 64, 150, 52, 52  # right margin fits direct series labels


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo or 1.0
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step / 2:
        if t >= lo - step / 2:
            ticks.append(round(t, 10))
        t += step
    return ticks


def render_sweep_svg(curves: dict[str, dict[float, dict]]) -> str:
    pts = [
        (s["pop_lift"], s["ndcg10"], d, name)
        for name, sweep in curves.items()
        for d, s in sweep.items()
    ]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xpad = (max(xs) - min(xs) or 1.0) * 0.08
    ypad = (max(ys) - min(ys) or 0.02) * 0.10
    x0, x1 = min(xs) - xpad, max(xs) + xpad
    y0, y1 = min(ys) - ypad, max(ys) + ypad

    def sx(v: float) -> float:
        return ML + (v - x0) / (x1 - x0) * (W - ML - MR)

    def sy(v: float) -> float:
        return H - MB - (v - y0) / (y1 - y0) * (H - MT - MB)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        'aria-label="Validation NDCG at 10 versus popularity lift across dial settings">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="{ML}" y="24" {FONT} font-size="15" font-weight="600" fill="{INK}">'
        "Dial sweep — validation NDCG@10 vs popularity lift</text>",
        f'<text x="{ML}" y="41" {FONT} font-size="12" fill="{INK_2}">'
        "Each point is one dial setting; dial rises as popularity lift falls</text>",
    ]

    for t in _nice_ticks(y0, y1):
        y = sy(t)
        parts += [
            f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>',
            f'<text x="{ML - 8}" y="{y + 4:.1f}" {FONT} font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{t:.3f}</text>',
        ]
    for t in _nice_ticks(x0, x1):
        x = sx(t)
        parts.append(
            f'<text x="{x:.1f}" y="{H - MB + 18}" {FONT} font-size="11" fill="{MUTED}" '
            f'text-anchor="middle">{t:+g}</text>'
        )
    parts += [
        f'<line x1="{ML}" y1="{H - MB}" x2="{W - MR}" y2="{H - MB}" '
        f'stroke="{BASELINE}" stroke-width="1"/>',
        f'<text x="{(ML + W - MR) / 2:.0f}" y="{H - 14}" {FONT} font-size="12" '
        f'fill="{INK_2}" text-anchor="middle">popularity lift '
        "(top-10 percentile minus profile percentile)</text>",
        f'<text x="18" y="{(MT + H - MB) / 2:.0f}" {FONT} font-size="12" fill="{INK_2}" '
        f'text-anchor="middle" transform="rotate(-90 18 {(MT + H - MB) / 2:.0f})">'
        "validation NDCG@10</text>",
    ]

    for i, (name, sweep) in enumerate(curves.items()):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        dials = sorted(sweep)
        line = " ".join(
            f"{sx(sweep[d]['pop_lift']):.1f},{sy(sweep[d]['ndcg10']):.1f}" for d in dials
        )
        parts.append(
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
            'stroke-linejoin="round"/>'
        )
        labeled = {dials[0], dials[-1], dials[len(dials) // 2]}
        for d in dials:
            x, y = sx(sweep[d]["pop_lift"]), sy(sweep[d]["ndcg10"])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
            if d in labeled:  # selective, never every point
                parts.append(
                    f'<text x="{x:.1f}" y="{y - 9:.1f}" {FONT} font-size="10.5" '
                    f'fill="{INK_2}" text-anchor="middle">dial {d:g}</text>'
                )
        # direct label at the dial-off end (rightmost: dial 0 has the highest lift)
        end = sweep[dials[0]]
        parts.append(
            f'<text x="{sx(end["pop_lift"]) + 10:.1f}" y="{sy(end["ndcg10"]) + 4:.1f}" '
            f'{FONT} font-size="12" font-weight="600" fill="{color}">{name}</text>'
        )

    # legend (always present for >= 2 series)
    for i, name in enumerate(curves):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        y = MT + 8 + i * 20
        parts += [
            f'<circle cx="{W - MR + 18}" cy="{y}" r="4" fill="{color}"/>',
            f'<text x="{W - MR + 28}" y="{y + 4}" {FONT} font-size="12" '
            f'fill="{INK}">{name}</text>',
        ]

    parts.append("</svg>")
    return "\n".join(parts)
