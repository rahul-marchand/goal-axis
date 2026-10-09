#!/usr/bin/env python3
"""Build the appendix figure: the value axis carried far past the fitted window.

    python3 figures/src/value-axis-far/build_fig.py

Fig 3's four panels, read off the same measures, but the x-axis runs as far as
anything was measured rather than stopping where the story is decided. Two
extensions per sweep: fine-tuned arms trained further past preference parity
(filled squares -- held out of the axis fit, as in fig 3) and the axis written
at ever larger multiples (open circles, dashed). Where a written agent stops
finishing episodes the series is cut and a bar marks the cut; a threshold read
off the survivors past that point is noise, not a measurement.

Reads `data.json['farpanels']` from the value-axis source directory (built by
`parse_far.py`). Chart primitives are duplicated from the value-axis builder on
purpose: that file is hand-tuned and must not be edited for an appendix.

Output: ../../fragments/value-axis-far.html -- splice with <!-- fig: value-axis-far -->.
"""

import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
DATA = json.loads((SRC.parent / "value-axis" / "data.json").read_text())

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

YR = (-34, 26)
YTICKS = (-30, -20, -10, 0, 10, 20)


def x_range(panels):
    """Wide enough for every surviving point and every reach cut, the same in all panels."""
    reach = 0.0
    for panel in panels:
        for series in panel["sweeps"].values():
            for s in series:
                reach = max(reach, *(abs(r["dv"]) for r in s["trained"] + s["written"]))
                reach = max(reach, *(c for c in s["exhausted"].values() if c))
    edge = 0.5 * round(reach / 0.5 + 0.5) + 0.15
    return (-edge, edge)


def x_ticks(xr):
    step = 1.0 if xr[1] > 2.4 else 0.5
    n = int(xr[1] // step)
    return tuple(step * k for k in range(-n, n + 1))


def label(x, y, text, anchor="middle", size=10.5):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" letter-spacing="0.11em" '
        f'fill="var(--ink-soft)" font-family="{SANS}">{text.upper()}</text>'
    )


def panel_letter(x, y, letter):
    return (
        f'<text x="{x}" y="{y}" font-size="13" font-style="italic" fill="var(--ink)" '
        f'font-family="{SERIF}">({letter})</text>'
    )


def subtitle(x, y, text):
    return f'<text x="{x}" y="{y}" font-size="13" font-weight="600" fill="var(--ink)" ' f'font-family="{SANS}">{text}</text>'


class Chart:
    def __init__(self, x, y, w, h, xr, yr):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xr, self.yr = xr, yr

    def px(self, vx):
        return self.x + (vx - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, vy):
        return self.y + self.h - (vy - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def clip(self, vy):
        return max(self.yr[0], min(self.yr[1], vy))

    def frame(self, xticks, yticks, xlabels=True, ylabels=True):
        out = [
            f'<line x1="{self.x}" y1="{self.y+self.h}" x2="{self.x+self.w}" y2="{self.y+self.h}" '
            f'stroke="var(--ink-soft)" stroke-width="1"/>',
            f'<line x1="{self.x}" y1="{self.y}" x2="{self.x}" y2="{self.y+self.h}" '
            f'stroke="var(--ink-soft)" stroke-width="1"/>',
        ]
        for t in xticks:
            px = self.px(t)
            out.append(
                f'<line x1="{px}" y1="{self.y+self.h}" x2="{px}" y2="{self.y+self.h+4}" '
                f'stroke="var(--ink-soft)" stroke-width="1"/>'
            )
            if xlabels:
                out.append(
                    f'<text x="{px}" y="{self.y+self.h+16}" text-anchor="middle" font-size="10" '
                    f'fill="var(--ink-soft)" font-family="{MONO}">{t:g}</text>'
                )
        for t in yticks:
            py = self.py(t)
            out.append(f'<line x1="{self.x-4}" y1="{py}" x2="{self.x}" y2="{py}" stroke="var(--ink-soft)" stroke-width="1"/>')
            if ylabels:
                out.append(
                    f'<text x="{self.x-8}" y="{py+3.5}" text-anchor="end" font-size="10" '
                    f'fill="var(--ink-soft)" font-family="{MONO}">{t:g}</text>'
                )
        return "".join(out)


def series_stats(seeds, kind):
    """Per offset: mean over bases, and the band (base range, or the decode CI if one)."""
    values = sorted({row["dv"] for seed in seeds for row in seed[kind]})
    stats = []
    for v in values:
        rows = [row for seed in seeds for row in seed[kind] if row["dv"] == v]
        steps = [row["steps"] for row in rows]
        if len(rows) > 1:
            band = (min(steps), max(steps))
        elif "lo" in rows[0]:
            band = (rows[0]["lo"], rows[0]["hi"])
        else:
            band = (steps[0], steps[0])
        stats.append((v, sum(steps) / len(steps), band, len(rows)))
    return stats


def marker(chart, v, m, colour, filled, square):
    x, y = chart.px(v), chart.py(chart.clip(m))
    if square:
        return (
            f'<rect x="{x - 2.4}" y="{y - 2.4}" width="4.8" height="4.8" fill="{colour}"/>'
            if filled
            else f'<rect x="{x - 2.4}" y="{y - 2.4}" width="4.8" height="4.8" fill="var(--panel)" '
            f'stroke="{colour}" stroke-width="1.5"/>'
        )
    if filled:
        return f'<circle cx="{x}" cy="{y}" r="2.6" fill="{colour}"/>'
    return f'<circle cx="{x}" cy="{y}" r="2.6" fill="var(--panel)" stroke="{colour}" stroke-width="1.5"/>'


def draw_series(chart, stats, colour, filled, dashed, square_beyond=None):
    out = []
    band_path = (
        " ".join(
            f"{'M' if i == 0 else 'L'}{chart.px(v)},{chart.py(chart.clip(hi))}" for i, (v, _, (lo, hi), _) in enumerate(stats)
        )
        + " "
        + " ".join(f"L{chart.px(v)},{chart.py(chart.clip(lo))}" for v, _, (lo, hi), _ in reversed(stats))
        + " Z"
    )
    out.append(f'<path d="{band_path}" fill="{colour}" opacity="0.13" stroke="none"/>')
    line = " ".join(f"{'M' if i == 0 else 'L'}{chart.px(v)},{chart.py(chart.clip(m))}" for i, (v, m, _, _) in enumerate(stats))
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out.append(f'<path d="{line}" fill="none" stroke="{colour}" stroke-width="1.8"{dash}/>')
    for v, m, _, _ in stats:
        square = square_beyond is not None and abs(v) > square_beyond
        out.append(marker(chart, v, m, colour, filled, square))
    return "".join(out)


def cut_marker(chart, dv, steps, colour):
    """Where a written agent stopped finishing episodes: a cross at the first failing offset."""
    x, y = chart.px(dv), chart.py(chart.clip(steps))
    return (
        f'<line x1="{x-3.2}" y1="{y-3.2}" x2="{x+3.2}" y2="{y+3.2}" stroke="{colour}" stroke-width="1.6"/>'
        f'<line x1="{x-3.2}" y1="{y+3.2}" x2="{x+3.2}" y2="{y-3.2}" stroke="{colour}" stroke-width="1.6"/>'
    )


def build(name="value-axis-far"):
    width = 790
    chart_w, chart_h = 300, 200
    col_x = (74, 452)
    row_y = (46, 316)
    RICH, POOR = "var(--rich)", "var(--poor)"
    sweep_colours = (("o0", RICH), ("o1", POOR))
    window = DATA["fit_window"]
    panels = DATA["farpanels"]
    XR = x_range(panels)
    XTICKS = x_ticks(XR)

    svg = []
    letters = "abcd"
    for i, panel in enumerate(panels):
        cx, cy = col_x[i % 2], row_y[i // 2]
        chart = Chart(cx, cy, chart_w, chart_h, XR, YR)
        bottom = i // 2 == 1
        left = i % 2 == 0
        svg.append(panel_letter(cx - 46 if left else cx - 40, cy - 12, letters[i]))
        svg.append(subtitle(cx - 20 if left else cx - 14, cy - 12, panel["title"]))

        # the offsets the axis was fitted from; everything outside is extrapolation
        svg.append(
            f'<rect x="{chart.px(-window)}" y="{chart.y}" width="{chart.px(window) - chart.px(-window)}" '
            f'height="{chart.h}" fill="var(--ink)" opacity="0.045"/>'
        )
        svg.append(chart.frame(XTICKS, YTICKS, xlabels=bottom, ylabels=left))
        # preference reversal
        svg.append(
            f'<line x1="{chart.x}" y1="{chart.py(0)}" x2="{chart.x + chart.w}" y2="{chart.py(0)}" '
            f'stroke="var(--ink-soft)" stroke-width="1" opacity="0.55"/>'
        )
        # the task's own exchange rate, identical in every panel
        y0 = (1.0 - 0.5) / DATA["step_penalty"]
        slope = 1.0 / DATA["step_penalty"]
        for sign in (+1, -1):
            # only the reachable stretch: clip at the y-range so the line does not
            # run off the frame carrying nothing
            pts = []
            for v in (XR[0] + 0.03, XR[1] - 0.03):
                pts.append((v, y0 + sign * slope * v))
            # clip to the frame by walking the segment
            (v1, s1), (v2, s2) = pts

            def at_y(target):
                return v1 + (target - s1) * (v2 - v1) / (s2 - s1)

            clipped = []
            for v, s in pts:
                if YR[0] <= s <= YR[1]:
                    clipped.append((v, s))
            for target in (YR[0], YR[1]):
                v = at_y(target)
                if XR[0] < v < XR[1]:
                    clipped.append((v, target))
            clipped = sorted(clipped)[:2] if len(clipped) > 2 else sorted(clipped)
            if len(clipped) == 2:
                svg.append(
                    '<path d="'
                    + " ".join(f"{'M' if j == 0 else 'L'}{chart.px(v)},{chart.py(s)}" for j, (v, s) in enumerate(clipped))
                    + '" fill="none" stroke="var(--line)" stroke-width="1.4" stroke-dasharray="5 4"/>'
                )

        for sweep, colour in sweep_colours:
            seeds = panel["sweeps"][sweep]
            if not seeds:
                continue
            trained = series_stats(seeds, "trained")
            written = series_stats(seeds, "written")
            svg.append(draw_series(chart, trained, colour, filled=True, dashed=False, square_beyond=window))
            svg.append(draw_series(chart, written, colour, filled=False, dashed=True))
            # the reach cut: drawn once per sweep, only where every base has
            # stopped finishing, at the furthest offset any of them survived to
            cuts = [s["exhausted"]["written"] for s in seeds]
            if cuts and all(cuts):
                sign = -1.0 if sweep == "o0" else 1.0
                cut = sign * max(cuts)
                last = max((v for v, *_ in written if v * sign > 0), key=abs, default=None)
                if last is not None:
                    height = next(m for v, m, _, _ in written if v == last)
                    svg.append(cut_marker(chart, cut, height, colour))

    svg.append(label(width / 2, row_y[1] + chart_h + 34, "change in the swept goal's value"))

    # two legend rows: what the colour means, then what the glyphs mean
    rows = [
        [(RICH, False, "circle", "1.0 goal&apos;s value moved"), (POOR, False, "circle", "0.5 goal&apos;s value moved")],
        [
            ("var(--ink)", False, "circle", "fine-tuned"),
            ("var(--ink)", False, "square", "held out of the axis fit"),
            ("var(--ink)", True, "open", "written, θ + α·v̂"),
            ("var(--ink)", False, "cut", "write stops finishing episodes"),
        ],
    ]
    est = lambda t: 5.4 * len(t.replace("&apos;", "'"))
    gap = 24
    for r, items in enumerate(rows):
        ly = row_y[1] + chart_h + 56 + 18 * r
        lx = (width - (sum(22 + est(t) for _, _, _, t in items) + gap * (len(items) - 1))) / 2
        for colour, dashed, glyph, text in items:
            dash = ' stroke-dasharray="4 3"' if dashed else ""
            if glyph == "cut":
                svg.append(
                    f'<line x1="{lx+4.8}" y1="{ly-3.2}" x2="{lx+11.2}" y2="{ly+3.2}" stroke="{colour}" stroke-width="1.6"/>'
                    f'<line x1="{lx+4.8}" y1="{ly+3.2}" x2="{lx+11.2}" y2="{ly-3.2}" stroke="{colour}" stroke-width="1.6"/>'
                )
            else:
                svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+16}" y2="{ly}" stroke="{colour}" stroke-width="1.8"{dash}/>')
                if glyph == "square":
                    svg.append(f'<rect x="{lx+5.6}" y="{ly-2.4}" width="4.8" height="4.8" fill="{colour}"/>')
                elif glyph == "open":
                    svg.append(
                        f'<circle cx="{lx+8}" cy="{ly}" r="2.6" fill="var(--panel)" stroke="{colour}" stroke-width="1.5"/>'
                    )
                else:
                    svg.append(f'<circle cx="{lx+8}" cy="{ly}" r="2.6" fill="{colour}"/>')
            svg.append(
                f'<text x="{lx+22}" y="{ly+3.5}" font-size="10.5" fill="var(--ink-soft)" font-family="{SANS}">{text}</text>'
            )
            lx += 22 + est(text) + gap
    svg.append(
        f'<text x="26" y="{(row_y[0]+row_y[1]+chart_h)/2}" text-anchor="middle" font-size="10.5" '
        f'letter-spacing="0.11em" fill="var(--ink-soft)" font-family="{SANS}" '
        f'transform="rotate(-90 26 {(row_y[0]+row_y[1]+chart_h)/2})">THRESHOLD τ (STEPS)</text>'
    )
    height = row_y[1] + chart_h + 90

    aria = (
        "Threshold against either goal's value, carried far past the fitted window, for four "
        "architectures: fine-tuned arms cross the preference reversal, written agents flatten or stop finishing."
    )
    fragment = f"""<figure class="fig-valueaxis wide" id="valueaxis-far-fig">
<style>
  .fig-valueaxis {{
    --wall: #2e313b; --floor: #eeebe3;
    --rich: #a63d2f; --poor: #2f5a78;
  }}
  .fig-valueaxis .stage {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 22px; }}
  .fig-valueaxis svg {{ display: block; width: 100%; height: auto; max-width: {width}px; margin: 0 auto; }}
  .fig-valueaxis figcaption {{
    font-family: {SERIF}; font-size: 15.5px;
    color: var(--ink); max-width: 70ch; margin: 18px auto 0; line-height: 1.65;
    border-top: 1px solid var(--line); padding-top: 16px;
  }}
</style>
<div class="stage">
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{aria}">
{''.join(svg)}
</svg>
<figcaption></figcaption>
</div>
</figure>
"""
    out = SRC.parent.parent / "fragments" / f"{name}.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(fragment)
    print(f"wrote {out} ({len(fragment)//1024} KB)")
    return fragment


if __name__ == "__main__":
    build()
