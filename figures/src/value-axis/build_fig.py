#!/usr/bin/env python3
"""Build the value-axis figure fragment: pure build-time SVG, no runtime JS.

    python3 figures/src/value-axis/build_fig.py

Four panels, one per architecture, all the same plot: the measured decision
threshold (steps) against the 0.5 goal's value, for agents fine-tuned at that
value (ink) and for the frozen base agent with the fitted axis written into its
weights (violet). Shading is the range over independent fine-tune draws where
more than one exists, and the decode's 95% interval otherwise. One dashed
reference in every panel: the task's own exchange rate, threshold =
(1.0 − v)/0.05 — a property of the maze, identical everywhere, so deviations
from it read as each architecture's bias.

`data.json['panels']`: [{key, title, seeds: [{name, trained: [{v, steps,
lo?, hi?}], written: [...]}]}]. `scurves` stays in the file for the separate
psychometric mini-figure.

Output: ../../fragments/value-axis.html — splice with <!-- fig: value-axis -->.
"""

import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
DATA = json.loads((SRC / "data.json").read_text())

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

TRAINED = "var(--ink)"
WRITTEN = "#6f5f9c"  # violet: the post-wide colour for reaching into the network


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
    return (
        f'<text x="{x}" y="{y}" font-size="13" font-weight="600" fill="var(--ink)" '
        f'font-family="{SANS}">{text}</text>'
    )


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
            out.append(
                f'<line x1="{self.x-4}" y1="{py}" x2="{self.x}" y2="{py}" stroke="var(--ink-soft)" stroke-width="1"/>'
            )
            if ylabels:
                out.append(
                    f'<text x="{self.x-8}" y="{py+3.5}" text-anchor="end" font-size="10" '
                    f'fill="var(--ink-soft)" font-family="{MONO}">{t:g}</text>'
                )
        return "".join(out)


def series_stats(seeds, kind, x="dv"):
    """Per value: mean over seeds, and the band (seed range, or CI if one seed)."""
    values = sorted({row[x] for seed in seeds for row in seed[kind]})
    stats = []
    for v in values:
        rows = [row for seed in seeds for row in seed[kind] if row[x] == v]
        steps = [row["steps"] for row in rows]
        if len(rows) > 1:
            band = (min(steps), max(steps))
        elif "lo" in rows[0]:
            band = (rows[0]["lo"], rows[0]["hi"])
        else:
            band = (steps[0], steps[0])
        stats.append((v, sum(steps) / len(steps), band))
    return stats


def marker(chart, v, m, colour, filled, square):
    """A point on a series. Square = an arm the axis was never fitted from."""
    x, y = chart.px(v), chart.py(chart.clip(m))
    if square:
        # sized to sit at the dots' visual weight rather than their radius
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
        " ".join(f"{'M' if i == 0 else 'L'}{chart.px(v)},{chart.py(chart.clip(hi))}" for i, (v, _, (lo, hi)) in enumerate(stats))
        + " "
        + " ".join(f"L{chart.px(v)},{chart.py(chart.clip(lo))}" for v, _, (lo, hi) in reversed(stats))
        + " Z"
    )
    out.append(f'<path d="{band_path}" fill="{colour}" opacity="0.13" stroke="none"/>')
    line = " ".join(f"{'M' if i == 0 else 'L'}{chart.px(v)},{chart.py(chart.clip(m))}" for i, (v, m, _) in enumerate(stats))
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out.append(f'<path d="{line}" fill="none" stroke="{colour}" stroke-width="1.8"{dash}/>')
    for v, m, _ in stats:
        square = square_beyond is not None and abs(v) > square_beyond
        out.append(marker(chart, v, m, colour, filled, square))
    return "".join(out)


def build(sweeps=("o0",), name="value-axis"):
    """Main figure carries the 1.0-goal sweep alone; the two-sweep version
    (sweeps=("o0", "o1")) is kept as value-axis-both.html for the appendix."""
    width = 790
    chart_w, chart_h = 300, 185
    col_x = (74, 452)
    row_y = (46, 300)
    xr, yr = (-0.75, 0.75), (-10, 26)
    RICH, POOR = "var(--rich)", "var(--poor)"
    sweep_colours = [(s, {"o0": RICH, "o1": POOR}[s]) for s in sweeps]

    svg = []
    letters = "abcd"
    panels = DATA["xpanels"]
    for i, panel in enumerate(panels):
        cx, cy = col_x[i % 2], row_y[i // 2]
        chart = Chart(cx, cy, chart_w, chart_h, xr, yr)
        bottom = i // 2 == 1
        left = i % 2 == 0
        svg.append(panel_letter(cx - 46 if left else cx - 40, cy - 12, letters[i]))
        svg.append(subtitle(cx - 20 if left else cx - 14, cy - 12, panel["title"]))
        svg.append(chart.frame((-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6), (-10, -5, 0, 5, 10, 15, 20, 25), xlabels=bottom, ylabels=left))

        # preference reversal. Above the line the agent still crosses the maze
        # for the richer goal; below it, it walks past the goal it used to
        # prefer. Drawn under the data because what matters is which series
        # reach it: the fine-tuned arms do, none of the written ones do.
        svg.append(
            f'<line x1="{chart.x}" y1="{chart.py(0)}" x2="{chart.x + chart.w}" y2="{chart.py(0)}" '
            f'stroke="var(--ink-soft)" stroke-width="1" opacity="0.55"/>'
        )

        # the utility rule's X — the task's optimum, deliberately identical in
        # every panel so each agent's offset from it reads as that agent's bias
        y0 = (1.0 - 0.5) / DATA["step_penalty"]
        slope = 1.0 / DATA["step_penalty"]
        # +1 is the o0 branch (raising the 1.0 goal raises the threshold),
        # -1 the o1 branch; only draw the branches whose sweep is shown
        for sign in tuple({"o0": +1, "o1": -1}[s] for s in sweeps):
            pts = [(-0.72, y0 - sign * slope * 0.72), (0.72, y0 + sign * slope * 0.72)]
            svg.append(
                '<path d="'
                + " ".join(f"{'M' if j == 0 else 'L'}{chart.px(v)},{chart.py(chart.clip(y))}" for j, (v, y) in enumerate(pts))
                + '" fill="none" stroke="var(--line)" stroke-width="1.4" stroke-dasharray="5 4"/>'
            )

        for sweep, colour in sweep_colours:
            seeds = panel["sweeps"][sweep]
            trained = series_stats(seeds, "trained")
            written = series_stats(seeds, "written")
            svg.append(draw_series(chart, trained, colour, filled=True, dashed=False, square_beyond=DATA["fit_window"]))
            svg.append(draw_series(chart, written, colour, filled=False, dashed=True))

    xlabel = "change in the 1.0 goal's value" if sweeps == ("o0",) else "change in the swept goal's value"
    svg.append(label(width / 2, row_y[1] + chart_h + 34, xlabel))

    # one figure-level legend row, below everything: colour = which goal's
    # value is swept, line style = how the shifted agent was produced
    ly = row_y[1] + chart_h + 56
    # with a single sweep the colour no longer encodes anything, so the style
    # swatches carry it directly and the colour-key entries disappear
    style_colour = sweep_colours[0][1] if len(sweeps) == 1 else "var(--ink)"
    items = [
        (RICH, False, False, "1.0 goal&apos;s value moved"),
        (POOR, False, False, "0.5 goal&apos;s value moved"),
    ][: 0 if len(sweeps) == 1 else 2] + [
        (style_colour, False, False, "fine-tuned"),
        (style_colour, False, True, "held out of the axis fit"),
        (style_colour, True, False, "written, \u03b8 + \u03b1\u00b7v\u0302"),
    ]
    est = lambda t: 5.4 * len(t.replace("&apos;", "'"))
    gap = 24
    lx = (width - (sum(22 + est(t) for _, _, _, t in items) + gap * (len(items) - 1))) / 2
    for colour, dashed, square, text in items:
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+16}" y2="{ly}" stroke="{colour}" stroke-width="1.8"{dash}/>')
        if square:
            svg.append(f'<rect x="{lx+5.6}" y="{ly-2.4}" width="4.8" height="4.8" fill="{colour}"/>')
        elif dashed:
            svg.append(f'<circle cx="{lx+8}" cy="{ly}" r="2.6" fill="var(--panel)" stroke="{colour}" stroke-width="1.5"/>')
        else:
            svg.append(f'<circle cx="{lx+8}" cy="{ly}" r="2.6" fill="{colour}"/>')
        svg.append(f'<text x="{lx+22}" y="{ly+3.5}" font-size="10.5" fill="var(--ink-soft)" font-family="{SANS}">{text}</text>')
        lx += 22 + est(text) + gap
    svg.append(
        f'<text x="26" y="{(row_y[0]+row_y[1]+chart_h)/2}" text-anchor="middle" font-size="10.5" '
        f'letter-spacing="0.11em" fill="var(--ink-soft)" font-family="{SANS}" '
        f'transform="rotate(-90 26 {(row_y[0]+row_y[1]+chart_h)/2})">THRESHOLD τ (STEPS)</text>'
    )

    height = row_y[1] + chart_h + 72


    # Caption lives in the post markdown (::: {.figcap for=value-axis}); the empty
    # figcaption below is the splice anchor.

    fig_id = "valueaxis-fig" if name == "value-axis" else "valueaxis-both-fig"
    aria = (
        "Threshold against the 1.0 goal's value" if sweeps == ("o0",) else "Threshold against either goal's value"
    ) + " for four architectures: fine-tuned agents and axis-written agents coincide along the task's exchange rate."
    fragment = f"""<figure class="fig-valueaxis wide" id="{fig_id}">
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


if __name__ == "__main__":
    build()                                          # main: 1.0-goal sweep only
    build(("o0", "o1"), name="value-axis-both")      # appendix: both sweeps
