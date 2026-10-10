#!/usr/bin/env python3
"""Build the arm-length appendix figure: pure build-time SVG, no runtime JS.

    python3 figures/src/arm-length/build_fig.py

Three panels, every quantity read down a ladder of fine-tune budgets from 100k
to 3M steps on the three DRC bases (line = mean, band = range across bases):

  (a) Table 1's decomposition of a fine-tune's weight change at |dv| = 0.3:
      the drift shared by every arm, the write along the fitted axis, and the
      remainder. Drift grows at the expense of both others.
  (b) What the arms learned: the fine-tuned threshold's slope in value, against
      the task's 20 steps per unit.
  (c) The reversal test: the threshold of an arm trained past parity (colour 1
      at 1.2, colour 0 at 0.3) and of the base with the same dose written along
      the axis. Trained arms cross zero after about 1M steps; the write does not.

`data.json` comes from scripts/studies/arm-length/arm_length_report.py --out.
Output: ../../fragments/arm-length.html - splice with <!-- fig: arm-length -->.
"""

import json
import math
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
DATA = json.loads((SRC / "data.json").read_text())

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

INK = "var(--ink)"
SOFT = "var(--ink-soft)"
RICH = "var(--rich)"  # the 1.0 goal, colour 0
POOR = "var(--poor)"  # the 0.5 goal, colour 1
WRITE = "var(--write)"  # decoded internals


def label(x, y, text, anchor="middle", size=10.5):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" letter-spacing="0.11em" '
        f'fill="{SOFT}" font-family="{SANS}">{text.upper()}</text>'
    )


def panel_letter(x, y, letter):
    return f'<text x="{x}" y="{y}" font-size="13" font-style="italic" fill="{INK}" font-family="{SERIF}">({letter})</text>'


def subtitle(x, y, text):
    return f'<text x="{x}" y="{y}" font-size="13" font-weight="600" fill="{INK}" font-family="{SANS}">{text}</text>'


def subtitle_u(x, y, title, unit):
    """Panel header as one line: bold title, soft y-unit."""
    return (
        f'<text x="{x}" y="{y}" font-family="{SANS}"><tspan font-size="13" font-weight="600" fill="{INK}">{title}</tspan>'
        f'<tspan font-size="10" fill="{SOFT}" dx="7">{unit}</tspan></text>'
    )


class Chart:
    def __init__(self, x, y, w, h, xr, yr):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xr, self.yr = xr, yr

    def px(self, steps):  # log axis in steps
        v = math.log10(steps)
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, vy):
        return self.y + self.h - (vy - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def frame(self, xticks, yticks, yfmt=lambda t: f"{t:g}"):
        out = [
            f'<line x1="{self.x}" y1="{self.y+self.h}" x2="{self.x+self.w}" y2="{self.y+self.h}" stroke="{SOFT}" stroke-width="1"/>',
            f'<line x1="{self.x}" y1="{self.y}" x2="{self.x}" y2="{self.y+self.h}" stroke="{SOFT}" stroke-width="1"/>',
        ]
        for steps, text in xticks:
            px = self.px(steps)
            out.append(
                f'<line x1="{px}" y1="{self.y+self.h}" x2="{px}" y2="{self.y+self.h+4}" stroke="{SOFT}" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{px}" y="{self.y+self.h+16}" text-anchor="middle" font-size="10" fill="{SOFT}" font-family="{MONO}">{text}</text>'
            )
        for t in yticks:
            py = self.py(t)
            out.append(f'<line x1="{self.x-4}" y1="{py}" x2="{self.x}" y2="{py}" stroke="{SOFT}" stroke-width="1"/>')
            out.append(
                f'<text x="{self.x-8}" y="{py+3.5}" text-anchor="end" font-size="10" fill="{SOFT}" font-family="{MONO}">{yfmt(t)}</text>'
            )
        return "".join(out)

    def hline(self, vy, colour, dash="5 4", width=1.2):
        return (
            f'<line x1="{self.x}" y1="{self.py(vy)}" x2="{self.x+self.w}" y2="{self.py(vy)}" '
            f'stroke="{colour}" stroke-width="{width}" stroke-dasharray="{dash}"/>'
        )


def series(rows, key, scale=1.0):
    """Mean and range across bases at each budget, as [(steps, mean, lo, hi)] in step order."""
    by_step = {}
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_step.setdefault(r["read_at"], []).append(v * scale)
    return [(s, sum(v) / len(v), min(v), max(v)) for s, v in sorted(by_step.items())]


def draw_series(chart, points, colour, filled=True, dashed=False, band=True, r=2.4):
    out = []
    if band and len(points) > 1:
        path = (
            " ".join(f"{'M' if i == 0 else 'L'}{chart.px(s)},{chart.py(hi)}" for i, (s, _, _, hi) in enumerate(points))
            + " "
            + " ".join(f"L{chart.px(s)},{chart.py(lo)}" for s, _, lo, _ in reversed(points))
            + " Z"
        )
        out.append(f'<path d="{path}" fill="{colour}" opacity="0.13" stroke="none"/>')
    line = " ".join(f"{'M' if i == 0 else 'L'}{chart.px(s)},{chart.py(m)}" for i, (s, m, _, _) in enumerate(points))
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out.append(f'<path d="{line}" fill="none" stroke="{colour}" stroke-width="1.8"{dash}/>')
    for s, m, _, _ in points:
        if filled:
            out.append(f'<circle cx="{chart.px(s)}" cy="{chart.py(m)}" r="{r}" fill="{colour}"/>')
        else:
            out.append(
                f'<circle cx="{chart.px(s)}" cy="{chart.py(m)}" r="{r}" fill="var(--panel)" stroke="{colour}" stroke-width="1.4"/>'
            )
    return "".join(out)


def legend(svg, centre, y, items):
    est = lambda t: 5.4 * len(t)
    lx = centre - (sum(22 + est(t) for *_, t in items) + 26 * (len(items) - 1)) / 2
    for colour, dashed, filled, text in items:
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx+16}" y2="{y}" stroke="{colour}" stroke-width="1.8"{dash}/>')
        if filled is True:
            svg.append(f'<circle cx="{lx+8}" cy="{y}" r="2.2" fill="{colour}"/>')
        elif filled is False:
            svg.append(f'<circle cx="{lx+8}" cy="{y}" r="2.2" fill="var(--panel)" stroke="{colour}" stroke-width="1.4"/>')
        svg.append(f'<text x="{lx+22}" y="{y+3.5}" font-size="10.5" fill="{SOFT}" font-family="{SANS}">{text}</text>')
        lx += 22 + est(text) + 26


def build():
    rows = DATA["rows"]
    width = 790
    chart_w, chart_h = 200, 170
    col_x = (56, 318, 580)
    top = 44
    xr = (math.log10(80_000), math.log10(3_400_000))
    xticks = ((100_000, "100k"), (400_000, "400k"), (1_000_000, "1M"), (3_000_000, "3M"))

    svg = []

    # (a) shares
    a = Chart(col_x[0], top, chart_w, chart_h, xr, (0, 60))
    svg.append(panel_letter(col_x[0] - 30, top - 12, "a"))
    svg.append(subtitle_u(col_x[0] - 6, top - 12, "Weight-change shares", "% of |Δθ|²"))
    svg.append(a.frame(xticks, (0, 20, 40, 60), yfmt=lambda t: f"{t:g}%"))
    for key, colour, dashed in (("remainder", INK, True), ("drift", SOFT, False), ("write", WRITE, False)):
        svg.append(draw_series(a, series(rows, key, 100.0), colour, filled=True, dashed=dashed))
    for key, text, dy in (("drift", "drift", -6), ("write", "write", 6), ("remainder", "residual", -6)):
        s, m, _, _ = series(rows, key, 100.0)[-1]
        svg.append(
            f'<text x="{a.px(s)+5}" y="{a.py(m)+dy+3}" font-size="10" fill="{ {"drift": SOFT, "write": WRITE, "remainder": INK}[key] }" font-family="{SANS}">{text}</text>'
        )

    # (b) fine-tuned slope
    b = Chart(col_x[1], top, chart_w, chart_h, xr, (0, 22))
    svg.append(panel_letter(col_x[1] - 30, top - 12, "b"))
    svg.append(subtitle_u(col_x[1] - 6, top - 12, "Learnt exchange rate", "steps per unit"))
    svg.append(b.frame(xticks, (0, 5, 10, 15, 20)))
    svg.append(b.hline(DATA["task_slope"], "var(--line)"))
    svg.append(draw_series(b, series(rows, "ft_slope"), INK, filled=True))

    # (c) reversal: trained past parity vs written
    c = Chart(col_x[2], top, chart_w, chart_h, xr, (-5, 3))
    svg.append(panel_letter(col_x[2] - 30, top - 12, "c"))
    svg.append(subtitle_u(col_x[2] - 6, top - 12, "Training at τ* = −4", "threshold, steps"))
    svg.append(c.frame(xticks, (-4, -2, 0, 2)))
    svg.append(c.hline(0, SOFT, dash="1.5 3", width=1))
    svg.append(c.hline(-4, "var(--line)"))
    for sweep, colour in (("k1", POOR), ("k0", RICH)):
        svg.append(draw_series(c, series(rows, f"written_{sweep}"), colour, filled=False, dashed=True))
        svg.append(draw_series(c, series(rows, f"tau_{sweep}"), colour, filled=True))

    svg.append(label(width / 2, top + chart_h + 34, "fine-tune length (steps, log scale)"))

    ly = top + chart_h + 56
    legend(
        svg,
        width / 2,
        ly,
        [
            (POOR, False, True, "0.5 goal raised to 1.2"),
            (RICH, False, True, "1.0 goal cut to 0.3"),
            ("var(--line)", True, None, "task optimum"),
        ],
    )
    height = ly + 16

    fragment = f"""<figure class="fig-arm-length wide" id="arm-length-fig">
<style>
  .fig-arm-length {{
    --rich: #a63d2f; --poor: #2f5a78; --write: #6f5f9c;
  }}
  .fig-arm-length .stage {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 22px; }}
  .fig-arm-length svg {{ display: block; width: 100%; height: auto; max-width: {width}px; margin: 0 auto; }}
  .fig-arm-length figcaption {{
    font-family: {SERIF}; font-size: 15.5px;
    color: var(--ink); max-width: 70ch; margin: 18px auto 0; line-height: 1.65;
    border-top: 1px solid var(--line); padding-top: 16px;
  }}
</style>
<div class="stage">
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Three panels against fine-tune length from 100k to 3M steps: the drift, write and residual shares of a fine-tune's weight change; the fine-tuned threshold's slope against the task's; and the threshold of arms trained past parity against the same dose written, which cross zero after about 750k steps while the write does not.">
{''.join(svg)}
</svg>
<figcaption></figcaption>
</div>
</figure>
"""
    out = SRC.parent.parent / "fragments" / "arm-length.html"
    out.write_text(fragment)
    print(f"wrote {out} ({len(fragment)//1024} KB)")


if __name__ == "__main__":
    build()
