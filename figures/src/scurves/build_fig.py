#!/usr/bin/env python3
"""Build the s-curve figure fragment: pure build-time SVG, no runtime JS.

    python3 figures/src/scurves/build_fig.py

Four panels, one per architecture, mirroring the value-axis figure's grid.
Each panel is the raw psychometric curve behind that figure's thresholds:
P(took the 1.0 goal) against the distance gap, for the base agent (ink,
solid) and for the same agent with the value axis written into its weights
at alpha = -0.2 and +0.2 (slate, dashed, open) - the write slides the whole
curve sideways. Every threshold in the value-axis figure is the 0.5-crossing
of a curve like these.

`data.json` comes from figures/extract_scurves.py. Panels flagged `pending`
(ResNet, ViT - no per-gap decodes yet) render as an empty frame with a note
and fill in when the gamma=1 rebuild's decodes land.

Output: ../../fragments/scurves.html - splice with <!-- fig: scurves -->.
"""

import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
DATA = json.loads((SRC / "data.json").read_text())

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

BASE = "var(--ink)"
WRITTEN = "var(--poor)"  # the 0.5 goal's colour: both writes move along its value axis


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


def draw_curve(chart, bins, colour, filled, dashed):
    out = []
    band = (
        " ".join(f"{'M' if i == 0 else 'L'}{chart.px(b['gap'])},{chart.py(b['hi'])}" for i, b in enumerate(bins))
        + " "
        + " ".join(f"L{chart.px(b['gap'])},{chart.py(b['lo'])}" for b in reversed(bins))
        + " Z"
    )
    out.append(f'<path d="{band}" fill="{colour}" opacity="0.13" stroke="none"/>')
    line = " ".join(f"{'M' if i == 0 else 'L'}{chart.px(b['gap'])},{chart.py(b['p'])}" for i, b in enumerate(bins))
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out.append(f'<path d="{line}" fill="none" stroke="{colour}" stroke-width="1.8"{dash}/>')
    for b in bins:
        if filled:
            out.append(f'<circle cx="{chart.px(b["gap"])}" cy="{chart.py(b["p"])}" r="2.2" fill="{colour}"/>')
        else:
            out.append(
                f'<circle cx="{chart.px(b["gap"])}" cy="{chart.py(b["p"])}" r="2.2" fill="var(--panel)" '
                f'stroke="{colour}" stroke-width="1.4"/>'
            )
    return "".join(out)


def build():
    width = 790
    chart_w, chart_h = 300, 185
    col_x = (74, 452)
    row_y = (46, 300)
    xr, yr = (-7, 22), (-0.03, 1.03)

    svg = []
    letters = "abcd"
    for i, panel in enumerate(DATA["panels"]):
        cx, cy = col_x[i % 2], row_y[i // 2]
        chart = Chart(cx, cy, chart_w, chart_h, xr, yr)
        bottom = i // 2 == 1
        left = i % 2 == 0
        svg.append(panel_letter(cx - 46 if left else cx - 40, cy - 12, letters[i]))
        svg.append(subtitle(cx - 20 if left else cx - 14, cy - 12, panel["title"]))
        svg.append(chart.frame((-5, 0, 5, 10, 15, 20), (0, 0.5, 1), xlabels=bottom, ylabels=left))

        # p = 0.5: the line every threshold is read off
        svg.append(
            f'<line x1="{chart.px(xr[0])}" y1="{chart.py(0.5)}" x2="{chart.px(xr[1])}" y2="{chart.py(0.5)}" '
            f'stroke="var(--line)" stroke-width="1" stroke-dasharray="1.5 3"/>'
        )
        # the task's optimal crossing, identical in every panel
        opt = chart.px(DATA["optimum_gap"])
        svg.append(
            f'<line x1="{opt}" y1="{chart.py(yr[1])}" x2="{opt}" y2="{chart.py(yr[0])}" '
            f'stroke="var(--line)" stroke-width="1.4" stroke-dasharray="5 4"/>'
        )

        if panel.get("pending"):
            mx, my = cx + chart_w / 2, cy + chart_h / 2
            svg.append(label(mx, my - 4, "decode pending"))
            svg.append(label(mx, my + 14, "lands with the γ = 1 rebuild", size=9.5))
            continue

        curves = {c["alpha"]: c for c in panel["curves"]}
        for alpha in (-0.2, 0.2, 0.0):  # base last, on top
            c = curves[alpha]
            is_base = alpha == 0.0
            colour = BASE if is_base else WRITTEN
            svg.append(draw_curve(chart, c["bins"], colour, filled=is_base, dashed=not is_base))

    svg.append(label(width / 2, row_y[1] + chart_h + 34, "distance gap to the 1.0 goal (steps)"))
    svg.append(
        f'<text x="26" y="{(row_y[0]+row_y[1]+chart_h)/2}" text-anchor="middle" font-size="10.5" '
        f'letter-spacing="0.11em" fill="var(--ink-soft)" font-family="{SANS}" '
        f'transform="rotate(-90 26 {(row_y[0]+row_y[1]+chart_h)/2})">P(TOOK THE 1.0 GOAL)</text>'
    )

    ly = row_y[1] + chart_h + 56
    items = [
        (BASE, False, True, "base agent"),
        (WRITTEN, True, False, "axis written, θ + α·v̂ (α = ±0.2)"),
        ("var(--line)", True, None, "task optimum, gap = 10"),
    ]
    est = lambda t: 5.4 * len(t)
    lx = (width - (sum(22 + est(t) for *_, t in items) + 30 * (len(items) - 1))) / 2
    for colour, dashed, filled, text in items:
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+16}" y2="{ly}" stroke="{colour}" stroke-width="1.8"{dash}/>')
        if filled is True:
            svg.append(f'<circle cx="{lx+8}" cy="{ly}" r="2.2" fill="{colour}"/>')
        elif filled is False:
            svg.append(f'<circle cx="{lx+8}" cy="{ly}" r="2.2" fill="var(--panel)" stroke="{colour}" stroke-width="1.4"/>')
        svg.append(f'<text x="{lx+22}" y="{ly+3.5}" font-size="10.5" fill="var(--ink-soft)" font-family="{SANS}">{text}</text>')
        lx += 22 + est(text) + 30

    height = row_y[1] + chart_h + 72

    drc = {c["alpha"]: c for c in DATA["panels"][0]["curves"]}
    bc = {c["alpha"]: c for c in DATA["panels"][3]["curves"]}
    # Caption lives in the post markdown (::: {.figcap for=scurves}); the empty
    # figcaption below is the splice anchor.

    fragment = f"""<figure class="fig-scurves wide" id="scurves-fig">
<style>
  .fig-scurves {{
    --rich: #a63d2f; --poor: #2f5a78;
  }}
  .fig-scurves .stage {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 22px; }}
  .fig-scurves svg {{ display: block; width: 100%; height: auto; max-width: {width}px; margin: 0 auto; }}
  .fig-scurves figcaption {{
    font-family: {SERIF}; font-size: 15.5px;
    color: var(--ink); max-width: 70ch; margin: 18px auto 0; line-height: 1.65;
    border-top: 1px solid var(--line); padding-top: 16px;
  }}
</style>
<div class="stage">
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Choice curves for four architectures: probability of taking the 1.0 goal against the distance gap, for the base agent and for axis-written agents; the write shifts each curve laterally.">
{''.join(svg)}
</svg>
<figcaption></figcaption>
</div>
</figure>
"""
    out = SRC.parent.parent / "fragments" / "scurves.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(fragment)
    print(f"wrote {out} ({len(fragment)//1024} KB)")


if __name__ == "__main__":
    build()
