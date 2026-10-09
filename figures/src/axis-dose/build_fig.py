#!/usr/bin/env python3
"""Build the dose-response figure fragment: pure build-time SVG, no runtime JS.

    python3 figures/src/axis-dose/build_fig.py

Four panels in the 2x2 grid of the where-axis figure. Each line is one
tensor: the threshold shift from writing its slice of the axis at half, one,
two, four and eight times its own size, on log-log axes, so a linear knob is
a straight line of slope one (the faint reference lines) and saturation is a
line that bends over. The three tensors with the largest effect at x1 are
drawn in colour and named; the rest are grey. The dashed line is the full
write. Shifts beyond three times the full write are off the chart: past that
the threshold lies outside the mazes' range of distance gaps and is an
extrapolation of the fit.

`data.json` comes from figures/extract_axis_dose.py.

Output: ../../fragments/axis-dose.html - splice with <!-- fig: axis-dose -->.
"""

import json
import math
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
DATA = json.loads((SRC / "data.json").read_text())

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"
COLOURS = ("var(--ink)", "var(--poor)", "var(--rich)")

WIDTH = 790
CHART_W, CHART_H = 232, 190
COL_X = (86, 462)
ROW_Y = (46, 300)
YMIN = 0.03


def label(x, y, text, anchor="middle", size=10.5, rotate=None):
    t = f' transform="rotate(-90 {x} {y})"' if rotate else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" letter-spacing="0.11em" '
        f'fill="var(--ink-soft)" font-family="{SANS}"{t}>{text.upper()}</text>'
    )


def panel_letter(x, y, letter):
    return (
        f'<text x="{x}" y="{y}" font-size="13" font-style="italic" fill="var(--ink)" font-family="{SERIF}">({letter})</text>'
    )


def subtitle(x, y, text):
    return f'<text x="{x}" y="{y}" font-size="13" font-weight="600" fill="var(--ink)" font-family="{SANS}">{text}</text>'


class Chart:
    def __init__(self, x, y, doses, ymax):
        self.x, self.y = x, y
        self.lx = (math.log2(doses[0]) - 0.35, math.log2(doses[-1]) + 0.35)
        self.ly = (math.log10(YMIN), math.log10(ymax))

    def px(self, dose):
        return self.x + (math.log2(dose) - self.lx[0]) / (self.lx[1] - self.lx[0]) * CHART_W

    def py(self, v):
        v = max(v, YMIN)
        return self.y + CHART_H - (math.log10(v) - self.ly[0]) / (self.ly[1] - self.ly[0]) * CHART_H

    def frame(self, doses, xlabels, ylabels):
        out = [
            f'<line x1="{self.x}" y1="{self.y+CHART_H}" x2="{self.x+CHART_W}" y2="{self.y+CHART_H}" stroke="var(--ink-soft)" stroke-width="1"/>',
            f'<line x1="{self.x}" y1="{self.y}" x2="{self.x}" y2="{self.y+CHART_H}" stroke="var(--ink-soft)" stroke-width="1"/>',
        ]
        for d in doses:
            px = self.px(d)
            out.append(
                f'<line x1="{px}" y1="{self.y+CHART_H}" x2="{px}" y2="{self.y+CHART_H+4}" stroke="var(--ink-soft)" stroke-width="1"/>'
            )
            if xlabels:
                out.append(
                    f'<text x="{px}" y="{self.y+CHART_H+16}" text-anchor="middle" font-size="10" fill="var(--ink-soft)" font-family="{MONO}">×{d:g}</text>'
                )
        for v in (0.1, 1, 10):
            if v > 10 ** self.ly[1]:
                continue
            py = self.py(v)
            out.append(f'<line x1="{self.x-4}" y1="{py}" x2="{self.x}" y2="{py}" stroke="var(--ink-soft)" stroke-width="1"/>')
            if ylabels:
                out.append(
                    f'<text x="{self.x-8}" y="{py+3.5}" text-anchor="end" font-size="10" fill="var(--ink-soft)" font-family="{MONO}">{v:g}</text>'
                )
        return "".join(out)


def build():
    svg = []
    for i, panel in enumerate(DATA["panels"]):
        cx, cy = COL_X[i % 2], ROW_Y[i // 2]
        left, bottom = i % 2 == 0, i // 2 == 1
        svg.append(panel_letter(cx - 46, cy - 12, "abcd"[i]))
        svg.append(subtitle(cx - 20, cy - 12, panel["title"]))
        if panel.get("pending"):
            svg.append(label(cx + CHART_W / 2, cy + CHART_H / 2, "pending"))
            continue
        doses, full = panel["doses"], panel["full_write"]
        ymax = 3 * full
        chart = Chart(cx, cy, doses, ymax)
        # slope-one references
        for level in (full / 16, full / 4, full):
            x0, x1 = doses[0], doses[-1]
            svg.append(
                f'<line x1="{chart.px(x0):.1f}" y1="{chart.py(level * x0 / 2):.1f}" x2="{chart.px(x1):.1f}" y2="{chart.py(level * x1 / 2):.1f}" '
                f'stroke="var(--line)" stroke-width="1" stroke-dasharray="2 3"/>'
            )
        svg.append(
            f'<line x1="{chart.px(doses[0]):.1f}" y1="{chart.py(full):.1f}" x2="{chart.px(doses[-1]):.1f}" y2="{chart.py(full):.1f}" '
            f'stroke="var(--ink)" stroke-width="1" stroke-dasharray="4 3"/>'
        )
        ranked = sorted(panel["modules"], key=lambda m: -panel["tensors"][m]["mean"][doses.index(1.0)])
        top = ranked[:3]
        clip = f'<clipPath id="clip{i}"><rect x="{cx}" y="{cy - 2}" width="{CHART_W}" height="{CHART_H + 2}"/></clipPath>'
        svg.append(clip)
        svg.append(f'<g clip-path="url(#clip{i})">')
        for m in reversed(ranked):
            ys = panel["tensors"][m]["mean"]
            path = " ".join(
                f"{'M' if k == 0 else 'L'}{chart.px(d):.1f},{chart.py(v):.1f}" for k, (d, v) in enumerate(zip(doses, ys))
            )
            if m in top:
                colour = COLOURS[top.index(m)]
                svg.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.8"/>')
                for d, v in zip(doses, ys):
                    svg.append(f'<circle cx="{chart.px(d):.1f}" cy="{chart.py(v):.1f}" r="2.4" fill="{colour}"/>')
            else:
                svg.append(f'<path d="{path}" fill="none" stroke="var(--ink-soft)" stroke-width="1" opacity="0.35"/>')
        svg.append("</g>")
        # names for the coloured tensors, to the right of the last point, nudged apart
        placed = []
        for m in sorted(top, key=lambda m: -min(panel["tensors"][m]["mean"][-1], ymax)):
            y = chart.py(min(panel["tensors"][m]["mean"][-1], ymax)) + 3.5
            for other in placed:
                if abs(y - other) < 11:
                    y = other + 11
            placed.append(y)
            svg.append(
                f'<text x="{chart.px(doses[-1]) + 7:.1f}" y="{y:.1f}" font-size="9" fill="{COLOURS[top.index(m)]}" font-family="{SANS}">{m}</text>'
            )
        svg.append(chart.frame(doses, xlabels=bottom, ylabels=True))

    svg.append(label(WIDTH / 2, ROW_Y[1] + CHART_H + 38, "dose, multiple of the slice's own size"))
    mid = (ROW_Y[0] + ROW_Y[1] + CHART_H) / 2
    svg.append(label(22, mid, "threshold shift (steps)", rotate=True))
    ly = ROW_Y[1] + CHART_H + 60
    items = [("dash", "full-axis write"), ("ref", "linear reference, slope 1")]
    est = lambda t: 5.6 * len(t)
    lx = (WIDTH - (sum(24 + est(t) for _, t in items) + 26)) / 2
    for kind, text in items:
        if kind == "dash":
            svg.append(
                f'<line x1="{lx}" y1="{ly}" x2="{lx+16}" y2="{ly}" stroke="var(--ink)" stroke-width="1" stroke-dasharray="4 3"/>'
            )
        else:
            svg.append(
                f'<line x1="{lx}" y1="{ly+5}" x2="{lx+16}" y2="{ly-5}" stroke="var(--line)" stroke-width="1.4" stroke-dasharray="2 3"/>'
            )
        svg.append(
            f'<text x="{lx+24}" y="{ly+3.5}" font-size="10.5" fill="var(--ink-soft)" font-family="{SANS}">{text}</text>'
        )
        lx += 24 + est(text) + 26
    height = ly + 18

    fragment = f"""<figure class="fig-axis-dose wide" id="axis-dose-fig">
<style>
  .fig-axis-dose {{
    --rich: #a63d2f; --poor: #2f5a78;
  }}
  .fig-axis-dose .stage {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 22px; }}
  .fig-axis-dose svg {{ display: block; width: 100%; height: auto; max-width: {WIDTH}px; margin: 0 auto; }}
  .fig-axis-dose figcaption {{
    font-family: {SERIF}; font-size: 15.5px;
    color: var(--ink); max-width: 70ch; margin: 18px auto 0; line-height: 1.65;
    border-top: 1px solid var(--line); padding-top: 16px;
  }}
</style>
<div class="stage">
<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-label="Dose-response of each tensor's slice of the value axis in four architectures: threshold shift against write size on log-log axes, where a linear knob is a straight line of slope one.">
{''.join(svg)}
</svg>
<figcaption></figcaption>
</div>
</figure>
"""
    out = SRC.parent.parent / "fragments" / "axis-dose.html"
    out.write_text(fragment)
    print(f"wrote {out} ({len(fragment)//1024} KB, {WIDTH}x{height:.0f})")


if __name__ == "__main__":
    build()
