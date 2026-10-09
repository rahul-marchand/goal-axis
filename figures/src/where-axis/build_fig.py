#!/usr/bin/env python3
"""Build the where-the-axis figure fragment: pure build-time SVG, no runtime JS.

    python3 figures/src/where-axis/build_fig.py

Four panels, one per architecture, in the 2x2 grid the value-axis and s-curve
figures use. Each row is one tensor, in forward order, with three bars that
are all shares of a whole:

* pale, behind: the tensor's share of the parameters -- what a direction
  spread evenly over the weights would give;
* slate: its share of the axis's squared length, estimated from the two
  sweeps' agreement so that fine-tuning noise cancels;
* brick: its share of the axis's effect -- the threshold shift from writing
  that tensor's slice of the axis alone, over the full write's shift.

`data.json` comes from figures/extract_where_axis.py. The dose-response
companion (linearity) is a separate fragment.

Output: ../../fragments/where-axis.html - splice with <!-- fig: where-axis -->.
"""

import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
DATA = json.loads((SRC / "data.json").read_text())

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

LENGTH = "var(--poor)"  # the 0.5 goal's colour: the sweep written is that goal's value axis
EFFECT = "var(--rich)"
NULL = "var(--line)"

WIDTH = 790
COL_X = (128, 522)  # left edge of each panel's bars
LABEL_PAD = 8
CHART_W = 240
ROW_H = 22
XMAX = 100


def label(x, y, text, anchor="middle", size=10.5):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" letter-spacing="0.11em" '
        f'fill="var(--ink-soft)" font-family="{SANS}">{text.upper()}</text>'
    )


def panel_letter(x, y, letter):
    return (
        f'<text x="{x}" y="{y}" font-size="13" font-style="italic" fill="var(--ink)" font-family="{SERIF}">({letter})</text>'
    )


def subtitle(x, y, text):
    return f'<text x="{x}" y="{y}" font-size="13" font-weight="600" fill="var(--ink)" font-family="{SANS}">{text}</text>'


def px(x0, v):
    return x0 + max(min(v, XMAX), 0) / XMAX * CHART_W


def bar(x0, row_y, value, height, fill):
    return f'<rect x="{x0}" y="{row_y - height/2:.1f}" width="{px(x0, value) - x0:.1f}" height="{height}" fill="{fill}"/>'


def whisker(x0, row_y, lo, hi):
    a, b = px(x0, lo), px(x0, hi)
    return (
        f'<line x1="{a:.1f}" y1="{row_y:.1f}" x2="{b:.1f}" y2="{row_y:.1f}" stroke="var(--ink)" stroke-width="0.9"/>'
        f'<line x1="{a:.1f}" y1="{row_y-2.2:.1f}" x2="{a:.1f}" y2="{row_y+2.2:.1f}" stroke="var(--ink)" stroke-width="0.9"/>'
        f'<line x1="{b:.1f}" y1="{row_y-2.2:.1f}" x2="{b:.1f}" y2="{row_y+2.2:.1f}" stroke="var(--ink)" stroke-width="0.9"/>'
    )


def frame(x0, y, h):
    out = [
        f'<line x1="{x0}" y1="{y}" x2="{x0}" y2="{y+h}" stroke="var(--ink-soft)" stroke-width="1"/>',
        f'<line x1="{x0}" y1="{y+h}" x2="{x0+CHART_W}" y2="{y+h}" stroke="var(--ink-soft)" stroke-width="1"/>',
    ]
    for t in (0, 25, 50, 75, 100):
        tx = px(x0, t)
        out.append(f'<line x1="{tx}" y1="{y+h}" x2="{tx}" y2="{y+h+4}" stroke="var(--ink-soft)" stroke-width="1"/>')
        out.append(
            f'<text x="{tx}" y="{y+h+15}" text-anchor="middle" font-size="10" fill="var(--ink-soft)" font-family="{MONO}">{t}</text>'
        )
    return "".join(out)


def build():
    svg = []
    panels = DATA["panels"]
    # row layout: panels 0,1 on the first row, 2,3 on the second; each row as tall as its taller panel
    heights = [ROW_H * len(p["modules"]) for p in panels]
    row_tops = [44, 44 + max(heights[0], heights[1]) + 62]
    row_frame = [max(heights[0], heights[1]), max(heights[2], heights[3])]
    for i, panel in enumerate(panels):
        x0, y = COL_X[i % 2], row_tops[i // 2]
        svg.append(panel_letter(x0 - 120, y - 14, "abcd"[i]))
        svg.append(subtitle(x0 - 96, y - 14, panel["title"]))
        modules = panel["modules"]
        h = ROW_H * len(modules)
        many = len(panel["bases"]) > 1
        for k, name in enumerate(modules):
            ry = y + ROW_H * (k + 0.5)
            svg.append(
                f'<text x="{x0 - LABEL_PAD}" y="{ry+3.5:.1f}" text-anchor="end" font-size="10" fill="var(--ink)" '
                f'font-family="{SANS}">{name}</text>'
            )
            # three side-by-side bars: parameters, length, effect
            s, e = panel["axis_share"][k], panel["alone"][k]
            full = panel["full_write"]
            svg.append(bar(x0, ry - 6.0, panel["params_share"][k] * 100, 5.6, NULL))
            svg.append(bar(x0, ry, s["mean"] * 100, 5.6, LENGTH))
            svg.append(bar(x0, ry + 6.0, e["mean"] / full * 100, 5.6, EFFECT))
            if many:
                svg.append(whisker(x0, ry, s["lo"] * 100, s["hi"] * 100))
                svg.append(whisker(x0, ry + 6.0, e["lo"] / full * 100, e["hi"] / full * 100))
        # frames share the row height so the x-axes align; bars hang from the top
        svg.append(frame(x0, y, row_frame[i // 2]))

    bottom = row_tops[1] + max(heights[2], heights[3])
    svg.append(label(WIDTH / 2, bottom + 40, "percent of the whole"))

    ly = bottom + 62
    items = [(NULL, "share of parameters"), (LENGTH, "share of the axis's length"), (EFFECT, "share of the axis's effect")]
    est = lambda t: 5.6 * len(t)
    lx = (WIDTH - (sum(24 + est(t) for _, t in items) + 40 + 26 * len(items))) / 2
    for colour, text in items:
        svg.append(f'<rect x="{lx}" y="{ly-5}" width="16" height="10" fill="{colour}"/>')
        svg.append(
            f'<text x="{lx+24}" y="{ly+3.5}" font-size="10.5" fill="var(--ink-soft)" font-family="{SANS}">{text}</text>'
        )
        lx += 24 + est(text) + 26
    svg.append(whisker(lx - 8, ly, 0, 100 * 16 / CHART_W))
    svg.append(
        f'<text x="{lx+16}" y="{ly+3.5}" font-size="10.5" fill="var(--ink-soft)" font-family="{SANS}">range across bases</text>'
    )
    height = ly + 18

    fragment = f"""<figure class="fig-where-axis wide" id="where-axis-fig">
<style>
  .fig-where-axis {{
    --rich: #a63d2f; --poor: #2f5a78;
  }}
  .fig-where-axis .stage {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 22px; }}
  .fig-where-axis svg {{ display: block; width: 100%; height: auto; max-width: {WIDTH}px; margin: 0 auto; }}
  .fig-where-axis figcaption {{
    font-family: {SERIF}; font-size: 15.5px;
    color: var(--ink); max-width: 70ch; margin: 18px auto 0; line-height: 1.65;
    border-top: 1px solid var(--line); padding-top: 16px;
  }}
</style>
<div class="stage">
<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-label="Where the value axis sits in four architectures: per tensor, its share of the parameters, of the axis's length, and of the axis's effect when written alone.">
{''.join(svg)}
</svg>
<figcaption></figcaption>
</div>
</figure>
"""
    out = SRC.parent.parent / "fragments" / "where-axis.html"
    out.write_text(fragment)
    print(f"wrote {out} ({len(fragment)//1024} KB, {WIDTH}x{height:.0f})")


if __name__ == "__main__":
    build()
