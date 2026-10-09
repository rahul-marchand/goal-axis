# Generates fragments/hierarchy.html in the site's fragment convention,
# and preview.svg beside this script.
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 790, 356
SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"
SERIF = "'Source Serif 4', Georgia, serif"
o = []


def t(x, y, s, size=12, fill="var(--ink)", anchor="middle", weight=None, family=SANS, italic=False, dy=None):
    a = f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-family="{family}"'
    if weight:
        a += f' font-weight="{weight}"'
    if italic:
        a += ' font-style="italic"'
    if dy:
        a += f' dy="{dy}"'
    o.append(f'<text x="{x}" y="{y}" {a}>{s}</text>')


def box(x, y, w, h, dashed=False):
    st = 'stroke="var(--ink-soft)" stroke-dasharray="5 4"' if dashed else 'stroke="var(--ink)"'
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="var(--paper)" {st} stroke-width="1.4"/>')


def arrow(x1, y1, x2, y2, dashed=False, path=None):
    st = (
        'stroke="var(--ink-soft)" stroke-dasharray="5 4" marker-end="url(#ah-soft)"'
        if dashed
        else 'stroke="var(--ink)" marker-end="url(#ah)"'
    )
    d = path or f"M {x1} {y1} L {x2} {y2}"
    o.append(f'<path d="{d}" fill="none" {st} stroke-width="1.4"/>')


# ---- geometry
Y, BH = 140, 70
OBS = (18, 92)
PLN = (150, 150)
VAL = (340, 150)
CMP = (530, 150)
ACT = (720, 52)
cx = lambda b: b[0] + b[1] / 2

# legend
o.append('<path d="M 18 30 L 48 30" stroke="var(--ink)" stroke-width="1.4"/>')
t(56, 34, "established in this post", 11.5, anchor="start")
o.append('<path d="M 18 50 L 48 50" stroke="var(--ink-soft)" stroke-width="1.4" stroke-dasharray="5 4"/>')
t(56, 54, "hypothesised, tested in post 2", 11.5, fill="var(--ink-soft)", anchor="start")

# main boxes
box(*OBS[:1], Y, OBS[1], BH)
box(PLN[0], Y, PLN[1], BH)
box(VAL[0], Y, VAL[1], BH, True)
box(CMP[0], Y, CMP[1], BH, True)
box(ACT[0], Y, ACT[1], BH)

# observation glyph: four offset planes
for i, col in enumerate(["var(--ink)", "var(--ink-soft)", "var(--rich)", "var(--poor)"]):
    o.append(
        f'<rect x="{34+i*6}" y="{Y+14+i*4}" width="22" height="22" fill="var(--panel)" stroke="{col}" stroke-width="1.2"/>'
    )
t(cx(OBS), Y + 62, "observation", 11.5, fill="var(--ink-soft)")

# planning glyph: mini maze with two routes
gx, gy = PLN[0] + 14, Y + 12
cs = 7
o.append(f'<rect x="{gx}" y="{gy}" width="{cs*5}" height="{cs*5}" fill="var(--panel)" stroke="var(--line)"/>')
o.append(
    f'<polyline points="{gx+cs*0.5},{gy+cs*4.5} {gx+cs*0.5},{gy+cs*2.5} {gx+cs*2.5},{gy+cs*2.5} {gx+cs*2.5},{gy+cs*0.5} {gx+cs*4.5},{gy+cs*0.5}" fill="none" stroke="var(--rich)" stroke-width="1.6"/>'
)
o.append(
    f'<polyline points="{gx+cs*0.5},{gy+cs*4.5} {gx+cs*2.5},{gy+cs*4.5} {gx+cs*2.5},{gy+cs*3.5} {gx+cs*4.5},{gy+cs*3.5}" fill="none" stroke="var(--poor)" stroke-width="1.6"/>'
)
o.append(f'<circle cx="{gx+cs*0.5}" cy="{gy+cs*4.5}" r="2.2" fill="var(--ink)"/>')
t(PLN[0] + 62, Y + 26, "planning", 13, anchor="start", weight="600")
t(PLN[0] + 62, Y + 44, "a route to", 11.5, fill="var(--ink-soft)", anchor="start")
t(PLN[0] + 62, Y + 58, "each goal", 11.5, fill="var(--ink-soft)", anchor="start")

# valuation
t(cx(VAL), Y + 26, "valuation", 13, weight="600")
t(cx(VAL), Y + 45, "score each route", 11.5, fill="var(--ink-soft)")
t(cx(VAL), Y + 61, "R − c · d", 12, family=MONO)

# comparison
t(cx(CMP), Y + 26, "comparison", 13, weight="600")
t(cx(CMP), Y + 45, "pick the higher score", 11.5, fill="var(--ink-soft)")
t(cx(CMP), Y + 61, "threshold τ̂", 12, family=MONO)

# action
o.append(f'<path d="M {ACT[0]+14} {Y+35} L {ACT[0]+36} {Y+35}" stroke="var(--ink)" stroke-width="1.8" marker-end="url(#ah)"/>')
t(cx(ACT), Y + 62, "action", 11.5, fill="var(--ink-soft)")

# pipeline arrows
arrow(OBS[0] + OBS[1], Y + BH / 2, PLN[0] - 3, Y + BH / 2)
arrow(PLN[0] + PLN[1], Y + BH / 2, VAL[0] - 3, Y + BH / 2, True)
arrow(VAL[0] + VAL[1], Y + BH / 2, CMP[0] - 3, Y + BH / 2, True)
arrow(CMP[0] + CMP[1], Y + BH / 2, ACT[0] - 3, Y + BH / 2, True)

# value parameter node + write
rx, ry = cx(VAL), 82
t(rx, 26, "write  θ + α v", 12.5, family=MONO, weight="500")
arrow(rx, 34, rx, ry - 19)
o.append(f'<circle cx="{rx}" cy="{ry}" r="16" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.4"/>')
t(rx, ry + 4.5, "τ", 13, family=MONO, weight="600")
t(rx + 24, ry + 4, "value–step trade-off, set by v", 11.5, fill="var(--ink-soft)", anchor="start")
arrow(rx, ry + 16, rx, Y - 3, True)
arrow(0, 0, 0, 0, True, path=f"M {rx-16} {ry+4} Q {PLN[0]+40} {ry+10} {PLN[0]+12} {Y-3}")
t(PLN[0] + 52, ry + 38, "in the DRC, v concentrates", 11, fill="var(--ink-soft)", anchor="start")
t(PLN[0] + 52, ry + 52, "in the input convolution", 11, fill="var(--ink-soft)", anchor="start")

# probe annotation under planning
py = Y + BH + 26
o.append(f'<circle cx="{PLN[0]+10}" cy="{py-4}" r="4.5" fill="var(--probe)"/>')
o.append(f'<path d="M {PLN[0]+10} {py-8} L {PLN[0]+10} {Y+BH+2}" stroke="var(--probe)" stroke-width="1.4"/>')
t(PLN[0] + 22, py, "linear probe reads the route", 12, anchor="start")
t(PLN[0] + 22, py + 16, "path channels, extended across ticks,", 11, fill="var(--ink-soft)", anchor="start")
t(PLN[0] + 22, py + 30, "pruned by backtracking", 11, fill="var(--ink-soft)", anchor="start")

# choice-curve glyph under comparison

x0, x1, yb, yt = CMP[0] + 18, CMP[0] + CMP[1] - 18, Y + BH + 62, Y + BH + 18
pts = []
for i in range(41):
    u = i / 40
    xx = x0 + u * (x1 - x0)
    p = 1 / (1 + math.exp((u - 0.5) * 10))
    pts.append(f"{xx:.1f},{yb - p*(yb-yt):.1f}")
o.append(f'<path d="M {x0} {yb} L {x1} {yb}" stroke="var(--line)" stroke-width="1"/>')
o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="var(--ink)" stroke-width="1.6"/>')
xm = (x0 + x1) / 2
o.append(f'<path d="M {xm} {yt} L {xm} {yb}" stroke="var(--ink-soft)" stroke-width="1" stroke-dasharray="3 3"/>')
o.append(
    f'<path d="M {xm-14} {yb+8} L {xm+14} {yb+8}" stroke="var(--ink)" stroke-width="1.2" marker-end="url(#ah)" marker-start="url(#ah-rev)"/>'
)
t(xm, yb + 22, "τ̂ moves with α, s stretches with it", 11, fill="var(--ink-soft)")
t(x0 - 4, yb + 3, "gap", 10, fill="var(--ink-soft)", anchor="end")

# bracket: route unchanged
by = Y + BH + 108
o.append(
    f'<path d="M {PLN[0]} {by-6} L {PLN[0]} {by} L {ACT[0]+ACT[1]} {by} L {ACT[0]+ACT[1]} {by-6}" fill="none" stroke="var(--ink)" stroke-width="1.2"/>'
)
t((PLN[0] + ACT[0] + ACT[1]) / 2, by + 16, "the route stays shortest-path under the write", 12)

defs = """<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink)"/></marker>
<marker id="ah-rev" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 10 0 L 0 5 L 10 10 z" fill="var(--ink)"/></marker>
<marker id="ah-soft" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-soft)"/></marker>
</defs>"""
label = (
    "Hypothesised organisation of the DRC agent: observation, planning, valuation, comparison, action. "
    "Planning and the effect of the weight write are established; valuation and comparison are hypothesised."
)
svg = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{label}">\n{defs}\n' + "\n".join(o) + "\n</svg>"
frag = f"""<figure class="fig-hierarchy wide" id="hierarchy-fig">
<style>
  .fig-hierarchy {{
    --rich: #a63d2f; --poor: #2f5a78; --probe: #6f5f9c;
  }}
  .fig-hierarchy .stage {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 22px; }}
  .fig-hierarchy svg {{ display: block; width: 100%; height: auto; max-width: 790px; margin: 0 auto; }}
  .fig-hierarchy figcaption {{
    font-family: 'Source Serif 4', Georgia, serif; font-size: 15.5px;
    color: var(--ink); max-width: 70ch; margin: 18px auto 0; line-height: 1.65;
    border-top: 1px solid var(--line); padding-top: 16px;
  }}
</style>
<div class="stage">
{svg}
<figcaption></figcaption>
</div>
</figure>
"""
(HERE.parent.parent / "fragments" / "hierarchy.html").write_text(frag)
# standalone preview with vars substituted
sub = {
    "--paper": "#f8f6f2",
    "--panel": "#fdfcfa",
    "--ink": "#23252b",
    "--ink-soft": "#5c5f6a",
    "--line": "#ddd8ce",
    "--rich": "#a63d2f",
    "--poor": "#2f5a78",
    "--probe": "#6f5f9c",
}
pv = svg
for k, v in sub.items():
    pv = pv.replace(f"var({k})", v)
(HERE / "preview.svg").write_text(
    '<?xml version="1.0"?>\n' + pv.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" style="background:#fdfcfa" ', 1)
)
print("written")
