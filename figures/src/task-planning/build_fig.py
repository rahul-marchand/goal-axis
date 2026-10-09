#!/usr/bin/env python3
"""Build the task-planning figure fragment: pure build-time SVG, no runtime JS.

    python3 figures/src/task-planning/build_fig.py

Two stacked panels, subfigure style: (a) the four observation planes the agent
receives, (b) the plan probe read out of the DRC's recurrent state before its
first action, with the route the agent then walked drawn on top. The model zoo
lives in `../../fragments/model-table.html` as a typeset table, not here.

`data.json` carries the planes' level (walls, objectives, agent) and
`plan_examples: [{walls, objectives, heat (11x11, null at walls), path
([[r,c],...]), reached_fid}]` — the first two entries are drawn, and the first
one should be the same level as the planes so the panels visibly share a world.

Output: ../../fragments/task-planning.html — splice with <!-- fig: task-planning -->.
"""

import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
DATA = json.loads((SRC / "data.json").read_text())

CELL = 15.0  # one cell size everywhere: the panels must weigh the same
PLAN_CELL = 15.0

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

# Heat colours are baked to hex rather than color-mix()ed from tokens: SVG fill
# attributes predate color-mix support in some renderers, and a baked ramp
# survives PNG/PDF export for the paper lift unchanged.
FLOOR_RGB = (238, 235, 227)
PROBE_RGB = (111, 95, 156)


def heat_hex(t: float) -> str:
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(round(f + (p - f) * t) for f, p in zip(FLOOR_RGB, PROBE_RGB))


def grid(x, y, cell, walls, fills, rx=1.0, border=True, lattice=False):
    """An 11x11 grid: floor everywhere, walls in ink, `fills` on top.

    ``lattice`` draws hairline cell divisions — a nearly-empty one-hot plane
    over a lattice reads as a matrix, not a blank card.
    """
    out = [f'<rect x="{x}" y="{y}" width="{cell*11}" height="{cell*11}" fill="var(--floor)" rx="2"/>']
    if lattice:
        for k in range(1, 11):
            out.append(
                f'<line x1="{x+k*cell}" y1="{y}" x2="{x+k*cell}" y2="{y+cell*11}" stroke="var(--line)" stroke-width="0.5"/>'
                f'<line x1="{x}" y1="{y+k*cell}" x2="{x+cell*11}" y2="{y+k*cell}" stroke="var(--line)" stroke-width="0.5"/>'
            )
    for r, row in enumerate(walls):
        for c, ch in enumerate(row):
            if ch == "1":
                out.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" fill="var(--wall)"/>')
    for (r, c), fill in fills.items():
        out.append(f'<rect x="{x+c*cell+0.5}" y="{y+r*cell+0.5}" width="{cell-1}" height="{cell-1}" fill="{fill}" rx="{rx}"/>')
    if border:
        out.append(
            f'<rect x="{x-0.5}" y="{y-0.5}" width="{cell*11+1}" height="{cell*11+1}" fill="none" '
            f'stroke="var(--line)" stroke-width="1" rx="2.5"/>'
        )
    return "".join(out)


def label(x: float, y: float, text: str, anchor: str = "middle", size: float = 10.5) -> str:
    """A small-caps figure label, the same voice as the post's .lbl class."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" letter-spacing="0.11em" '
        f'fill="var(--ink-soft)" font-family="{SANS}">{text.upper()}</text>'
    )


def panel_letter(x: float, y: float, letter: str) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="13" font-style="italic" fill="var(--ink)" '
        f'font-family="{SERIF}">({letter})</text>'
    )


def draw_example(x, y, example):
    """One plan maze: heat under walls and goals, the walked route on top."""
    out = []
    fills = {}
    for r in range(11):
        for c in range(11):
            heat = example["heat"][r][c]
            if heat is not None and example["walls"][r][c] == "0":
                fills[(r, c)] = heat_hex(heat)
    out.append(grid(x, y, PLAN_CELL, example["walls"], fills, rx=1.5))
    for o in example["objectives"]:
        orow, ocol = o["position"]
        colour = "var(--rich)" if o["feature_id"] == 0 else "var(--poor)"
        out.append(
            f'<rect x="{x+ocol*PLAN_CELL+0.5}" y="{y+orow*PLAN_CELL+0.5}" '
            f'width="{PLAN_CELL-1}" height="{PLAN_CELL-1}" fill="{colour}"/>'
        )
    path = example.get("path") or []
    if path:
        points = " ".join(f"{x+c*PLAN_CELL+PLAN_CELL/2},{y+r*PLAN_CELL+PLAN_CELL/2}" for r, c in path)
        colour = "var(--rich)" if example.get("reached_fid") == 0 else "var(--poor)"
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2.6" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>'
        )
        r0, c0 = path[0]
        out.append(
            f'<circle cx="{x+c0*PLAN_CELL+PLAN_CELL/2}" cy="{y+r0*PLAN_CELL+PLAN_CELL/2}" '
            f'r="{PLAN_CELL*0.17}" fill="var(--ink)"/>'
        )
    return "".join(out)


def build():
    level = DATA["planes_level"]
    walls = level["walls"]
    objectives = level["objectives"]
    agent_cell = tuple(level["agent"])

    w = CELL * 11  # 165, both panels
    svg = []

    # Two identical 2x2 blocks on one row grid: every grid top in (a) aligns
    # with its counterpart in (b), and the annotation zone under each grid is
    # the same height on both sides — a channel label on the left, the fig-5
    # utility arithmetic (value − 0.05·distance, chosen goal bold) on the right.
    head_y = 26
    row0 = head_y + 16
    stride = w + 44
    block_w = 2 * w + 18
    a_x, b_x = 24, 24 + block_w + 46

    def subtitle(x, text):
        return (
            f'<text x="{x}" y="{head_y}" font-size="13" font-weight="600" fill="var(--ink)" '
            f'font-family="{SANS}">{text}</text>'
        )

    svg.append(panel_letter(a_x, head_y, "a"))
    svg.append(subtitle(a_x + 26, "Input channels"))
    svg.append(panel_letter(b_x, head_y, "b"))
    svg.append(subtitle(b_x + 26, "Decoded plans"))

    # --- (a) the observation: four binary planes, values nowhere among them
    planes = [
        ("walls", {(r, c): "var(--wall)" for r in range(11) for c in range(11) if walls[r][c] == "1"}),
        ("agent", {agent_cell: "var(--ink)"}),
        ("colour 0", {tuple(o["position"]): "var(--rich)" for o in objectives if o["feature_id"] == 0}),
        ("colour 1", {tuple(o["position"]): "var(--poor)" for o in objectives if o["feature_id"] == 1}),
    ]
    for i, (name, fills) in enumerate(planes):
        x = a_x + (i % 2) * (w + 18)
        y = row0 + (i // 2) * stride
        svg.append(grid(x, y, CELL, ["0" * 11] * 11, fills, lattice=True))
        svg.append(label(x + w / 2, y + w + 17, name))

    # --- (b) the plan probe on four held-out mazes, walked routes on top
    examples = DATA.get("plan_examples", [])[:4]
    for i, example in enumerate(examples):
        x = b_x + (i % 2) * (w + 18)
        y = row0 + (i // 2) * stride
        svg.append(draw_example(x, y, example))
        chosen = example["reached_fid"]
        for j, (fid, value, d) in enumerate(((0, "1.0", example["d0"]), (1, "0.5", example["d1"]))):
            utility = float(value) - 0.05 * d
            colour = "var(--rich)" if fid == 0 else "var(--poor)"
            weight = "600" if fid == chosen else "400"
            svg.append(
                f'<text x="{x + w / 2}" y="{y + w + 14 + j * 14}" text-anchor="middle" '
                f'font-size="11" font-weight="{weight}" fill="{colour}" font-family="{MONO}">'
                f"{value} − 0.05×{d} = {utility:+.2f}</text>"
            )

    width = b_x + block_w + 24

    ly = row0 + stride + w + 40
    ramp_w = 5 * 13 + 4
    ramp_x = b_x + block_w / 2 - (ramp_w + 74) / 2
    for i in range(5):
        svg.append(f'<rect x="{ramp_x+i*13}" y="{ly}" width="12" height="8" rx="1.5" fill="{heat_hex(i/4)}"/>')
    svg.append(label(ramp_x + ramp_w + 8, ly + 7.5, "p(will visit)", anchor="start"))

    height = ly + 30

    # Caption lives in the post markdown (::: {.figcap for=task-planning}); the empty
    # figcaption below is the splice anchor.

    fragment = f"""<figure class="fig-taskplan wide" id="taskplan-fig">
<style>
  .fig-taskplan {{
    --wall: #2e313b; --floor: #eeebe3;
    --rich: #a63d2f; --poor: #2f5a78; --probe: #6f5f9c;
  }}
  .fig-taskplan .stage {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 22px; }}
  .fig-taskplan svg {{ display: block; width: 100%; height: auto; max-width: {width}px; margin: 0 auto; }}
  .fig-taskplan figcaption {{
    font-family: {SERIF}; font-size: 15.5px;
    color: var(--ink); max-width: 70ch; margin: 18px auto 0; line-height: 1.65;
    border-top: 1px solid var(--line); padding-top: 16px;
  }}
</style>
<div class="stage">
<svg viewBox="0 0 {width} {height}" role="img" aria-label="The agent's four observation planes, and a linear probe's per-cell prediction of the route the DRC will walk, shown with the route it then took.">
{''.join(svg)}
</svg>
<figcaption></figcaption>
</div>
</figure>
"""
    out = SRC.parent.parent / "fragments" / "task-planning.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(fragment)
    print(f"wrote {out} ({len(fragment)//1024} KB)")


if __name__ == "__main__":
    build()
