"""The crafting task in one figure, drawn from a recorded rollout of the trained policy.

    uv run python figures/make_craft_fig1.py [--data figures/data/craftax/craft/fig1.npz] [--out figures/craftax/fig_craft_task]

A two-by-two of equal panels in the blog's voice (italic serif letter, bold
subtitle, small-caps labels, rich red for the richer goal and slate blue for the
poorer). (a) One held-out field with the engine's own block textures and the
policy's closed-loop rollout drawn over the expert's two plans. (b) The
observation the policy gets at one step of that rollout: seven map planes and
eight planes that are constant over the grid. (c) The tech tree the expert
plans over. (d) The cost gap between the two chains over the held-out fields
and the threshold the values set. Everything is read from ``fig1.npz`` /
``fig1.json``, written by ``figures/craft_fig1_data.py`` on a machine holding
the checkpoint.

Bedrock walls are the engine's furnace block (unminable, recipe-inert); they are
drawn as dark stone rather than with the furnace's own lit texture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from goalmisgen.craftax import simulate  # noqa: E402
from goalmisgen.craftax.blocks import BEDROCK, Action, Block  # noqa: E402

RICH, POOR, PROBE = "#a63d2f", "#2f5a78", "#6f5f9c"
INK, SOFT, LINE = "#1a1a1a", "#5c5a55", "#c9c7bf"
WALL, FLOOR, PANEL = "#2e313b", "#eeebe3", "#fcfcfb"
PX = 64
SANS = ["IBM Plex Sans", "Source Sans 3", "DejaVu Sans"]
SERIF = ["Source Serif 4", "Georgia", "DejaVu Serif"]

mpl.rcParams.update(
    {
        "figure.facecolor": PANEL,
        "axes.facecolor": PANEL,
        "savefig.facecolor": PANEL,
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": SOFT,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.edgecolor": LINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 170,
    }
)

DIR = {int(Action.UP): (-1, 0), int(Action.DOWN): (1, 0), int(Action.LEFT): (0, -1), int(Action.RIGHT): (0, 1)}
HALO = [pe.Stroke(linewidth=3.4, foreground="white", alpha=0.95), pe.Normal()]


def caps(text: str) -> str:
    """Small-caps label voice: upper case with a hair of tracking."""
    return " ".join(text.upper())


def header(fig, ax, letter: str, title: str, sub: str | None = None) -> None:
    box = ax.get_position(fig) if hasattr(ax, "get_position") else ax
    fig.text(box.x0, box.y1 + 0.026, f"({letter})", fontsize=12, fontstyle="italic", family=SERIF, color=INK, va="bottom")
    fig.text(box.x0 + 0.032, box.y1 + 0.026, title, fontsize=10.5, fontweight=600, color=INK, va="bottom")
    if sub:
        fig.text(box.x0 + 0.032, box.y1 + 0.009, sub, fontsize=7.4, color=SOFT, va="bottom")


# ---------------------------------------------------------------- assets
def textures() -> dict:
    from craftax.craftax_classic.constants import load_all_textures

    tex = load_all_textures()[PX]
    blocks = np.asarray(tex["block_textures"], dtype=np.float32) / 255.0
    blocks = blocks.copy()
    stone = blocks[int(Block.STONE)]
    blocks[int(BEDROCK)] = 0.42 * stone.mean(axis=-1, keepdims=True) + 0.03
    players = [np.asarray(p, dtype=np.float32) / 255.0 for p in tex["player_textures"]]
    icons = {name: np.asarray(tex[f"{name}_texture"], dtype=np.float32) / 255.0 for name in ("wood_pickaxe", "stone_pickaxe")}
    return {"blocks": blocks, "players": players, "icons": icons}


def field_image(tiles: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    size = tiles.shape[0]
    image = np.zeros((size * PX, size * PX, 3), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            image[r * PX : (r + 1) * PX, c * PX : (c + 1) * PX] = blocks[int(tiles[r, c])]
    return image


def paste(image: np.ndarray, rgba: np.ndarray, r: int, c: int, alpha: float = 1.0) -> None:
    patch = image[r * PX : (r + 1) * PX, c * PX : (c + 1) * PX]
    a = rgba[..., 3:4] * alpha if rgba.shape[-1] == 4 else alpha
    patch[:] = patch * (1 - a) + rgba[..., :3] * a


def walk(tiles, start, facing, actions) -> list[tuple[int, int]]:
    """Positions visited by a plan, from the numpy engine twin."""
    state = simulate.initial(np.asarray(tiles), tuple(int(v) for v in start), int(facing))
    path = [state.position]
    for action in actions:
        state = simulate.step(state, int(action))
        path.append(state.position)
    return path


def events(data) -> list[tuple[str, tuple[int, int], int]]:
    """What happened where along the rollout: (label, cell, step)."""
    actions, positions, facings, inventories, tiles = (
        data["actions"], data["positions"], data["facings"], data["inventories"], data["tiles"]
    )
    out = []
    for t, action in enumerate(actions):
        pos = tuple(int(v) for v in positions[t])
        dr, dc = DIR[int(facings[t])]
        faced = (pos[0] + dr, pos[1] + dc)
        gained = inventories[t + 1] - inventories[t]
        action = int(action)
        if action == int(Action.DO):
            if gained[0] > 0:
                out.append(("tree cut", faced, t))
            elif gained[1] > 0:
                out.append(("stone mined", faced, t))
            elif t == len(actions) - 1:
                out.append((("iron" if int(tiles[faced]) == int(Block.IRON) else "coal") + " mined", faced, t))
        elif action == int(Action.PLACE_TABLE):
            out.append(("table placed", faced, t))
        elif action == int(Action.MAKE_WOOD_PICKAXE):
            out.append(("wood pickaxe", pos, t))
        elif action == int(Action.MAKE_STONE_PICKAXE):
            out.append(("stone pickaxe", pos, t))
    return out


def centre(cell) -> tuple[float, float]:
    return cell[1] + 0.5, cell[0] + 0.5


# ---------------------------------------------------------------- panel (a)
def panel_field(fig, ax, data, meta, tex) -> int:
    tiles = data["tiles"]
    start = data["agent_start"]
    image = field_image(tiles, tex["blocks"])
    for r, c in zip(*np.nonzero(data["maps"][-1] == int(Block.CRAFTING_TABLE))):
        paste(image, np.concatenate([tex["blocks"][int(Block.CRAFTING_TABLE)], np.ones((PX, PX, 1))], -1), r, c, alpha=0.55)
    paste(image, tex["players"][int(data["start_direction"]) - 1], int(start[0]), int(start[1]))
    size = tiles.shape[0]
    ax.imshow(image, extent=(0, size, size, 0), interpolation="nearest")

    iron = walk(tiles, start, data["start_direction"], data["plan_iron"])
    coal = walk(tiles, start, data["start_direction"], data["plan_coal"])
    rollout = [tuple(int(v) for v in p) for p in data["positions"]]
    xs, ys = zip(*[centre(p) for p in iron])
    ax.plot(xs, ys, color=RICH, lw=7.5, alpha=0.38, solid_capstyle="round", solid_joinstyle="round")
    xs, ys = zip(*[(x + 0.2, y + 0.2) for x, y in (centre(p) for p in coal)])
    ax.plot(xs, ys, color=POOR, lw=1.7, ls=(0, (3, 2)), solid_capstyle="round")[0].set_path_effects(HALO)
    xs, ys = zip(*[centre(p) for p in rollout])
    ax.plot(xs, ys, color=INK, lw=1.5, solid_capstyle="round", solid_joinstyle="round")[0].set_path_effects(HALO)

    marks = events(data)
    seen: dict[tuple[int, int], int] = {}
    key = []
    for k, (label, cell, t) in enumerate(marks, 1):
        x, y = centre(cell)
        n = seen.get(cell, 0)
        seen[cell] = n + 1
        x, y = x + 0.34 * n, y - 0.34 * n
        ax.plot(x, y, marker="o", ms=9.5, color="white", mec=INK, mew=0.8, zorder=5)
        ax.text(x, y, str(k), ha="center", va="center", fontsize=6.2, color=INK, zorder=6, fontweight="bold")
        key.append((k, label, t + 1))
    frame = next(t + 1 for label, _, t in marks if label == "wood pickaxe")
    fx, fy = centre(rollout[frame])
    ax.plot(fx, fy, marker="s", ms=14, mfc="none", mec=PROBE, mew=1.7, zorder=7)
    ax.text(fx + 0.65, fy, "(b)", color=PROBE, fontsize=8.5, fontstyle="italic", family=SERIF, va="center", zorder=7).set_path_effects(HALO)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend and the numbered key, in the band under the map.
    y0 = -0.04
    swatches = [
        (RICH, 7.0, 0.38, "-", f"expert's iron plan · {len(data['plan_iron'])}"),
        (POOR, 1.7, 1.0, (0, (3, 2)), f"expert's coal plan · {len(data['plan_coal'])}"),
        (INK, 1.5, 1.0, "-", f"policy's rollout · {len(rollout) - 1} actions"),
    ]
    for i, (colour, lw, alpha, ls, text) in enumerate(swatches):
        x = 0.33 * i
        ax.plot([x, x + 0.04], [y0, y0], color=colour, lw=lw, alpha=alpha, ls=ls, transform=ax.transAxes, clip_on=False, solid_capstyle="round")
        ax.text(x + 0.052, y0, caps(text), transform=ax.transAxes, fontsize=5.6, color=SOFT, va="center", ha="left")
    for i, (k, label, step) in enumerate(key):
        row, col = divmod(i, 3)
        x = 0.35 * col
        y = y0 - 0.05 - 0.037 * row
        ax.plot(x + 0.012, y, marker="o", ms=8, color="white", mec=INK, mew=0.7, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.012, y, str(k), transform=ax.transAxes, fontsize=5.4, color=INK, ha="center", va="center", fontweight="bold")
        ax.text(x + 0.04, y, f"{label} · step {step}", transform=ax.transAxes, fontsize=6.9, color=SOFT, va="center")
    return frame


# ---------------------------------------------------------------- panel (b)
def plane_grid(ax, plane: np.ndarray, colour: str, constant: bool) -> None:
    size = plane.shape[0]
    ax.add_patch(Rectangle((0, 0), size, size, color=FLOOR, lw=0))
    if constant:
        value = float(plane[0, 0])
        ax.add_patch(Rectangle((0, 0), size, size, color=colour, alpha=value, lw=0))
        ax.text(size / 2, size / 2, f"{value:g}", ha="center", va="center", fontsize=8.5, color=INK if value < 0.5 else "white")
    else:
        for k in range(1, size):
            ax.plot([k, k], [0, size], color=LINE, lw=0.35)
            ax.plot([0, size], [k, k], color=LINE, lw=0.35)
        for r, c in zip(*np.nonzero(plane > 0)):
            ax.add_patch(Rectangle((c + 0.08, r + 0.08), 0.84, 0.84, color=colour, lw=0))
    ax.add_patch(Rectangle((0, 0), size, size, fill=False, ec=LINE, lw=0.8))
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_planes(fig, box, data, meta, frame) -> None:
    """Fifteen planes as a 3 x 5 grid of lattice cards filling ``box`` (figure coordinates)."""
    obs = data["observations"][frame]
    names = meta["channels"]
    colours = [WALL, INK, INK, INK, INK, RICH, POOR] + [INK] * 8
    cols, rows = 4, 4
    gap_x, gap_y = 0.012, 0.022
    w = (box.width - gap_x * (cols - 1)) / cols
    h = min(w * (fig.get_figwidth() / fig.get_figheight()), (box.height - gap_y * (rows - 1)) / rows)
    top = box.y1
    axes = []
    for i in range(obs.shape[-1]):
        r, c = divmod(i, cols)
        ax = fig.add_axes([box.x0 + c * (w + gap_x), top - h - r * (h + gap_y), w, h])
        plane_grid(ax, obs[..., i], colours[i], constant=i >= 7)
        ax.text(obs.shape[0] / 2, obs.shape[0] + 1.0, caps(names[i]), ha="center", va="top", fontsize=5.8, color=SOFT)
        axes.append(ax)
    # The spare card carries the colour key.
    ax = fig.add_axes([box.x0 + 3 * (w + gap_x), top - h - 3 * (h + gap_y), w, h])
    ax.axis("off")
    for j, (colour, text) in enumerate(((WALL, "bedrock"), (INK, "agent, tree, stone, table"), (RICH, "iron"), (POOR, "coal"))):
        ax.add_patch(Rectangle((0.02, 0.86 - 0.2 * j), 0.1, 0.1, color=colour, transform=ax.transAxes, clip_on=False))
        ax.text(0.17, 0.91 - 0.2 * j, caps(text), transform=ax.transAxes, fontsize=5.4, color=SOFT, va="center")
    ax.text(0.02, 0.06, "1 = present in that cell", transform=ax.transAxes, fontsize=6.2, color=SOFT, va="bottom")
    return axes


# ---------------------------------------------------------------- panel (c)
def panel_tree(ax, tex, meta, ratio: float) -> None:
    blocks, icons = tex["blocks"], tex["icons"]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10 * ratio)
    ax.set_aspect("equal")
    ax.axis("off")

    def icon(x, y, image, size=1.2):
        ax.imshow(image, extent=(x - size / 2, x + size / 2, y - size / 2, y + size / 2), interpolation="nearest", zorder=3)

    def arrow(x0, y0, x1, y1, text=None, above=True):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=SOFT, zorder=2, shrinkA=0, shrinkB=0))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.28 if above else -0.32), caps(text), ha="center", va="bottom" if above else "top", fontsize=5.8, color=SOFT)

    def caption(x, y, text, colour=INK, size=7.4):
        ax.text(x, y, text, ha="center", va="top", fontsize=size, color=colour, linespacing=1.2)

    top, mid, bottom = 10 * ratio - 1.5, 10 * ratio - 4.9, 10 * ratio - 7.6
    # Shared trunk.
    for k in range(3):
        icon(1.0 + 0.34 * k, top + 0.17 * k, blocks[int(Block.TREE)], 1.0)
    caption(1.35, top - 0.85, "cut three trees\n3 wood")
    arrow(2.2, top, 3.35, top)
    icon(4.0, top, blocks[int(Block.CRAFTING_TABLE)])
    caption(4.0, top - 0.85, "place a table\n2 wood")
    arrow(4.7, top, 5.85, top)
    icon(6.5, top, icons["wood_pickaxe"])
    caption(6.5, top - 0.85, "wood pickaxe\n1 wood, at the table")
    # Coal branch: straight on.
    arrow(7.2, top, 8.35, top)
    icon(9.0, top, blocks[int(Block.COAL)])
    caption(9.0, top - 0.85, f"coal\nvalue {meta['values'][1]}", POOR)
    ax.text(9.0, top + 0.85, caps("coal chain"), ha="center", va="bottom", fontsize=6.2, color=POOR)
    # Iron branch: down and along.
    arrow(6.5, top - 1.75, 6.5, mid + 0.75)
    icon(1.0, mid, blocks[int(Block.TREE)], 1.0)
    ax.text(1.75, mid, "+", ha="center", va="center", fontsize=11, color=SOFT)
    icon(2.5, mid, blocks[int(Block.STONE)], 1.0)
    caption(1.75, mid - 0.85, "a fourth tree, a stone deposit\nstone needs the wood pickaxe")
    arrow(3.2, mid, 5.85, mid, "back to the table")
    icon(6.5, mid, icons["stone_pickaxe"])
    caption(6.5, mid - 0.85, "stone pickaxe\n1 wood + 1 stone")
    arrow(7.2, mid, 8.35, mid)
    icon(9.0, mid, blocks[int(Block.IRON)])
    caption(9.0, mid - 0.85, f"iron\nvalue {meta['values'][0]}", RICH)
    ax.text(9.0, mid + 0.85, caps("iron chain"), ha="center", va="bottom", fontsize=6.2, color=RICH)
    # The rule, in words.
    v = meta["values"]
    ax.text(
        0.2,
        bottom,
        "The expert plans both chains from the field, prices each at value minus\n"
        f"{meta['step_penalty']} per action, and takes the better one. Iron is worth {v[0] - v[1]:.1f} more than\n"
        f"coal, so its chain wins whenever its detour costs fewer than {meta['threshold']:.0f} actions.\n"
        "Every field is a different price for the same two goods.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=INK,
        linespacing=1.45,
    )


# ---------------------------------------------------------------- panel (d)
def panel_gap(ax, data, meta) -> None:
    gap = np.asarray(data["gap"])
    threshold = meta["threshold"]
    lo, hi = np.percentile(gap, [0.5, 99.5])
    bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1)
    ax.hist(gap[gap < threshold], bins=bins, color=RICH, alpha=0.8, lw=0)
    ax.hist(gap[gap >= threshold], bins=bins, color=POOR, alpha=0.65, lw=0)
    ax.axvline(threshold, color=INK, lw=1.2)
    top = ax.get_ylim()[1]
    ax.text(threshold - 0.6, top * 0.98, caps(f"expert takes iron · {meta['expert_takes_iron']:.0%}"), ha="right", va="top", fontsize=6.4, color=RICH)
    ax.text(threshold + 0.6, top * 0.98, caps(f"expert takes coal · {1 - meta['expert_takes_iron']:.0%}"), ha="left", va="top", fontsize=6.4, color=POOR)
    ax.text(threshold + 0.6, top * 0.88, caps(f"threshold {threshold:.0f}"), ha="left", va="top", fontsize=6.4, color=INK)
    this = meta["cost_iron"] - meta["cost_coal"]
    ax.plot(this, 0, marker="^", color=INK, ms=7, clip_on=False, zorder=5)
    ax.annotate(
        "the field in (a)", (this, top * 0.01), (this - 7, top * 0.32), fontsize=7.2, color=SOFT,
        arrowprops=dict(arrowstyle="-", color=LINE, lw=0.8), ha="center",
    )
    ax.set_xlabel(caps("extra actions the iron chain costs over the coal chain"), fontsize=6.8, labelpad=6)
    ax.set_ylabel(caps(f"held-out fields, of {meta['n_fields']}"), fontsize=6.8, labelpad=6)
    ax.grid(True, axis="y")
    ax.tick_params(labelsize=7.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "craftax" / "craft" / "fig1.npz")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "craftax" / "fig_craft_task")
    args = parser.parse_args()
    data = np.load(args.data)
    meta = json.loads(args.data.with_suffix(".json").read_text())
    tex = textures()
    v = meta["values"]

    fw, fh = 10.6, 12.75
    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.985, top=0.95, bottom=0.03, wspace=0.16, hspace=0.13)
    cells = {k: gs[i, j].get_position(fig) for k, (i, j) in zip("abcd", ((0, 0), (0, 1), (1, 0), (1, 1)))}
    # Every panel's content is one rectangle: the map's square plus the key band under it.
    band_in = 0.85
    side_in = min(cells["a"].width * fw, cells["a"].height * fh - band_in)
    side_x, side_y, band_y = side_in / fw, side_in / fh, band_in / fh
    content = {k: mpl.transforms.Bbox.from_bounds(b.x0, b.y1 - side_y - band_y, side_x, side_y + band_y) for k, b in cells.items()}

    box = content["a"]
    ax_a = fig.add_axes([box.x0, box.y1 - side_y, side_x, side_y])
    frame = panel_field(fig, ax_a, data, meta, tex)
    header(
        fig, box, "a", "One held-out field, and what the policy did on it",
        f"iron {v[0]}, coal {v[1]}, {meta['step_penalty']} per action · iron chain {meta['cost_iron']}, coal chain {meta['cost_coal']}: "
        f"a detour of {meta['cost_iron'] - meta['cost_coal']}, under {meta['threshold']:.0f}, so iron",
    )

    box = content["b"]
    panel_planes(fig, box, data, meta, frame)
    header(
        fig, box, "b", f"What the policy sees at step {frame}",
        "table down, wood pickaxe in hand · map planes, then facing and inventory as constant planes (counts / 4)"[:-12],
    )

    box = content["c"]
    ax_c = fig.add_axes([box.x0, box.y0, box.width, box.height])
    panel_tree(ax_c, tex, meta, ratio=(box.height * fh) / (box.width * fw))
    header(fig, box, "c", "The tech tree the expert plans over", "the iron chain is the coal chain plus a detour for its ingredients")

    box = content["d"]
    ax_d = fig.add_axes([box.x0 + 0.05, box.y0 + 0.045, box.width - 0.055, box.height - 0.06])
    panel_gap(ax_d, data, meta)
    header(
        fig, box, "d", "The decision rule over the held-out fields",
        f"take iron iff its detour costs fewer than ({v[0]} − {v[1]}) / {meta['step_penalty']} = {meta['threshold']:.0f} actions",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()
