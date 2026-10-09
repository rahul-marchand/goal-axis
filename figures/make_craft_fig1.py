"""The crafting task in one figure, drawn from a recorded rollout of the trained policy.

    uv run python figures/make_craft_fig1.py [--data figures/data/craftax/craft/fig1.npz] [--out figures/craftax/fig_craft_task]

Three panels, letters only, the rest in the caption. (a) One held-out field
with the engine's own block textures and the policy's closed-loop rollout.
(b) The observation the policy gets at one step of that rollout: seven map
planes and eight planes constant over the grid, three wide and five tall down
the right-hand column. (c) The two chains the expert prices. ``--with-gap``
adds the cost-gap histogram as a fourth panel. Everything is read from
``fig1.npz`` / ``fig1.json``, written by ``figures/craft_fig1_data.py`` on a
machine holding the checkpoint.

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


def letter(fig, box, text: str) -> None:
    fig.text(box.x0, box.y1 + 0.008, f"({text})", fontsize=11, fontstyle="italic", family=SERIF, color=INK, va="bottom")


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
        data["actions"],
        data["positions"],
        data["facings"],
        data["inventories"],
        data["tiles"],
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

    # Only the policy's own route is drawn: on this field it is the expert's iron plan, action for action.
    rollout = [tuple(int(v) for v in p) for p in data["positions"]]
    xs, ys = zip(*[centre(p) for p in rollout])
    ax.plot(xs, ys, color=INK, lw=1.6, solid_capstyle="round", solid_joinstyle="round")[0].set_path_effects(HALO)

    marks = events(data)
    frame = next(t + 1 for label, _, t in marks if label == "wood pickaxe")
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return frame


# ---------------------------------------------------------------- panel (b)
def plane_grid(ax, plane: np.ndarray, colour: str, constant: bool) -> None:
    size = plane.shape[0]
    ax.add_patch(Rectangle((0, 0), size, size, color=FLOOR, lw=0))
    if constant:
        value = float(plane[0, 0])
        ax.add_patch(Rectangle((0, 0), size, size, color=colour, alpha=value, lw=0))
        ax.text(size / 2, size / 2, f"{value:g}", ha="center", va="center", fontsize=7, color=INK if value < 0.5 else "white")
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


def panel_planes(fig, box, data, meta, frame, cols: int = 3, rows: int = 5) -> None:
    """Fifteen planes as a grid of lattice cards filling ``box`` (figure coordinates)."""
    obs = data["observations"][frame]
    names = meta["channels"]
    colours = [WALL, INK, INK, INK, INK, RICH, POOR] + [INK] * 8
    gap_x, gap_y = 0.012, 0.022
    w = (box.width - gap_x * (cols - 1)) / cols
    h = min(w * (fig.get_figwidth() / fig.get_figheight()), (box.height - gap_y * rows - 0.014) / rows)
    w = h * (fig.get_figheight() / fig.get_figwidth())
    top = box.y1
    axes = []
    for i in range(obs.shape[-1]):
        r, c = divmod(i, cols)
        ax = fig.add_axes([box.x0 + c * (w + gap_x), top - h - r * (h + gap_y), w, h])
        plane_grid(ax, obs[..., i], colours[i], constant=i >= 7)
        ax.text(obs.shape[0] / 2, obs.shape[0] + 1.0, caps(names[i]), ha="center", va="top", fontsize=4.8, color=SOFT)
        axes.append(ax)
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
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=SOFT, zorder=2, shrinkA=0, shrinkB=0
            )
        )
        if text:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + (0.28 if above else -0.32),
                caps(text),
                ha="center",
                va="bottom" if above else "top",
                fontsize=5.8,
                color=SOFT,
            )

    def caption(x, y, text, colour=INK, size=7.4):
        ax.text(x, y, text, ha="center", va="top", fontsize=size, color=colour, linespacing=1.2)

    top, mid = 10 * ratio - 1.6, 10 * ratio - 6.0
    # Shared trunk.
    for k in range(3):
        icon(1.0 + 0.34 * k, top + 0.17 * k, blocks[int(Block.TREE)], 1.0)
    caption(1.35, top - 0.85, "3 trees")
    arrow(2.2, top, 3.3, top)
    icon(4.0, top, blocks[int(Block.CRAFTING_TABLE)])
    caption(4.0, top - 0.85, "table\n2 wood")
    arrow(4.7, top, 5.8, top)
    icon(6.5, top, icons["wood_pickaxe"])
    caption(6.5, top - 0.85, "wood pickaxe\n1 wood")
    arrow(7.2, top, 8.3, top)
    icon(9.0, top, blocks[int(Block.COAL)])
    caption(9.0, top - 0.85, f"coal\nworth {meta['values'][1]}", POOR)
    # Iron branch: down and along.
    arrow(6.5, top - 1.75, 6.5, mid + 0.75)
    icon(1.0, mid, blocks[int(Block.TREE)], 1.0)
    ax.text(1.75, mid, "+", ha="center", va="center", fontsize=10, color=SOFT)
    icon(2.5, mid, blocks[int(Block.STONE)], 1.0)
    caption(1.75, mid - 0.85, "a 4th tree, a stone")
    arrow(3.2, mid, 5.8, mid)
    icon(6.5, mid, icons["stone_pickaxe"])
    caption(6.5, mid - 0.85, "stone pickaxe\n1 wood + 1 stone")
    arrow(7.2, mid, 8.3, mid)
    icon(9.0, mid, blocks[int(Block.IRON)])
    caption(9.0, mid - 0.85, f"iron\nworth {meta['values'][0]}", RICH)


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
    ax.text(
        threshold - 0.7,
        top * 0.98,
        caps(f"iron · {meta['expert_takes_iron']:.0%}"),
        ha="right",
        va="top",
        fontsize=6.2,
        color=RICH,
    )
    ax.text(
        threshold + 0.7,
        top * 0.98,
        caps(f"coal · {1 - meta['expert_takes_iron']:.0%}"),
        ha="left",
        va="top",
        fontsize=6.2,
        color=POOR,
    )
    ax.text(threshold + 0.7, top * 0.88, caps(f"threshold {threshold:.0f}"), ha="left", va="top", fontsize=6.2, color=INK)
    this = meta["cost_iron"] - meta["cost_coal"]
    ax.plot(this, 0, marker="^", color=INK, ms=6, clip_on=False, zorder=5)
    ax.text(this - 0.9, top * 0.04, "(a)", ha="right", va="bottom", fontsize=7, fontstyle="italic", family=SERIF, color=INK)
    ax.set_xlabel(caps("detour: extra actions for iron over coal"), fontsize=6.2, labelpad=8)
    ax.set_ylabel(caps("held-out fields"), fontsize=6.2, labelpad=6)
    ax.grid(True, axis="y")
    ax.tick_params(labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "craftax" / "craft" / "fig1.npz")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "craftax" / "fig_craft_task")
    parser.add_argument("--with-gap", action="store_true", help="also write the cost-gap histogram as <out>_gap")
    args = parser.parse_args()
    data = np.load(args.data)
    meta = json.loads(args.data.with_suffix(".json").read_text())
    tex = textures()

    fw, fh = 7.2, 6.7
    fig = plt.figure(figsize=(fw, fh))
    left, right, top, bottom, gap, vgap = 0.04, 0.985, 0.965, 0.03, 0.06, 0.075
    # Left column: the map over the chains, each half the height. Right column: the fifteen cards, full height.
    half_y = (top - bottom - vgap) / 2
    side_x = half_y * fh / fw
    box_a = mpl.transforms.Bbox.from_bounds(left, top - half_y, side_x, half_y)
    box_c = mpl.transforms.Bbox.from_bounds(left, bottom, side_x, half_y)
    box_b = mpl.transforms.Bbox.from_bounds(left + side_x + gap, bottom, right - left - side_x - gap, top - bottom)

    ax_a = fig.add_axes([box_a.x0, box_a.y0, box_a.width, box_a.height])
    frame = panel_field(fig, ax_a, data, meta, tex)
    letter(fig, box_a, "a")

    panel_planes(fig, box_b, data, meta, frame, cols=3, rows=5)
    letter(fig, box_b, "b")

    ax_c = fig.add_axes([box_c.x0, box_c.y0, box_c.width, box_c.height])
    panel_tree(ax_c, tex, meta, ratio=(box_c.height * fh) / (box_c.width * fw))
    letter(fig, box_c, "c")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", pad_inches=0.1)
    print(f"wrote {args.out}.png")
    if args.with_gap:
        fig_d, ax_d = plt.subplots(figsize=(4.8, 3.2))
        panel_gap(ax_d, data, meta)
        for ext in ("png", "pdf"):
            fig_d.savefig(f"{args.out}_gap.{ext}", bbox_inches="tight", pad_inches=0.1)
        print(f"wrote {args.out}_gap.png")


if __name__ == "__main__":
    main()
