"""The crafting task in one figure, drawn from a recorded rollout of the trained policy.

    uv run python figures/make_craft_fig1.py [--data figures/data/craftax/craft/fig1.npz] [--out figures/craftax/fig_craft_task]

Four panels. (a) One held-out field with the engine's own block textures and the
policy's closed-loop rollout drawn over the expert's two plans. (b) The observation
the policy gets at one step of that rollout: seven map planes and eight planes that
are constant over the grid (facing, inventory). (c) The tech tree the expert plans
over. (d) The cost gap between the two chains over the held-out fields and the
threshold the values set. Everything is read from ``fig1.npz`` / ``fig1.json``,
written by ``figures/craft_fig1_data.py`` on a machine holding the checkpoint.

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
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from goalmisgen.craftax import simulate  # noqa: E402
from goalmisgen.craftax.blocks import BEDROCK, Action, Block  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6f5f9c"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9992"
SURFACE = "#fcfcfb"
PX = 64

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 160,
    }
)

DIR = {int(Action.UP): (-1, 0), int(Action.DOWN): (1, 0), int(Action.LEFT): (0, -1), int(Action.RIGHT): (0, 1)}
HALO = [pe.Stroke(linewidth=3.2, foreground="white", alpha=0.9), pe.Normal()]


def textures() -> dict:
    from craftax.craftax_classic.constants import load_all_textures

    tex = load_all_textures()[PX]
    blocks = np.asarray(tex["block_textures"], dtype=np.float32) / 255.0
    # Bedrock: the furnace block in the engine, drawn as darkened stone.
    blocks = blocks.copy()
    stone = blocks[int(Block.STONE)]
    blocks[int(BEDROCK)] = 0.45 * stone.mean(axis=-1, keepdims=True) + 0.02
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


def events(actions, positions, facings, inventories) -> list[tuple[str, tuple[int, int], int]]:
    """What happened where along the rollout: (label, cell, step)."""
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
                out.append(("ore mined", faced, t))
        elif action == int(Action.PLACE_TABLE):
            out.append(("table placed", faced, t))
        elif action == int(Action.MAKE_WOOD_PICKAXE):
            out.append(("wood pickaxe made", pos, t))
        elif action == int(Action.MAKE_STONE_PICKAXE):
            out.append(("stone pickaxe made", pos, t))
    return out


def centre(cell) -> tuple[float, float]:
    return cell[1] + 0.5, cell[0] + 0.5


# ---------------------------------------------------------------- panel (a)
def panel_field(ax, data, meta, tex) -> int:
    tiles = data["tiles"]
    start = data["agent_start"]
    image = field_image(tiles, tex["blocks"])
    final = data["maps"][-1]
    for r, c in zip(*np.nonzero(final == int(Block.CRAFTING_TABLE))):
        paste(image, np.concatenate([tex["blocks"][int(Block.CRAFTING_TABLE)], np.ones((PX, PX, 1))], -1), r, c, alpha=0.55)
    paste(image, tex["players"][int(data["start_direction"]) - 1], int(start[0]), int(start[1]))
    size = tiles.shape[0]
    ax.imshow(image, extent=(0, size, size, 0), interpolation="nearest")

    iron = walk(tiles, start, data["start_direction"], data["plan_iron"])
    coal = walk(tiles, start, data["start_direction"], data["plan_coal"])
    rollout = [tuple(int(v) for v in p) for p in data["positions"]]
    for path, colour, width, alpha, style, label in (
        (iron, ORANGE, 7.0, 0.35, "-", f"expert's iron plan, {len(data['plan_iron'])} actions"),
        (coal, BLUE, 1.6, 0.9, (0, (3, 2)), f"expert's coal plan, {len(data['plan_coal'])} actions"),
    ):
        shift = 0.0 if width >= 3 else 0.2  # the coal plan shares the trunk; draw it beside the rollout, not under it
        xs, ys = zip(*[(x + shift, y + shift) for x, y in (centre(p) for p in path)])
        line = ax.plot(xs, ys, color=colour, lw=width, alpha=alpha, ls=style, solid_capstyle="round", label=label)
        if width < 3:
            line[0].set_path_effects(HALO)
    xs, ys = zip(*[centre(p) for p in rollout])
    ax.plot(xs, ys, color=INK, lw=1.4, solid_capstyle="round", label=f"policy's rollout, {len(rollout) - 1} actions")[
        0
    ].set_path_effects(HALO)

    marks = events(data["actions"], data["positions"], data["facings"], data["inventories"])
    seen: dict[tuple[int, int], int] = {}
    key = []
    for k, (label, cell, t) in enumerate(marks, 1):
        x, y = centre(cell)
        n = seen.get(cell, 0)
        seen[cell] = n + 1
        x += 0.33 * n
        y -= 0.33 * n
        ax.plot(x, y, marker="o", ms=9, color="white", mec=INK, mew=0.8, zorder=5)
        ax.text(x, y, str(k), ha="center", va="center", fontsize=6, color=INK, zorder=6, fontweight="bold")
        if label == "ore mined":
            label = ("iron" if int(data["tiles"][cell]) == int(Block.IRON) else "coal") + " mined"
        key.append(f"{k}  {label}, step {t + 1}")
    frame = next(t + 1 for label, _, t in marks if label == "wood pickaxe made")
    fx, fy = centre(rollout[frame])
    ax.plot(fx, fy, marker="s", ms=13, mfc="none", mec=VIOLET, mew=1.6, zorder=7)
    ax.text(fx + 0.6, fy + 0.1, "(b)", color=VIOLET, va="center", fontsize=8, fontstyle="italic", zorder=7).set_path_effects(HALO)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.004), ncol=3, fontsize=7.2, handlelength=2.4, columnspacing=1.4)
    ax.text(
        1.01,
        1.0,
        "\n".join(key),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        color=INK2,
        linespacing=1.55,
    )
    v = meta["values"]
    ax.set_title(
        f"held-out field {meta['field']}: iron worth {v[0]}, coal worth {v[1]}, {meta['step_penalty']} per action\n"
        f"iron chain {meta['cost_iron']} actions, coal chain {meta['cost_coal']}: a detour of {meta['cost_iron'] - meta['cost_coal']} < {meta['threshold']:.0f}, so iron",
        fontsize=8.2,
        loc="left",
        x=0.06,
        color=INK2,
    )
    return frame


# ---------------------------------------------------------------- panel (b)
def panel_planes(fig, gs, data, meta, frame) -> None:
    obs = data["observations"][frame]
    names = meta["channels"]
    n = obs.shape[-1]
    sub = gs.subgridspec(1, n, wspace=0.12)
    for i in range(n):
        ax = fig.add_subplot(sub[i])
        plane = obs[..., i]
        ax.imshow(plane, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_anchor("N")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(MUTED)
            spine.set_linewidth(0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(names[i].replace(" ", "\n", 1), fontsize=5.8, color=INK2, pad=2, linespacing=1.0)
        if i >= 7:
            value = float(plane[0, 0])
            ax.text(
                0.5,
                0.5,
                f"{value:g}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.5,
                color=INK if value < 0.5 else "white",
            )
    fig.text(
        gs.get_position(fig).x0,
        gs.get_position(fig).y1 + 0.036,
        f"the observation at step {frame}, table down and wood pickaxe in hand: seven map planes,\n"
        "then facing and inventory as planes constant over the grid (counts / 4)",
        fontsize=8.2,
        color=INK2,
        ha="left",
        va="bottom",
        linespacing=1.3,
    )


# ---------------------------------------------------------------- panel (c)
def panel_tree(ax, tex, meta) -> None:
    blocks, icons = tex["blocks"], tex["icons"]
    ax.set_xlim(0, 10)
    ax.set_ylim(0.35, 3.6)
    ax.axis("off")

    def icon(x, y, image, size=0.9):
        ax.imshow(image, extent=(x - size / 2, x + size / 2, y - size / 2, y + size / 2), interpolation="nearest", zorder=3)

    def arrow(x0, y0, x1, y1, text=None, colour=INK2, dy=0.22):
        ax.add_patch(
            FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=colour, zorder=2)
        )
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.6, color=colour)

    def caption(x, y, text, colour=INK):
        ax.text(x, y, text, ha="center", va="top", fontsize=6.8, color=colour, linespacing=1.15)

    top, bottom = 3.1, 1.1
    # Shared trunk.
    for k in range(3):
        icon(0.9 + 0.32 * k, top + 0.14 * k, blocks[int(Block.TREE)], 0.8)
    caption(1.2, top - 0.6, "cut 3 trees\n(3 wood)")
    arrow(1.9, top, 2.85, top)
    icon(3.3, top, blocks[int(Block.CRAFTING_TABLE)])
    caption(3.3, top - 0.6, "place table\n(2 wood)")
    arrow(3.8, top, 4.75, top)
    icon(5.2, top, icons["wood_pickaxe"])
    caption(5.2, top - 0.6, "wood pickaxe\n(1 wood, at the table)")
    # Coal branch.
    arrow(5.7, top, 8.05, top, "walk to the coal")
    icon(8.5, top, blocks[int(Block.COAL)])
    caption(8.5, top - 0.6, f"mine coal\nvalue {meta['values'][1]}", ORANGE)
    # Iron branch.
    arrow(5.2, top - 0.95, 5.2, bottom + 0.55)
    icon(1.0, bottom, blocks[int(Block.TREE)], 0.8)
    ax.text(1.55, bottom, "+", ha="center", va="center", fontsize=9, color=INK2)
    icon(2.1, bottom, blocks[int(Block.STONE)], 0.8)
    caption(1.55, bottom - 0.55, "a 4th tree and a stone deposit\n(stone needs the wood pickaxe)")
    arrow(2.65, bottom, 4.75, bottom, "back beside the table")
    icon(5.2, bottom, icons["stone_pickaxe"])
    caption(5.2, bottom - 0.55, "stone pickaxe\n(1 wood + 1 stone)")
    arrow(5.7, bottom, 8.05, bottom, "walk to the iron")
    icon(8.5, bottom, blocks[int(Block.IRON)])
    caption(8.5, bottom - 0.55, f"mine iron\nvalue {meta['values'][0]}", ORANGE)
    ax.set_title(
        "the tech tree the expert plans over: the iron chain is the coal chain plus a detour for its ingredients",
        fontsize=8.2,
        loc="left",
        color=INK2,
    )


# ---------------------------------------------------------------- panel (d)
def panel_gap(ax, data, meta) -> None:
    gap = np.asarray(data["gap"])
    threshold = meta["threshold"]
    lo, hi = np.percentile(gap, [0.5, 99.5])
    bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1)
    ax.hist(gap[gap < threshold], bins=bins, color=ORANGE, alpha=0.75, lw=0, label="expert takes iron")
    ax.hist(gap[gap >= threshold], bins=bins, color=BLUE, alpha=0.55, lw=0, label="expert takes coal")
    ax.axvline(threshold, color=INK, lw=1.2)
    ax.text(threshold + 0.4, ax.get_ylim()[1] * 0.97, f"threshold {threshold:.0f}", va="top", fontsize=7.5, color=INK)
    this = meta["cost_iron"] - meta["cost_coal"]
    ax.plot(this, 0, marker="^", color=INK, ms=7, clip_on=False, zorder=5)
    ax.annotate("the field in (a)", (this, 0), (this - 6, ax.get_ylim()[1] * 0.3), fontsize=7.2, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), ha="center")
    ax.set_xlabel("extra actions the iron chain costs over the coal chain")
    ax.set_ylabel(f"held-out fields (of {meta['n_fields']})")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=7.5)
    v = meta["values"]
    ax.set_title(
        f"the rule: iron iff its detour costs under ({v[0]} − {v[1]}) / {meta['step_penalty']} = {threshold:.0f} actions; "
        f"iron on {meta['expert_takes_iron']:.0%} of fields",
        fontsize=8.2,
        loc="left",
        color=INK2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "craftax" / "craft" / "fig1.npz")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "craftax" / "fig_craft_task")
    args = parser.parse_args()
    data = np.load(args.data)
    meta = json.loads(args.data.with_suffix(".json").read_text())
    tex = textures()

    fig = plt.figure(figsize=(12.4, 6.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.92], height_ratios=[1.0, 0.17], hspace=0.3, wspace=0.34,
                          left=0.03, right=0.99, top=0.95, bottom=0.05)
    ax_a = fig.add_subplot(gs[0, 0])
    frame = panel_field(ax_a, data, meta, tex)
    panel_planes(fig, gs[1, 0], data, meta, frame)
    right = gs[:, 1].subgridspec(2, 1, height_ratios=[0.42, 1.0], hspace=0.28)
    panel_tree(fig.add_subplot(right[0]), tex, meta)
    panel_gap(fig.add_subplot(right[1]), data, meta)
    for ax, letter, dx in ((ax_a, "a", 0.0), (fig.axes[-2], "c", -0.045), (fig.axes[-1], "d", -0.1)):
        ax.text(dx, 1.0, f"({letter})", transform=ax.transAxes, fontsize=11, fontstyle="italic", color=INK, va="bottom")
    strip = gs[1, 0].get_position(fig)
    fig.text(strip.x0 - 0.005, strip.y1 + 0.036, "(b)", fontsize=11, fontstyle="italic", color=INK, va="bottom", ha="right")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()
