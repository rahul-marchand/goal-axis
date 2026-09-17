"""The value-axis figure across tasks: the maze route model beside the two Craftax tasks.

    uv run python figures/make_craftax_axis_figure.py [--maze figures/data/bc] [--data figures/data/craftax]
        [--layout single|all] [--out figures/craftax/fig_craftax_value_axis]

``single`` (the default) is the write-up's figure: the crafting base trained
from scratch, alone. ``all`` is the appendix version, one panel per base agent,
maze route model first. Every panel is the same plot as the blog's value-axis figure: the
decision threshold (extra actions the policy will spend for the richer goal)
against the change in one goal's value, for agents fine-tuned at that value
(filled, solid) and for the frozen base with the fitted axis written into its
weights and that arm held out of the fit (hollow, dashed). Red sweeps the
richer goal's value, blue the poorer goal's. The grey dashed line is the
expert's own exchange rate; the horizontal line at zero is preference
reversal. Bands are the range over seeds where more than one exists. Text in
each panel: the base's competence, the null arm's threshold, the split-half
reliability of the axis, and cos(axis_0, axis_1) where 028 has been run.
Everything comes from ``value_axis.<base>.o{0,1}.json`` and
``value_or_gap.<base>.json``; nothing is typed in.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RICH, POOR = "#a63d2f", "#2f5a78"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#9a9992", "#c9c7bf"
SURFACE = "#fcfcfb"
OFFSETS = (0.05, 0.1, 0.2, 0.3, 0.45)
"""Arm offsets drawn; the maze runs also carry a dense ladder near 0.4 from the dose study."""

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


def load_panel(directory: Path, base: str) -> dict:
    """All seeds of one base: {sweep: [payload per seed]}, and the 028 gap payloads."""
    pattern = re.compile(rf"value_axis\.{re.escape(base)}\.s\d+\.(o[01])\.json$")
    sweeps: dict[str, list[dict]] = {}
    for path in sorted(directory.glob(f"value_axis.{base}.s*.o*.json")):
        match = pattern.search(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text())
        if payload.get("behaviour", {}).get("arms"):
            sweeps.setdefault(match.group(1), []).append(payload)
    gaps = [json.loads(p.read_text()) for p in sorted(directory.glob(f"value_or_gap.{base}.s*.json"))]
    return {"sweeps": sweeps, "gaps": gaps}


def series(payloads: list[dict], kind: str) -> list[tuple[float, float, float, float]]:
    """(offset, mean, low, high) over seeds for the fine-tuned arm or the held-out write."""
    rows: dict[float, list[float]] = {}
    for payload in payloads:
        for key, arm in payload["behaviour"]["arms"].items():
            offset = float(key)
            if not any(abs(abs(offset) - o) < 1e-9 for o in OFFSETS):
                continue
            value = arm[kind]["indifference"]
            if value is not None and np.isfinite(value):
                rows.setdefault(offset, []).append(float(value))
    out = []
    for offset in sorted(rows):
        values = rows[offset]
        out.append((offset, float(np.mean(values)), float(min(values)), float(max(values))))
    return out


def expert_line(payloads: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = sorted({(float(k), float(a["expected"])) for p in payloads for k, a in p["behaviour"]["arms"].items()})
    xs, ys = zip(*points)
    return np.asarray(xs), np.asarray(ys)


def draw_panel(ax, panel: dict, title: str, ylim: tuple[float, float], goals=("richer goal", "poorer goal"), stats: bool = True) -> None:
    sweeps = panel["sweeps"]
    ax.axhline(0, color=INK2, lw=0.8, alpha=0.55)
    ax.grid(True, axis="y")
    for sweep, colour in (("o0", RICH), ("o1", POOR)):
        payloads = sweeps.get(sweep)
        if not payloads:
            continue
        xs, ys = expert_line(payloads)
        ax.plot(xs, ys, color=LINE, lw=1.3, ls=(0, (4, 3)), zorder=1)
        for kind, filled, dashed in (("arm", True, False), ("written", False, True)):
            rows = series(payloads, kind)
            if not rows:
                continue
            x = [r[0] for r in rows]
            m = [np.clip(r[1], *ylim) for r in rows]
            lo = [np.clip(r[2], *ylim) for r in rows]
            hi = [np.clip(r[3], *ylim) for r in rows]
            if len(payloads) > 1:
                ax.fill_between(x, lo, hi, color=colour, alpha=0.13, lw=0)
            ax.plot(x, m, color=colour, lw=1.6, ls=(0, (4, 3)) if dashed else "-", zorder=3)
            ax.plot(
                x, m, ls="none", marker="o", ms=4.2, mfc=colour if filled else SURFACE, mec=colour, mew=1.3, zorder=4
            )
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(*ylim)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_yticks([-10, 0, 10, 20, 30])

    text = []
    first = next((p for s in ("o0", "o1") for p in sweeps.get(s, [])), None)
    if first is not None and stats:
        bases = [p["behaviour"]["base"] for s in sweeps.values() for p in s]
        reached = np.mean([b["reached"] for b in bases])
        optimal = np.mean([b["chose_optimal"] for b in bases])
        text.append(f"base: reaches {reached:.0%}, chooses optimally {optimal:.0%}")
        nulls = [p["behaviour"]["null"]["indifference"] for s in sweeps.values() for p in s if p["behaviour"].get("null")]
        expected = first["behaviour"]["arms"]["+0.10"]["expected"] - 2.0 if "+0.10" in first["behaviour"]["arms"] else None
        if nulls:
            text.append(f"null arm τ = {np.mean(nulls):.1f}" + (f" (expert {expected:.0f})" if expected is not None else ""))
        for sweep, name in (("o0", goals[0]), ("o1", goals[1])):
            if sweeps.get(sweep):
                rel = np.mean([p["reliability"] for p in sweeps[sweep] if p.get("reliability") is not None])
                slope = np.mean([p["behaviour"]["slope_arms"] for p in sweeps[sweep]])
                written = np.mean([p["behaviour"]["slope_written"] for p in sweeps[sweep]])
                text.append(f"{name} moved: slope {slope:+.0f} / written {written:+.0f}, reliability {rel:.2f}")
        cos = [g["cos"] for g in panel["gaps"] if g.get("cos") is not None]
        if cos:
            text.append(f"cos(axis_0, axis_1) = {np.mean(cos):+.2f}")
    ax.text(0.03, 0.03, "\n".join(text), transform=ax.transAxes, va="bottom", ha="left", fontsize=7.0, color=INK2, linespacing=1.4)
    ax.set_title(title, loc="left", fontsize=9.5, x=0.055)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maze", type=Path, default=Path(__file__).parent / "data" / "bc")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "craftax")
    parser.add_argument("--layout", choices=("single", "all"), default="single")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(__file__).parent / "craftax" / ("fig_craftax_value_axis" if args.layout == "single" else "fig_craftax_value_axis_all")
    if args.layout == "single":
        draw_single(load_panel(args.data / "craft", "cxcraft15t"), out)
        return

    panels = [
        ("maze, route model (bcnv11)", load_panel(args.maze, "bcnv11")),
        ("ore field in the engine, route model (cxnv15c)", load_panel(args.data / "ore", "cxnv15c")),
        ("crafting, warm start (cxcraft15.s1t)", load_panel(args.data / "craft", "cxcraft15")),
        ("crafting, from scratch (cxcraft15t.s1)", load_panel(args.data / "craft", "cxcraft15t")),
    ]
    # cxcraft15.s1t's files are value_axis.cxcraft15.s1t.o*.json: seed token "s1t".
    panels[2] = (panels[2][0], load_panel_seed(args.data / "craft", "cxcraft15", "s1t"))

    ylim = (-17, 32)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True, gridspec_kw={"hspace": 0.32, "wspace": 0.12})
    for k, (ax, (title, panel)) in enumerate(zip(axes.ravel(), panels)):
        draw_panel(ax, panel, title, ylim)
        ax.text(-0.01, 1.02, f"({'abcd'[k]})", transform=ax.transAxes, fontsize=11, fontstyle="italic", va="bottom", ha="right")
    for ax in axes[1]:
        ax.set_xlabel("change in the swept goal's value")
    fig.supylabel("threshold τ: extra actions the policy spends for the richer goal", fontsize=9, color=INK2, x=0.04)
    handles = [
        Line2D([], [], color=RICH, lw=1.6, label="richer goal's value moved"),
        Line2D([], [], color=POOR, lw=1.6, label="poorer goal's value moved"),
        Line2D([], [], color=INK2, lw=1.6, marker="o", ms=4.2, label="fine-tuned at that value"),
        Line2D([], [], color=INK2, lw=1.6, ls=(0, (4, 3)), marker="o", ms=4.2, mfc=SURFACE, mew=1.3, label="written along the axis, arm held out"),
        Line2D([], [], color=LINE, lw=1.3, ls=(0, (4, 3)), label="expert's exchange rate"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.8, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.png")


def draw_single(panel: dict, out: Path) -> None:
    """The write-up's figure at the blog's size: one base, both sweeps, nothing but the plot."""
    ylim = (-11, 30)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    draw_panel(ax, panel, "", ylim, stats=False)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("change in the swept goal's value", fontsize=8)
    ax.set_ylabel("threshold τ (extra actions for iron)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    handles = [
        Line2D([], [], color=RICH, lw=1.6, label="iron's value moved"),
        Line2D([], [], color=POOR, lw=1.6, label="coal's value moved"),
        Line2D([], [], color=INK2, lw=1.6, marker="o", ms=4, label="fine-tuned"),
        Line2D([], [], color=INK2, lw=1.6, ls=(0, (4, 3)), marker="o", ms=4, mfc=SURFACE, mew=1.2, label="written, arm held out"),
        Line2D([], [], color=LINE, lw=1.3, ls=(0, (4, 3)), label="expert"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=6.4, ncol=3, handlelength=2.0, columnspacing=1.2, labelspacing=0.3, borderaxespad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.08)
    print(f"wrote {out}.png")


def load_panel_seed(directory: Path, base: str, seed: str) -> dict:
    """One base whose seed token is not ``s<digit>`` (a warm-started continuation)."""
    sweeps: dict[str, list[dict]] = {}
    for sweep in ("o0", "o1"):
        path = directory / f"value_axis.{base}.{seed}.{sweep}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("behaviour", {}).get("arms"):
                sweeps[sweep] = [payload]
    path = directory / f"value_or_gap.{base}.{seed}.json"
    gaps = [json.loads(path.read_text())] if path.exists() else []
    return {"sweeps": sweeps, "gaps": gaps}


if __name__ == "__main__":
    main()
