"""The two core plots of the retargeting-cost study, as PNG and SVG files for notes.

    uv run python scripts/studies/retarget/plot_core.py [--data figures/src/retarget-pilot/data.json] [--out ~/retarget-plots]

1. ``metric``: the fraction of the behaviour change each agent pays to be
   retargeted, per new goal. 0 is a perfect dial (only the goal is named);
   1 is an agent that learns nothing reusable and pays for every changed route.
2. ``gradient-descent``: the goal-slot agent, which has a dial by construction,
   retargeted by two learners. Tuning only the slot reaches the best dial
   setting within a few demonstrations; fine-tuning every weight takes
   thousands, or does not get there by 16k.

Reads the rows ``retarget_report.py --data`` writes. The report page
(``figures/src/retarget-pilot``) has the full set of figures; these two are for
reading in isolation.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NULL_TAU = 10.0
STANDARD = "#2f5a78"
GOAL_SLOT = "#6f5f9c"
SOFT = "#5c5f6a"
GRID = "#e6e3dc"


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": SOFT,
            "axes.labelcolor": "#23252b",
            "xtick.color": SOFT,
            "ytick.color": SOFT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


class Rows:
    def __init__(self, rows):
        self.by_key = {(r["agent"], r["variant"], r["tau"]): r for r in rows}

    def get(self, agent, variant, tau):
        return self.by_key.get((agent, variant, tau))

    def taus(self, agent, variant):
        return sorted(t for a, v, t in self.by_key if a == agent and v == variant and t != NULL_TAU)

    def fraction_paid(self, agent, variant, tau):
        """The spec's cost (SDL less the null arm's, per route) over the size of the change.

        Also returns the per-token version: the two bracket the bias from the
        arms' routes differing in length and destination from the null arm's.
        """
        r = self.get(agent, variant, tau)
        change = r["ceiling_minus_null_raw"]
        return r["sdl_minus_null_raw"] / change, r["sdl_minus_null"] / change


def plot_metric(rows: Rows, out: pathlib.Path):
    series = [
        ("bcnv11.s1", "main", "Standard agent, fine-tune everything", STANDARD, "-", "o"),
        ("bcgvr11.s1", "main", "Goal-slot agent, fine-tune everything", GOAL_SLOT, "-", "o"),
        ("bcgvr11.s1", "dial only", "Goal-slot agent, tune the goal slot only", GOAL_SLOT, "--", "o"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvline(NULL_TAU, color=SOFT, lw=1, ls=":")
    ax.text(NULL_TAU + 0.2, 1.2, "agent's own goal", ha="left", va="top", color=SOFT, fontsize=8.5)
    ax.axhline(0, color=SOFT, lw=1)
    ax.axhline(1, color=SOFT, lw=1, ls="--")
    ax.text(19.9, 1.02, "learns nothing reusable", ha="right", va="bottom", color=SOFT, fontsize=8.5)
    ax.text(NULL_TAU + 0.2, -0.03, "perfect dial", ha="left", va="top", color=SOFT, fontsize=8.5)

    seeds = [s for s in ("bcnv11.s1", "bcnv11.s2", "bcnv11.s3") if rows.taus(s, "main")]
    taus = rows.taus("bcnv11.s1", "main")
    for side in (lambda t: t < NULL_TAU, lambda t: t > NULL_TAU):
        ts = [t for t in taus if side(t)]
        spread = [[rows.fraction_paid(s, "main", t)[0] for s in seeds] for t in ts]
        ax.fill_between(ts, [min(v) for v in spread], [max(v) for v in spread], color=STANDARD, alpha=0.15, lw=0)
    for agent, variant, name, colour, dash, mark in series:
        ts = rows.taus(agent, variant)
        if not ts:
            continue
        for side in (lambda t: t < NULL_TAU, lambda t: t > NULL_TAU):
            part = [t for t in ts if side(t)]
            pts = [rows.fraction_paid(agent, variant, t) for t in part]
            y = [p[0] for p in pts]
            ax.plot(part, y, color=colour, ls=dash, lw=2, marker=mark, ms=5, mfc="white" if dash == "--" else colour)
            ax.vlines(part, [min(p) for p in pts], [max(p) for p in pts], color=colour, lw=4, alpha=0.18)
        ax.plot([], [], color=colour, ls=dash, lw=2, marker=mark, ms=5, mfc="white" if dash == "--" else colour, label=name)
    ax.fill_between([], [], color=STANDARD, alpha=0.15, label="Standard agent, range over 3 trained seeds")
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.3, 1.25)
    ax.set_xticks([0.5, 2.5, 6.5, 8.5, 10, 12.5, 14.5, 18.5])
    ax.set_xticklabels(["0.5", "2.5", "6.5", "8.5", "10", "12.5", "14.5", "18.5"])
    ax.set_xlabel("new goal: exchange rate τ (steps the richer objective is worth)")
    ax.set_ylabel("fraction of the behaviour change paid")
    ax.set_title("Cost of retargeting, as a fraction of the behaviour change", loc="left", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0, -0.16), ncol=2)
    ax.text(
        0,
        -0.36,
        "Shaded bars: the null arm subtracted per route (dot) to per token. They differ because the new goal's "
        "routes differ in length; near τ 10 the change is small and that bias dominates.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=SOFT,
        va="top",
        wrap=True,
    )
    save(fig, out, "metric")


def plot_gradient_descent(rows: Rows, out: pathlib.Path, taus=(2.5, 6.5, 18.5)):
    agent = "bcgvr11.s1"
    fig, axes = plt.subplots(1, len(taus), figsize=(9.6, 3.4), sharey=False)
    for ax, tau in zip(axes, taus):
        for variant, name, dash in (("main", "fine-tune everything", "-"), ("dial only", "tune the goal slot only", "--")):
            r = rows.get(agent, variant, tau)
            if r is None:
                continue
            x = [max(t, 0.5) for t in r["bounds"]]
            y = [v - r["oracle_test_bits"] for v in r["test_bits"]]
            ax.plot(x, y, color=GOAL_SLOT, ls=dash, lw=2, label=name)
        ax.axhline(0, color=SOFT, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 4, 32, 256, 2048, 16384])
        ax.set_xticklabels(["0", "4", "32", "256", "2k", "16k"])
        ax.set_title(f"new goal τ {tau:g}", loc="left", fontsize=10.5)
        ax.set_xlabel("demonstrations seen")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("extra bits per route\nabove the best goal-slot setting")
    axes[-1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Goal-slot agent: the goal parameter exists, but fine-tuning every weight is slow to use it",
        x=0.01,
        ha="left",
        fontsize=11.5,
    )
    fig.tight_layout()
    save(fig, out, "gradient-descent")


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"-> {out / name}.{{png,svg}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("figures/src/retarget-pilot/data.json"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path.home() / "retarget-plots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    rows = Rows(json.loads(args.data.read_text())["rows"])
    plot_metric(rows, args.out)
    plot_gradient_descent(rows, args.out)


if __name__ == "__main__":
    main()
