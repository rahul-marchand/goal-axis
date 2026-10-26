"""The warning lights of retargeting runs, with one standard figure per run.

    uv run python scripts/studies/retarget/diagnose.py /workspace/data/offline/results/retarget \
        [--only bcnv11.s1] [--out ~/retarget-diagnostics] [--competence 1.0]

Reads every ``<agent>/<arm>.json`` record under the root (``goalmisgen.retarget.Record``,
any coding mechanism), prints each run's checks (``goalmisgen.retarget.diagnostics``)
and the study checks, and with ``--out`` writes ``<agent>.<arm>.png``:

1. learning curves on T (the coded model, split into levels the goal changes
   and levels it leaves alone, and a teacher's where the mechanism has one),
   against the finish line, the base's own score at its own goal;
2. the cumulative cost, with the per-message overhead and the unpaid side
   channel where the mechanism reports them.

A run's finish line is its own record's, or its null arm's L(0) for records
that predate it.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from goalmisgen.offline.retarget import NULL_TAU  # noqa: E402
from goalmisgen.retarget import Record, diagnostics, metric  # noqa: E402

COLOURS = {
    "coded model": "#2f5a78",
    "coded model, levels the goal changes": "#c0703a",
    "coded model, levels it leaves alone": "#5c8a5a",
    "teacher": "#6f5f9c",
    "teacher, levels it leaves alone": "#a99bd0",
}


def plot(r: Record, finish: float | None, findings: list[diagnostics.Finding], title: str, path: pathlib.Path) -> None:
    bounds = r.result.bounds
    x = [max(t, 0.5) for t in bounds]
    fig, (curves_ax, cost_ax) = plt.subplots(1, 2, figsize=(11, 4))

    for name, curve in diagnostics.curves(r).items():
        dashed = name.startswith("teacher")
        curves_ax.plot(x, curve, color=COLOURS.get(name), ls="--" if dashed else "-", lw=1.8, marker="o", ms=3, label=name)
    if finish is not None:
        curves_ax.axhline(finish, color="black", lw=1, ls=":", label=f"finish line {finish:.3f}")
    for ax in (curves_ax, cost_ax):
        ax.set_xscale("log", base=2)
        shown, last = [0], x[0]
        for i in range(1, len(x)):
            if x[i] >= 2.5 * last:  # label only checkpoints far enough apart on the log axis to read
                shown.append(i)
                last = x[i]
        ax.set_xticks([x[i] for i in shown])
        ax.set_xticklabels([f"{bounds[i]:,}" if bounds[i] < 1000 else f"{bounds[i] / 1000:.0f}k" for i in shown], fontsize=7)
        ax.minorticks_off()
    curves_ax.set_xlabel("examples coded")
    curves_ax.set_ylabel("bits per example on T")
    curves_ax.set_title("learning curves", loc="left")
    curves_ax.legend(fontsize=7.5)

    cost = metric.cumulative(r.result.example_bits, bounds)
    cost_ax.plot(x, cost, color=COLOURS["coded model"], lw=1.8, marker="o", ms=3, label="per-example cost")
    extras = [(r.result.overhead_bits, "overhead"), (r.result.side_channel_bits or 0.0, "unpaid side channel (bound)")]
    for bits, label in extras:
        if bits:
            cost_ax.axhline(bits, color="grey", lw=1, ls="--", label=f"{label} {bits:,.0f}")
    cost_ax.set_xlabel("examples coded")
    cost_ax.set_ylabel("bits")
    cost_ax.set_title(f"cumulative cost (total {r.result.total_bits:,.0f})", loc="left")
    cost_ax.legend(fontsize=7.5)

    flagged = [f for f in findings if f.status in (diagnostics.Status.WARN, diagnostics.Status.FAIL)]
    summary = "; ".join(f"{f.status.value}: {f.check}" for f in flagged) or "all checks ok or not applicable"
    fig.suptitle(f"{title}  [{r.result.kind}]\n{summary}", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--only", nargs="*", default=None, help="Agents (directory names) to diagnose; default all.")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Write one PNG per run here.")
    parser.add_argument("--competence", type=float, default=None, help="diagnostics.Thresholds.competence.")
    args = parser.parse_args()

    thresholds = diagnostics.Thresholds(competence=args.competence)
    records = {
        (path.parent.name, path.stem): Record.load(path)
        for path in sorted(args.root.glob("*/*.json"))
        if args.only is None or path.parent.name in args.only
    }
    nulls = metric.null_arms(((agent, r) for (agent, _), r in records.items()), NULL_TAU)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    counts = {status: 0 for status in diagnostics.Status}
    for (agent, arm), r in records.items():
        finish = diagnostics.finish_line(r, metric.null_of(nulls, agent, r))
        findings = diagnostics.run_checks(r, finish, thresholds)
        status = diagnostics.worst(findings)
        counts[status] += 1
        print(f"{agent}/{arm}: {status.value}\n{diagnostics.render(findings, skipped=False)}\n")
        if args.out:
            plot(r, finish, findings, f"{agent}/{arm}", args.out / f"{agent}.{arm}.png")

    study = diagnostics.study_checks(((f"{agent}/{arm}", r) for (agent, arm), r in records.items()), NULL_TAU, thresholds)
    print(f"study: {diagnostics.worst(study).value}\n{diagnostics.render(study)}\n")
    print("runs by worst light: " + ", ".join(f"{s.value} {n}" for s, n in counts.items() if n))
    if args.out:
        print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()
