"""Write one module's slice of the route model's value axis, and read the behaviour.

    uv run python scripts/restricted_writes_bc.py /workspace/data/offline/runs/bcnv11.s1 \
        --demos /workspace/data/offline/demos/test.rho100 --json figures/data/restricted/bcnv11.s1.o1.json

The offline twin of ``scripts/restricted_writes.py``: the same three variants per
module (alone as is, alone rescaled to the axis's length, everything except the
module), the same largest-``k`` cumulative writes ranked by the other sweep, the
same module grouping from :func:`goalmisgen.analysis.layers.module_groups`, and
the same JSON shape, so one figure builder reads both. What differs is only the
loading and the decode, which go through :mod:`goalmisgen.offline.axis` as
``027`` and ``029`` do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.layers import module_groups, restrict
from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.offline.axis import arm_dirs, load_base, load_sweep, measure_flat
from goalmisgen.offline.demos import DemoSet
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Base run directory; arms under <run>/arms/.")
    parser.add_argument("--demos", type=Path, required=True, help="Held-out demonstrations at the base values.")
    parser.add_argument("--sweep", type=str, default="o1", help="Which sweep's axis to write; the other ranks modules.")
    parser.add_argument("--steps", type=int, default=1000, help="Arm budget, as in the arm directory names.")
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.3, 0.3])
    parser.add_argument(
        "--doses",
        type=float,
        nargs="+",
        default=None,
        help="Dose-response mode: write each module's slice at these multiples of its own size, at every "
        "offset, and nothing else. Straight lines through the origin are what a linear knob gives.",
    )
    parser.add_argument("--cumulative", type=int, default=4)
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    base = load_base(args.run)
    objective = int(args.sweep[-1])
    other = f"o{1 - objective}"
    print(f"base   {args.run.name} @ step {base.step:,}  ({base.flat.size:,} parameters), values {base.values}")

    groups, order = module_groups(base.params)
    sizes = {name: int(len(groups[name])) for name in order}

    offsets, diffs = load_sweep(base, args.sweep, args.steps, fit_window=args.fit_window, minimum=3)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    print(f"sweep  {args.sweep}: {len(offsets)} arms at {np.round(offsets, 2).tolist()}; |axis| {np.linalg.norm(axis):.3g}")

    axis_share = {name: float(axis[groups[name]] @ axis[groups[name]]) / float(axis @ axis) for name in order}
    rank_by, rank_label = axis_share, f"share of ||axis||^2 in sweep {args.sweep} (no second sweep to rank by)"
    if arm_dirs(args.run, other, args.steps):
        other_offsets, other_diffs = load_sweep(base, other, args.steps, fit_window=args.fit_window, minimum=3)
        other_axis, _ = fit_axis_and_drift(other_offsets, other_diffs)
        del other_diffs
        coherent = {name: float(-(axis[groups[name]] @ other_axis[groups[name]])) for name in order}
        total = sum(coherent.values())
        rank_by = {name: value / total for name, value in coherent.items()}
        rank_label = f"share of the sweep-agreeing axis, -axis_{args.sweep} . axis_{other} per module"
    ranking = sorted(order, key=lambda name: -rank_by[name])

    print(f"\nmodules, forward order; ranked by {rank_label}\n")
    print(f"  {'module':<20}{'params':>10}{'params %':>10}{'axis %':>9}{'rank %':>9}{'rank':>6}")
    for name in order:
        print(
            f"  {name:<20}{sizes[name]:>10,}{100 * sizes[name] / base.flat.size:>10.2f}{100 * axis_share[name]:>9.1f}"
            f"{100 * rank_by[name]:>9.1f}{ranking.index(name) + 1:>6}"
        )

    out = {
        "run": str(args.run),
        "checkpoint": f"step {base.step}",
        "values": list(base.values),
        "sweep": args.sweep,
        "offsets_fitted": offsets.tolist(),
        "modules": order,
        "sizes": sizes,
        "axis_share": axis_share,
        "rank_share": rank_by,
        "rank_label": rank_label,
        "ranking": ranking,
        "write_offsets": args.offsets,
        "writes": [],
    }
    if args.dry_run:
        finish(args, out)
        return

    demos = DemoSet.load(args.demos, hide_values=base.hide_values)
    indices = np.arange(min(args.levels, len(demos)))
    axis_norm = float(np.linalg.norm(axis))

    def measure(flat: np.ndarray, label: str, **tags) -> dict:
        m = measure_flat(base, flat, demos, indices)
        print(f"  {label:>44}{m.indifference:>8.1f}{m.reached:>10.1%}{m.chose_optimal:>10.1%}", flush=True)
        row = {"label": label, "steps": m.indifference, "reached": m.reached, "chose_optimal": m.chose_optimal, **tags}
        out["writes"].append(row)
        return row

    print(f"\n=== writes, greedy decode on {len(indices)} held-out levels at the base values ===\n")
    print(f"  {'':>44}{'steps':>8}{'reached':>10}{'optimal':>10}")
    measure(base.flat, "base, untouched", kind="base", offset=0.0)
    if args.doses:
        out["doses"] = args.doses
        for offset in args.offsets:
            print()
            measure(base.flat + offset * axis, f"full axis @ {offset:+.2f}", kind="full", offset=offset)
            for name in order:
                part = restrict(axis, groups, name)
                for dose in args.doses:
                    measure(
                        base.flat + offset * dose * part,
                        f"only {name} x{dose:g} @ {offset:+.2f}",
                        kind="dose",
                        module=name,
                        offset=offset,
                        dose=dose,
                    )
        finish(args, out)
        return
    for offset in args.offsets:
        print()
        measure(base.flat + offset * axis, f"full axis @ {offset:+.2f}", kind="full", offset=offset)
        for name in order:
            part = restrict(axis, groups, name)
            share = float(part @ part) / float(axis @ axis)
            measure(base.flat + offset * part, f"only {name} @ {offset:+.2f}", kind="alone", module=name, offset=offset, axis_share=share)
            norm = float(np.linalg.norm(part))
            if norm > 0:
                scale = axis_norm / norm
                measure(
                    base.flat + offset * scale * part,
                    f"only {name}, rescaled x{scale:.1f} @ {offset:+.2f}",
                    kind="rescaled",
                    module=name,
                    offset=offset,
                    scale=scale,
                )
            measure(base.flat + offset * (axis - part), f"all but {name} @ {offset:+.2f}", kind="without", module=name, offset=offset)
        for count in range(1, min(args.cumulative, len(ranking)) + 1):
            keep = ranking[:count]
            part = restrict(axis, groups, keep)
            measure(
                base.flat + offset * part,
                f"largest {count} ({', '.join(keep)}) @ {offset:+.2f}",
                kind="cumulative",
                modules=keep,
                offset=offset,
                axis_share=float(part @ part) / float(axis @ axis),
            )
    finish(args, out)


def finish(args: argparse.Namespace, out: dict) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
