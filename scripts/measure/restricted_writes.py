"""Write one module's slice of the value axis into the base, and read the behaviour.

    uv run python scripts/measure/restricted_writes.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --json figures/data/restricted/novalue11g1.s1234.json

A share of ``||axis||^2`` in a module says where the fitted direction *landed*;
it does not say whether writing that much of it does anything. Those come apart
whenever a quantity is recomputed downstream of where the gradient found it
cheapest to write, which is the Hase et al. point and the reason
``experiments/019`` found the enriched channels read well and wrote badly. So the
per-module figure needs a companion: the axis restricted to each module, added
onto the base's own weights, and the exchange rate decoded on held-out levels.

Three variants per module, as in ``019``:

* **alone, as is** -- those coordinates at their own magnitude. Sufficiency at
  the size the axis actually gave them.
* **alone, rescaled** -- the same direction stretched to the full axis's length.
  Whether the direction inside the module is right irrespective of how much of
  it survived the restriction.
* **without** -- everything except the module. Necessity: if the write still
  lands without it, the module was not where the effect was carried.

plus the largest-``k`` modules cumulatively, ranked by how much of the axis the
two independent sweeps agree on there (so the ranking does not come from the
sweep being written, which would leak). Every write is measured at both signs
of the offset so a module that moves the threshold only one way shows up.

Groups follow the architecture's own top-level modules, in forward order, so the
same table reads for a DRC, a ResNet and a ViT; the grouping is the same one the
per-module figure uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.layers import module_groups, restrict
from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.provenance import header
from goalmisgen.volume import sweep_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Agent directory holding BASE.json and arms/.")
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sweep", type=str, default="o1", help="Which sweep's axis to write; the other ranks modules.")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45, help="Arms beyond |offset| > this stay out of the fit.")
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.3, 0.3], help="Value offsets to write.")
    parser.add_argument(
        "--doses",
        type=float,
        nargs="+",
        default=None,
        help="Dose-response mode: write each module's slice at these multiples of its own size, at every "
        "offset, and nothing else. Straight lines through the origin are what a linear knob gives.",
    )
    parser.add_argument("--cumulative", type=int, default=4, help="How many largest-k cumulative writes to try.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Fit and rank only; no rollouts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    from goalmisgen.axis import Greedy, has_sweep, load_base, load_sweep

    base = load_base(args.run, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed)
    checkpoint, values, base_flat = base.checkpoint, base.values, base.flat
    objective = sweep_index(args.sweep)
    other = f"o{1 - objective}" if len(values) == 2 else None
    print(f"base   {checkpoint}  ({base_flat.size:,} parameters), values {values}")

    groups, order = module_groups(base.params)
    sizes = {name: int(len(groups[name])) for name in order}

    offsets, diffs = load_sweep(base, args.sweep, arm_steps=args.arm_steps, fit_window=args.fit_window)
    axis, drift = fit_axis_and_drift(offsets, diffs)
    print(f"sweep  {args.sweep}: {len(offsets)} arms at {np.round(offsets, 2).tolist()}; |axis| {np.linalg.norm(axis):.3g}")

    axis_share = {name: float(axis[groups[name]] @ axis[groups[name]]) / float(axis @ axis) for name in order}
    rank_by, rank_label = axis_share, f"share of ||axis||^2 in sweep {args.sweep} (no second sweep to rank by)"
    if other is not None and has_sweep(base, other, arm_steps=args.arm_steps):
        other_offsets, other_diffs = load_sweep(base, other, arm_steps=args.arm_steps, fit_window=args.fit_window)
        other_axis, _ = fit_axis_and_drift(other_offsets, other_diffs)
        coherent = {name: float(-(axis[groups[name]] @ other_axis[groups[name]])) for name in order}
        total = sum(coherent.values())
        rank_by = {name: value / total for name, value in coherent.items()}
        rank_label = f"share of the sweep-agreeing axis, -axis_{args.sweep} . axis_{other} per module"
        del other_diffs
    ranking = sorted(order, key=lambda name: -rank_by[name])
    del diffs

    print(f"\nmodules, forward order; ranked by {rank_label}\n")
    print(f"  {'module':<20}{'params':>10}{'params %':>10}{'axis %':>9}{'rank %':>9}{'rank':>6}")
    for name in order:
        print(
            f"  {name:<20}{sizes[name]:>10,}{100 * sizes[name] / base_flat.size:>10.2f}{100 * axis_share[name]:>9.1f}"
            f"{100 * rank_by[name]:>9.1f}{ranking.index(name) + 1:>6}"
        )

    out = {
        "run": str(args.run),
        "checkpoint": str(checkpoint),
        "values": list(values),
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

    greedy = Greedy.of(base)
    axis_norm = float(np.linalg.norm(axis))

    def measure(flat: np.ndarray, label: str, **tags) -> dict:
        found = greedy.threshold(base.params_of(flat), args.episodes, args.seed)
        point, low, high, reached = found.point, found.low, found.high, found.reached
        print(f"  {label:>44}{point:>8.1f}  [{low:5.1f}, {high:5.1f}]{reached:>10.1%}", flush=True)
        row = {"label": label, **found.as_row(), **tags}
        out["writes"].append(row)
        return row

    print(f"\n=== writes, decoded on {args.episodes} held-out levels at the base values ===\n")
    print(f"  {'':>44}{'steps':>8}  {'95% interval':^14}{'reached':>10}")
    measure(base_flat, "base, untouched", kind="base", offset=0.0)
    if args.doses:
        out["doses"] = args.doses
        for offset in args.offsets:
            print()
            measure(base_flat + offset * axis, f"full axis @ {offset:+.2f}", kind="full", offset=offset)
            for name in order:
                part = restrict(axis, groups, name)
                for dose in args.doses:
                    measure(
                        base_flat + offset * dose * part,
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
        measure(base_flat + offset * axis, f"full axis @ {offset:+.2f}", kind="full", offset=offset)
        for name in order:
            part = restrict(axis, groups, name)
            share = float(part @ part) / float(axis @ axis)
            measure(
                base_flat + offset * part,
                f"only {name} @ {offset:+.2f}",
                kind="alone",
                module=name,
                offset=offset,
                axis_share=share,
            )
            scale = axis_norm / float(np.linalg.norm(part)) if np.linalg.norm(part) > 0 else float("nan")
            if np.isfinite(scale):
                measure(
                    base_flat + offset * scale * part,
                    f"only {name}, rescaled x{scale:.1f} @ {offset:+.2f}",
                    kind="rescaled",
                    module=name,
                    offset=offset,
                    scale=scale,
                )
            measure(
                base_flat + offset * (axis - part),
                f"all but {name} @ {offset:+.2f}",
                kind="without",
                module=name,
                offset=offset,
            )
        for count in range(1, min(args.cumulative, len(ranking)) + 1):
            keep = ranking[:count]
            part = restrict(axis, groups, keep)
            measure(
                base_flat + offset * part,
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
