"""Table 1 read down a ladder of arm lengths, from the checkpoints the arms saved.

    uv run python scripts/arm_length_ladder.py /workspace/data/runs/novalue11g1.s1234 \
        --arm-steps 400000 3000000 --out results/len-novalue11g1.s1234-weights.json

Each ``013`` arm saves evenly spaced checkpoints and trains at a constant learning
rate, so an arm read at its k-th checkpoint is the arm it would have been with
that budget. Both sweeps are decomposed at every checkpoint index of every arm
length given, and the drift / write / remainder shares, the axis norms, the
split-half reliability and ``cos(axis_0, axis_1)`` are printed as one row per
budget and written to JSON for the appendix table.

Weight space only: run it with ``JAX_PLATFORMS=cpu`` beside a busy GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen import provenance
from goalmisgen.analysis.arm_length import decompose, table_row
from goalmisgen.analysis.weights import cosine, fit_axis_and_drift
from goalmisgen.volume import discover_arms, parse_checkpoint_dirname


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Agent directory holding BASE.json and arms/.")
    parser.add_argument("--arm-steps", type=int, nargs="+", required=True, help="Arm budgets to read, e.g. 400000 3000000.")
    parser.add_argument("--dose", type=float, default=0.3, help="|offset| the shares are read at, as in Table 1.")
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None, help="JSON destination.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")
    from goalmisgen.axis import diff_from, load_base

    # Weights only: the environment is built to load them, never stepped.
    base = load_base(args.run, None, split="train", num_envs=1, seed=None)
    checkpoint, values, base_flat = base.checkpoint, base.values, base.flat
    print(f"base {checkpoint}  (update {base.update}, {base_flat.size:,} parameters)")
    print(f"shares read at |offset| = {args.dose}, averaged over both sweeps' arms at that dose\n")

    def diffs_for(objective: int, steps: int, at: int) -> tuple[np.ndarray, np.ndarray, int]:
        found = discover_arms(args.run / "arms", objective, values[objective], steps=steps, at=at)
        offsets, rows, reached = [], [], set()
        for value, path in sorted(found.items()):
            rows.append(diff_from(base, path, subtract_in="float64"))
            offsets.append(round(value - values[objective], 10))
            reached.add(parse_checkpoint_dirname(path.name))
        if len(reached) > 1:
            raise SystemExit(f"o{objective}@{steps} index {at}: arms at different steps {sorted(reached)}")
        return np.array(offsets), np.stack(rows), (reached.pop() if reached else 0)

    header = f"{'budget':>8}{'read at':>11}{'drift%':>8}{'write%':>8}{'resid%':>8}{'cross%':>8}{'|drift|':>9}{'|axis|':>8}{'rel0':>7}{'rel1':>7}{'cos01':>8}"
    print(header)
    rows_out = []
    for steps in args.arm_steps:
        probe = discover_arms(args.run / "arms", 1, values[1], steps=steps, at=-1)
        if not probe:
            print(f"{steps:>8}  no arms at this budget, skipping")
            continue
        count = min(len(sorted((p.parent).glob("cp_*"))) for p in probe.values())
        for at in range(count):
            parts, axes, reached_steps = {}, {}, set()
            for objective in (0, 1):
                offsets, diffs, reached = diffs_for(objective, steps, at)
                reached_steps.add(reached)
                decomposition = decompose(offsets, diffs, splits=args.splits)
                parts[f"o{objective}"] = decomposition
                fitted = np.abs(offsets) > 1e-9
                axes[objective] = fit_axis_and_drift(offsets[fitted], diffs[fitted])[0]
            row = table_row(parts, args.dose)
            read_at = max(reached_steps)
            cos01 = cosine(axes[0], axes[1])
            print(
                f"{steps:>8}{read_at:>11,}{100 * row['drift']:>8.1f}{100 * row['write']:>8.1f}"
                f"{100 * row['remainder']:>8.1f}{100 * row['cross']:>8.1f}"
                f"{np.mean([p.drift_norm for p in parts.values()]):>9.2f}"
                f"{np.mean([p.axis_norm for p in parts.values()]):>8.2f}"
                f"{parts['o0'].reliability:>7.2f}{parts['o1'].reliability:>7.2f}{cos01:>8.3f}"
            )
            rows_out.append(
                {
                    "arm_steps": steps,
                    "index": at,
                    "read_at": read_at,
                    "shares": row,
                    "cos_axis0_axis1": cos01,
                    "sweeps": {k: v.to_json() for k, v in parts.items()},
                }
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"agent": args.run.name, "base": base.marker.payload["checkpoint"], "dose": args.dose, "rows": rows_out}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
