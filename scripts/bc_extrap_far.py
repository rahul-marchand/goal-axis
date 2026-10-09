"""Axis-only writes for bcnv11 far past the fitted window (both sweeps, 3 seeds).

    uv run python scripts/bc_extrap_far.py

Same convention as extrap_axisonly.json (base + o*axis, no drift), at offsets
matching the RL bases' far extrapolation, so the appendix reads all four
architectures off one grid. Resumable: sections already in the json are skipped.

It ran on the volume by loading ``experiments/027_bc_value_axis.py`` from its
path and borrowing 027's defaults and re-exports; the defaults here are those
(2,048 held-out levels, arms at 1k steps, every arm away from zero fitted), and
the library is imported directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs, measure_flat
from goalmisgen.offline.demos import DemoSet
from goalmisgen.paths import figures_data_dir, offline_demos, offline_runs_dir

MAGNITUDES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=offline_runs_dir(), help="Holds bcnv11.s<seed>.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sweeps", nargs="+", default=["o0", "o1"])
    parser.add_argument("--steps", type=int, default=1000, help="Arm budget, as in the arm directory names.")
    parser.add_argument("--demos", type=Path, default=offline_demos("test"))
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=figures_data_dir() / "bc" / "extrap_axisonly_far.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = json.loads(args.out.read_text()) if args.out.is_file() else {}
    for seed in args.seeds:
        for sweep in args.sweeps:
            if out.get(f"s{seed}", {}).get(sweep):
                print(f"s{seed} {sweep}: already measured, skipping")
                continue
            # Writes go the way the sweep's own arms went past parity: down for o0, up for o1.
            sign = -1.0 if sweep == "o0" else 1.0
            run = args.runs / f"bcnv11.s{seed}"
            base = load_base(run)
            diffs = load_diffs(base, arm_dirs(run, sweep, args.steps))
            fitted = {o: d for o, d in diffs.items() if abs(o) > 1e-9}
            offsets = np.array(sorted(fitted))
            axis, _ = fit_axis_and_drift(offsets, np.stack([fitted[o] for o in offsets]))
            demos = DemoSet.load(args.demos, hide_values=base.hide_values)
            indices = np.arange(min(args.levels, len(demos)))
            rows = {}
            for magnitude in MAGNITUDES:
                offset = sign * magnitude
                m = measure_flat(base, base.flat + offset * axis, demos, indices)
                rows[f"{offset:+.2f}"] = m.as_row()
                print(f"s{seed} {sweep} {offset:+.2f}: indiff {m.indifference:.2f} reached {m.reached:.3f}", flush=True)
            out.setdefault(f"s{seed}", {})[sweep] = rows
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, indent=1))
    print("BC_EXTRAP_FAR_DONE")


if __name__ == "__main__":
    main()
