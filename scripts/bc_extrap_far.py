"""Axis-only writes for bcnv11 far past the fitted window (both sweeps, 3 seeds).

Same convention as extrap_axisonly.json (base + o*axis, no drift), at offsets
matching the RL bases' far extrapolation, so the appendix reads all four
architectures off one grid. Resumable: sections already in the json are skipped.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace/GoalMisgeneralisation")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("va027", REPO / "experiments/027_bc_value_axis.py")
va = importlib.util.module_from_spec(spec)
spec.loader.exec_module(va)

OUT = Path("/workspace/data/figures-data/bc/extrap_axisonly_far.json")
out = json.loads(OUT.read_text()) if OUT.is_file() else {}
MAG = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5]

for seed in (1, 2, 3):
    for sweep in ("o0", "o1"):
        if out.get(f"s{seed}", {}).get(sweep):
            print(f"s{seed} {sweep}: already measured, skipping")
            continue
        sign = -1.0 if sweep == "o0" else 1.0
        sys.argv = ["bc_extrap_far", f"/workspace/data/offline/runs/bcnv11.s{seed}", "--sweep", sweep,
                    "--steps", "1000", "--demos", "/workspace/data/offline/demos/test.rho100"]
        args = va.parse_args()
        base = va.load_base(args.run)
        arms = va.arm_dirs(args.run, args.sweep, args.steps)
        diffs = va.load_diffs(base, arms)
        fitted = {o: d for o, d in diffs.items() if abs(o) > 1e-9}
        offsets = np.array(sorted(fitted))
        axis, drift = va.fit_axis_and_drift(offsets, np.stack([fitted[o] for o in offsets]))
        demos = va.DemoSet.load(args.demos, hide_values=base.hide_values)
        indices = np.arange(min(args.levels, len(demos)))
        rows = {}
        for mag in MAG:
            o = sign * mag
            m = va.measure_flat(base, base.flat + o * axis, demos, indices)
            rows[f"{o:+.2f}"] = m.as_row()
            print(f"s{seed} {sweep} {o:+.2f}: indiff {m.indifference:.2f} reached {m.reached:.3f}", flush=True)
        out.setdefault(f"s{seed}", {})[sweep] = rows
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
print("BC_EXTRAP_FAR_DONE")
