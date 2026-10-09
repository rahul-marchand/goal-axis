"""The route model's twin of ``far_axis_fit.py``: a fresh axis through the far arms.

    uv run python scripts/far_axis_fit_bc.py RUN --family k1 --far 0.8 0.9 1.0 1.25 1.5 \
        --demos /workspace/data/offline/demos/test.rho100

Offsets here are relative, as in ``027``: ``k1+080`` is +0.8 on colour 1's
value. Same three readings as the agents' script -- weight-space leave-one-out
within the far set, a write from the base with the far fit's drift, and a write
from the nearest far arm along the far axis -- so the numbers land in the same
table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from goalmisgen.analysis.weights import (
    cosine,
    explained,
    fit_axis_and_drift,
    projected_offset,
    readback_slope,
)
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--family", type=str, required=True, help="k1 or k0.")
    parser.add_argument("--far", type=float, nargs="+", required=True, help="Relative offsets of the far arms.")
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument("--window", type=float, default=0.45, help="|offset| of the window arms to fit against.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument("--window-loo", action="store_true", help="Measure the window's leave-one-out writes instead.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header() + "\n")
    from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs, measure_flat
    from goalmisgen.offline.demos import DemoSet

    objective = int(args.family[-1])
    base = load_base(args.run)
    window_arms = {
        o: p for o, p in arm_dirs(args.run, f"o{objective}", args.steps).items() if 1e-9 < abs(o) <= args.window + 1e-9
    }
    window = load_diffs(base, window_arms, dtype=np.float64)
    w_offsets = np.array(sorted(window))
    axis_window, drift_window = fit_axis_and_drift(w_offsets, np.stack([window[o] for o in w_offsets]))

    far_all = arm_dirs(args.run, args.family, args.steps)
    wanted = {round(o, 10) for o in args.far}
    far_arms = {o: p for o, p in far_all.items() if round(o, 10) in wanted}
    missing = wanted - {round(o, 10) for o in far_arms}
    if missing:
        sys.exit(f"far arms not found at offsets {sorted(missing)}; have {sorted(far_all)}")
    far = load_diffs(base, far_arms, dtype=np.float64)
    f_offsets = np.array(sorted(far))
    axis_far, drift_far = fit_axis_and_drift(f_offsets, np.stack([far[o] for o in f_offsets]))

    print(f"base       {args.run.name} @ step {base.step:,}   ({base.flat.size:,} parameters)")
    print(f"window     o{objective}: {len(w_offsets)} arms at offsets {w_offsets.min():+.2f}..{w_offsets.max():+.2f}")
    print(f"far        {args.family}: {len(f_offsets)} arms at offsets {f_offsets.min():+.2f}..{f_offsets.max():+.2f}\n")

    print("=== the far axis against the window axis ===\n")
    print(f"  |axis| per unit value   window {np.linalg.norm(axis_window):.4g}   far {np.linalg.norm(axis_far):.4g}")
    print(f"  |drift|                 window {np.linalg.norm(drift_window):.4g}   far {np.linalg.norm(drift_far):.4g}")
    print(f"  cos(axis_far, axis_window)    {cosine(axis_far, axis_window):+.3f}")
    print(f"  cos(drift_far, axis_window)   {cosine(drift_far, axis_window):+.3f}")
    print(f"  cos(drift_far, drift_window)  {cosine(drift_far, drift_window):+.3f}")

    print("\n=== graded and predictive? leave-one-out within the far set ===\n")
    print(f"  {'offset':>8}{'|delta|':>10}{'held-out R^2':>14}{'held-out cos':>14}{'implied':>10}")
    far_loo = {}
    for o in f_offsets:
        others = [u for u in f_offsets if u != o]
        held_axis, held_drift = fit_axis_and_drift(np.array(others), np.stack([far[u] for u in others]))
        left = far[o] - held_drift
        far_loo[o] = {
            "axis": held_axis,
            "drift": held_drift,
            "r2": explained(left, o, held_axis),
            "cos": cosine(left, held_axis),
            "implied": projected_offset(left, held_axis),
        }
        print(
            f"  {o:>+8.2f}{np.linalg.norm(far[o]):>10.4g}{far_loo[o]['r2']:>14.3f}{far_loo[o]['cos']:>14.3f}{far_loo[o]['implied']:>+10.2f}"
        )
    far_slope = readback_slope(f_offsets, [far_loo[o]["implied"] for o in f_offsets])
    window_implied = []
    for o in w_offsets:
        others = [u for u in w_offsets if u != o]
        held_axis, held_drift = fit_axis_and_drift(np.array(others), np.stack([window[u] for u in others]))
        window_implied.append(projected_offset(window[o] - held_drift, held_axis))
    window_slope = readback_slope(w_offsets, window_implied)
    implied_by_window = [projected_offset(far[o] - drift_window, axis_window) for o in f_offsets]
    print(f"\n  readback slope (implied on true offset)   window {window_slope:.2f}   far {far_slope:.2f}")
    print(
        f"  far arms read by the window axis: implied {', '.join(f'{x:+.2f}' for x in implied_by_window)}"
        f"   slope {readback_slope(f_offsets, implied_by_window):.2f}"
    )

    result = {
        "run": args.run.name,
        "family": args.family,
        "far_offsets": list(map(float, f_offsets)),
        "axis_norm": {"window": float(np.linalg.norm(axis_window)), "far": float(np.linalg.norm(axis_far))},
        "cos_far_window": cosine(axis_far, axis_window),
        "loo": {f"{o:+.2f}": {k: float(far_loo[o][k]) for k in ("r2", "cos", "implied")} for o in f_offsets},
        "readback_slope": {
            "window": window_slope,
            "far": far_slope,
            "far_by_window_axis": readback_slope(f_offsets, implied_by_window),
        },
    }
    if args.skip_behaviour:
        finish(args, result, axis_far)
        return

    demos = DemoSet.load(args.demos, hide_values=base.hide_values)
    indices = np.arange(min(args.levels, len(demos)))
    rng = np.random.default_rng(args.seed)

    def measure(flat, label):
        m = measure_flat(base, flat, demos, indices)
        print(f"  {label:>36}{m.indifference:>8.1f}{m.reached:>11.1%}", flush=True)
        return float(m.indifference), float(m.reached)

    if args.window_loo:
        print("\n\n=== the window's own leave-one-out writes, same levels ===\n")
        print(f"  {'':>36}{'steps':>8}{'reached':>11}")
        wrows = []
        for o in w_offsets:
            others = [u for u in w_offsets if u != o]
            held_axis, _ = fit_axis_and_drift(np.array(others), np.stack([window[u] for u in others]))
            arm_point, arm_reach = measure(base.flat + window[o], f"{o:+.2f} fine-tuned")
            written, w_reach = measure(base.flat + o * held_axis, f"{o:+.2f} written, held out")
            wrows.append(
                {"offset": float(o), "arm": arm_point, "arm_reach": arm_reach, "written": written, "written_reach": w_reach}
            )
        err = np.array([r["written"] - r["arm"] for r in wrows])
        print(
            f"\n  median |error| {np.median(np.abs(err)):.1f} steps;  slope of written on fine-tuned "
            f"{readback_slope([r['arm'] for r in wrows], [r['written'] for r in wrows]):.2f}"
        )
        result["window_loo"] = {
            "rows": wrows,
            "median_abs_error": float(np.median(np.abs(err))),
            "written_on_arm_slope": readback_slope([r["arm"] for r in wrows], [r["written"] for r in wrows]),
        }
        finish(args, result, axis_far)
        return

    print("\n\n=== written within the reversed regime, each arm from a fit that never saw it ===\n")
    print(f"  {'':>36}{'steps':>8}{'reached':>11}")
    rows = []
    for o in f_offsets:
        held_axis, held_drift = far_loo[o]["axis"], far_loo[o]["drift"]
        arm_point, arm_reach = measure(base.flat + far[o], f"{o:+.2f} fine-tuned")
        from_base, base_reach = measure(base.flat + held_drift + o * held_axis, f"{o:+.2f} written from base")
        anchor = min((u for u in f_offsets if u != o), key=lambda u: abs(u - o))
        step = (o - anchor) * held_axis
        from_anchor, anchor_reach = measure(base.flat + far[anchor] + step, f"{o:+.2f} written from {anchor:+.2f}")
        direction = rng.normal(size=step.size)
        direction *= np.linalg.norm(step) / np.linalg.norm(direction)
        random_point, random_reach = measure(base.flat + far[anchor] + direction, f"{o:+.2f} random from {anchor:+.2f}")
        rows.append(
            {
                "offset": float(o),
                "anchor": float(anchor),
                "arm": arm_point,
                "arm_reach": arm_reach,
                "from_base": from_base,
                "from_base_reach": base_reach,
                "from_anchor": from_anchor,
                "from_anchor_reach": anchor_reach,
                "random_from_anchor": random_point,
                "random_reach": random_reach,
            }
        )

    print(
        f"\n  {'offset':>8}{'fine-tuned':>12}{'from base':>11}{'err':>7}{'from nbr':>10}{'err':>7}{'random':>9}{'err':>7}{'nbr arm':>9}"
    )
    for r in rows:
        nbr = next(x["arm"] for x in rows if x["offset"] == r["anchor"])
        print(
            f"  {r['offset']:>+8.2f}{r['arm']:>12.1f}{r['from_base']:>11.1f}{r['from_base'] - r['arm']:>+7.1f}"
            f"{r['from_anchor']:>10.1f}{r['from_anchor'] - r['arm']:>+7.1f}{r['random_from_anchor']:>9.1f}{r['random_from_anchor'] - r['arm']:>+7.1f}{nbr:>9.1f}"
        )
    arms_ = np.array([r["arm"] for r in rows])
    err_base = np.array([r["from_base"] - r["arm"] for r in rows])
    err_anchor = np.array([r["from_anchor"] - r["arm"] for r in rows])
    err_random = np.array([r["random_from_anchor"] - r["arm"] for r in rows])
    asked = np.array([r["arm"] - next(x["arm"] for x in rows if x["offset"] == r["anchor"]) for r in rows])
    moved = np.array([r["from_anchor"] - next(x["arm"] for x in rows if x["offset"] == r["anchor"]) for r in rows])
    print(
        f"\n  median |error|   from base {np.median(np.abs(err_base)):.1f}   from neighbour {np.median(np.abs(err_anchor)):.1f}"
        f"   random {np.median(np.abs(err_random)):.1f} steps"
    )
    print(
        f"  neighbour write: asked to move {np.median(np.abs(asked)):.1f} steps (median), moved {np.median(np.abs(moved)):.1f};"
        f" slope of moved on asked {readback_slope(asked, moved):.2f}"
    )
    print(
        f"  slope of written on fine-tuned across far arms   from base {readback_slope(arms_, [r['from_base'] for r in rows]):.2f}"
        f"   from neighbour {readback_slope(arms_, [r['from_anchor'] for r in rows]):.2f}"
    )
    result["behaviour"] = {
        "rows": rows,
        "median_abs_error": {
            "from_base": float(np.median(np.abs(err_base))),
            "from_anchor": float(np.median(np.abs(err_anchor))),
            "random": float(np.median(np.abs(err_random))),
        },
        "asked_median": float(np.median(np.abs(asked))),
        "moved_median": float(np.median(np.abs(moved))),
        "moved_on_asked_slope": readback_slope(asked, moved),
        "written_on_arm_slope": {
            "from_base": readback_slope(arms_, [r["from_base"] for r in rows]),
            "from_anchor": readback_slope(arms_, [r["from_anchor"] for r in rows]),
        },
    }
    finish(args, result, axis_far)


def finish(args: argparse.Namespace, result: dict, axis_far: np.ndarray | None = None) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1))
        if axis_far is not None:
            np.save(args.json.with_suffix(".axis.npy"), np.asarray(axis_far, dtype=np.float32))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
