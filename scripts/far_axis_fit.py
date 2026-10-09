"""Fit a fresh axis to the arms past parity, and ask whether it carries anything.

    uv run python scripts/far_axis_fit.py RUN --family k1 --base-value 0.5 \
        --far 1.2 1.3 1.4 1.5 1.75 2.0 --levels /workspace/data/levels/values/1.00-0.50@500k

The writes fitted inside the window flatten at parity. One explanation is
distance: the arms past parity sit further from the base than anything the
axis was fitted from, and a linear map is not owed anything out there. The
test of that explanation is to fit an axis to the far arms *themselves* -- the
ones that already hold a reversed preference -- and write it inside the regime
they occupy. If distance were the problem, this fit would work where the
window fit does not.

Three readings, mirroring ``014``'s on the window:

* weight space: leave-one-out held-out R^2, cosine and implied offset within
  the far set, and the slope of implied against true offset (``readback``);
  plus the cosine between the far axis and the window axis.
* behaviour, from the base: each far arm written as ``base + drift + o * axis``
  with the axis and drift fitted without it. Drift is included because the far
  fit's intercept is where the far arms sit at offset zero, and that is not the
  base.
* behaviour, from a neighbour: each far arm written from the nearest other far
  arm's weights, ``theta_a + (o - o_a) * axis``, again fitted without it. This
  is the write that never leaves the reversed regime, so it is the one the
  distance explanation predicts should succeed.

A norm-matched random direction from the same neighbour is the control for
the second write.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GOALMISGEN_REPO") or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(REPO))

from goalmisgen import provenance  # noqa: E402
from goalmisgen.analysis.weights import cosine, explained, fit_axis_and_drift, projected_offset  # noqa: E402
from goalmisgen.volume import discover_arms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Agent directory holding BASE.json and arms/.")
    parser.add_argument("--family", type=str, required=True, help="k1 or k0.")
    parser.add_argument("--base-value", type=float, required=True)
    parser.add_argument("--far", type=float, nargs="+", required=True, help="Values of the far arms to fit.")
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument(
        "--window-loo",
        action="store_true",
        help="Instead of the far writes, measure the window's own leave-one-out writes (axis-only from "
        "the base, the paper's convention) with this script's seeds and episodes, so the two regimes "
        "are compared like for like.",
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def readback_slope(true: list[float], implied: list[float]) -> float:
    """Slope of implied offset on true offset; 1 is a perfect scale, 0 is no signal."""
    x, y = np.asarray(true), np.asarray(implied)
    x = x - x.mean()
    return float((x @ (y - y.mean())) / (x @ x)) if float(x @ x) > 0 else float("nan")


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")
    objective = int(args.family[-1])
    window_prefix = f"o{objective}"

    spec = importlib.util.spec_from_file_location(
        "va014", REPO / "experiments" / "014_value_axis_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    va = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(va)

    import jax
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    payload = json.loads((args.run / "BASE.json").read_text())
    checkpoint = args.run / payload["checkpoint"]
    config = va.eval_config(args)
    policy, _, _, base_state, _ = load_train_state(checkpoint, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)

    def diffs_of(selected: dict[float, Path]) -> dict[float, np.ndarray]:
        out = {}
        for value, path in sorted(selected.items()):
            _, _, _, state, _ = load_train_state(path, env_cfg=config)
            flat, _ = ravel_pytree(state.params)
            out[value] = np.asarray(flat, dtype=np.float64) - base_flat
        return out

    window_arms = discover_arms(args.run / "arms", objective, args.base_value, steps=args.arm_steps, at=-1, family="o")
    window = {v: d for v, d in diffs_of(window_arms).items() if abs(v - args.base_value) > 1e-9}
    w_values = sorted(window)
    w_offsets = np.array([v - args.base_value for v in w_values])
    axis_window, drift_window = fit_axis_and_drift(w_offsets, np.stack([window[v] for v in w_values]))

    far_arms_all = discover_arms(args.run / "arms", objective, args.base_value, steps=args.arm_steps, at=-1, family="k")
    wanted = {round(v, 10) for v in args.far}
    far_arms = {v: p for v, p in far_arms_all.items() if round(v, 10) in wanted}
    missing = wanted - {round(v, 10) for v in far_arms}
    if missing:
        sys.exit(f"far arms not found at values {sorted(missing)}; have {sorted(far_arms_all)}")
    far = diffs_of(far_arms)
    f_values = sorted(far)
    f_offsets = np.array([v - args.base_value for v in f_values])
    axis_far, drift_far = fit_axis_and_drift(f_offsets, np.stack([far[v] for v in f_values]))

    print(f"base       {checkpoint}   ({base_flat.size:,} parameters)")
    print(f"window     {window_prefix}: {len(w_values)} arms at offsets {w_offsets.min():+.2f}..{w_offsets.max():+.2f}")
    print(f"far        {args.family}: {len(f_values)} arms at offsets {f_offsets.min():+.2f}..{f_offsets.max():+.2f}\n")

    print("=== the far axis against the window axis ===\n")
    print(f"  |axis| per unit value   window {np.linalg.norm(axis_window):.4g}   far {np.linalg.norm(axis_far):.4g}")
    print(f"  |drift|                 window {np.linalg.norm(drift_window):.4g}   far {np.linalg.norm(drift_far):.4g}")
    print(f"  cos(axis_far, axis_window)    {cosine(axis_far, axis_window):+.3f}")
    print(f"  cos(drift_far, axis_window)   {cosine(drift_far, axis_window):+.3f}")
    print(f"  cos(drift_far, drift_window)  {cosine(drift_far, drift_window):+.3f}")
    print(f"  |far arm| mean {np.mean([np.linalg.norm(far[v]) for v in f_values]):.4g}   |window arm| mean {np.mean([np.linalg.norm(window[v]) for v in w_values]):.4g}")

    print("\n=== graded and predictive? leave-one-out within the far set ===\n")
    print(f"  {'value':>7}{'offset':>8}{'|delta|':>10}{'held-out R^2':>14}{'held-out cos':>14}{'implied':>10}")
    far_loo: dict[float, dict] = {}
    for v in f_values:
        others = [u for u in f_values if u != v]
        held_axis, held_drift = fit_axis_and_drift(
            np.array([u - args.base_value for u in others]), np.stack([far[u] for u in others])
        )
        o = v - args.base_value
        left = far[v] - held_drift
        far_loo[v] = {
            "axis": held_axis,
            "drift": held_drift,
            "r2": explained(left, o, held_axis),
            "cos": cosine(left, held_axis),
            "implied": projected_offset(left, held_axis),
        }
        print(f"  {v:>7.2f}{o:>+8.2f}{np.linalg.norm(far[v]):>10.4g}{far_loo[v]['r2']:>14.3f}{far_loo[v]['cos']:>14.3f}{far_loo[v]['implied']:>+10.2f}")
    far_slope = readback_slope(list(f_offsets), [far_loo[v]["implied"] for v in f_values])

    window_loo_implied = []
    for v in w_values:
        others = [u for u in w_values if u != v]
        held_axis, held_drift = fit_axis_and_drift(
            np.array([u - args.base_value for u in others]), np.stack([window[u] for u in others])
        )
        window_loo_implied.append(projected_offset(window[v] - held_drift, held_axis))
    window_slope = readback_slope(list(w_offsets), window_loo_implied)
    print(f"\n  readback slope (implied on true offset)   window {window_slope:.2f}   far {far_slope:.2f}")

    # the window axis read on the far arms: does it see them as graded at all?
    implied_by_window = [projected_offset(far[v] - drift_window, axis_window) for v in f_values]
    print(f"  far arms read by the window axis: implied {', '.join(f'{x:+.2f}' for x in implied_by_window)}"
          f"   slope {readback_slope(list(f_offsets), implied_by_window):.2f}")

    result = {
        "run": args.run.name,
        "family": args.family,
        "base_value": args.base_value,
        "far_values": f_values,
        "far_offsets": list(map(float, f_offsets)),
        "axis_norm": {"window": float(np.linalg.norm(axis_window)), "far": float(np.linalg.norm(axis_far))},
        "cos_far_window": cosine(axis_far, axis_window),
        "loo": {f"{v:.2f}": {k: float(far_loo[v][k]) for k in ("r2", "cos", "implied")} for v in f_values},
        "readback_slope": {"window": window_slope, "far": far_slope,
                           "far_by_window_axis": readback_slope(list(f_offsets), implied_by_window)},
    }

    if args.skip_behaviour:
        finish(args, result, axis_far)
        return

    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    rng = np.random.default_rng(args.seed)

    if args.window_loo:
        print("\n\n=== the window's own leave-one-out writes, same seeds and episodes ===\n")
        print(f"  {'':>36}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
        wrows = []
        for v in w_values:
            o = v - args.base_value
            others = [u for u in w_values if u != v]
            held_axis, _ = fit_axis_and_drift(
                np.array([u - args.base_value for u in others]), np.stack([window[u] for u in others])
            )
            _, _, _, arm_state, _ = load_train_state(window_arms[v], env_cfg=config)
            arm_point, _, _, arm_reach = va.measure(arm_state.params, policy, get_action, envs, args, f"v={v:.2f} fine-tuned")
            written, _, _, w_reach = va.measure(
                unravel(base_flat + o * held_axis), policy, get_action, envs, args, f"v={v:.2f} written, held out"
            )
            wrows.append({"value": v, "offset": o, "arm": arm_point, "arm_reach": arm_reach, "written": written, "written_reach": w_reach})
        err = np.array([r["written"] - r["arm"] for r in wrows])
        print(f"\n  median |error| {np.median(np.abs(err)):.1f} steps;  slope of written on fine-tuned "
              f"{readback_slope([r['arm'] for r in wrows], [r['written'] for r in wrows]):.2f}")
        result["window_loo"] = {"rows": wrows, "median_abs_error": float(np.median(np.abs(err))),
                                "written_on_arm_slope": readback_slope([r["arm"] for r in wrows], [r["written"] for r in wrows])}
        finish(args, result, axis_far)
        return

    print("\n\n=== written within the reversed regime, each arm from a fit that never saw it ===\n")
    print(f"  {'':>36}{'steps':>8}  {'95% interval':>14}{'reached':>11}")
    rows = []
    for v in f_values:
        o = v - args.base_value
        held_axis, held_drift = far_loo[v]["axis"], far_loo[v]["drift"]
        _, _, _, arm_state, _ = load_train_state(far_arms[v], env_cfg=config)
        arm_point, _, _, arm_reach = va.measure(arm_state.params, policy, get_action, envs, args, f"v={v:.2f} fine-tuned")
        from_base, _, _, base_reach = va.measure(
            unravel(base_flat + held_drift + o * held_axis), policy, get_action, envs, args, f"v={v:.2f} written from base"
        )
        anchor = min((u for u in f_values if u != v), key=lambda u: abs(u - v))
        step = (v - anchor) * held_axis
        from_anchor, _, _, anchor_reach = va.measure(
            unravel(base_flat + far[anchor] + step), policy, get_action, envs, args, f"v={v:.2f} written from v={anchor:.2f}"
        )
        direction = rng.normal(size=step.size)
        direction *= np.linalg.norm(step) / np.linalg.norm(direction)
        random_point, _, _, random_reach = va.measure(
            unravel(base_flat + far[anchor] + direction), policy, get_action, envs, args, f"v={v:.2f} random from v={anchor:.2f}"
        )
        rows.append({
            "value": v, "offset": o, "anchor": anchor,
            "arm": arm_point, "arm_reach": arm_reach,
            "from_base": from_base, "from_base_reach": base_reach,
            "from_anchor": from_anchor, "from_anchor_reach": anchor_reach,
            "random_from_anchor": random_point, "random_reach": random_reach,
        })

    print(f"\n  {'value':>7}{'fine-tuned':>12}{'from base':>11}{'err':>7}{'from nbr':>10}{'err':>7}{'random':>9}{'err':>7}{'nbr arm':>9}")
    for r in rows:
        nbr = next(x["arm"] for x in rows if x["value"] == r["anchor"])
        print(
            f"  {r['value']:>7.2f}{r['arm']:>12.1f}{r['from_base']:>11.1f}{r['from_base'] - r['arm']:>+7.1f}"
            f"{r['from_anchor']:>10.1f}{r['from_anchor'] - r['arm']:>+7.1f}{r['random_from_anchor']:>9.1f}{r['random_from_anchor'] - r['arm']:>+7.1f}{nbr:>9.1f}"
        )
    arms_ = np.array([r["arm"] for r in rows])
    err_base = np.array([r["from_base"] - r["arm"] for r in rows])
    err_anchor = np.array([r["from_anchor"] - r["arm"] for r in rows])
    err_random = np.array([r["random_from_anchor"] - r["arm"] for r in rows])
    # what the write had to achieve: the gap between the target arm and the arm it started from
    asked = np.array([r["arm"] - next(x["arm"] for x in rows if x["value"] == r["anchor"]) for r in rows])
    moved = np.array([r["from_anchor"] - next(x["arm"] for x in rows if x["value"] == r["anchor"]) for r in rows])
    print(
        f"\n  median |error|   from base {np.median(np.abs(err_base)):.1f}   from neighbour {np.median(np.abs(err_anchor)):.1f}"
        f"   random {np.median(np.abs(err_random)):.1f} steps"
    )
    print(f"  neighbour write: asked to move {np.median(np.abs(asked)):.1f} steps (median), moved {np.median(np.abs(moved)):.1f};"
          f" slope of moved on asked {readback_slope(list(asked), list(moved)):.2f}")
    print(f"  slope of written on fine-tuned across far arms   from base {readback_slope(list(arms_), [r['from_base'] for r in rows]):.2f}"
          f"   from neighbour {readback_slope(list(arms_), [r['from_anchor'] for r in rows]):.2f}")
    result["behaviour"] = {
        "rows": rows,
        "median_abs_error": {"from_base": float(np.median(np.abs(err_base))), "from_anchor": float(np.median(np.abs(err_anchor))),
                             "random": float(np.median(np.abs(err_random)))},
        "asked_median": float(np.median(np.abs(asked))), "moved_median": float(np.median(np.abs(moved))),
        "moved_on_asked_slope": readback_slope(list(asked), list(moved)),
        "written_on_arm_slope": {"from_base": readback_slope(list(arms_), [r["from_base"] for r in rows]),
                                 "from_anchor": readback_slope(list(arms_), [r["from_anchor"] for r in rows])},
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
