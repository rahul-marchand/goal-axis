"""What a ResNet or a ViT propagates, and where value enters it.

The feedforward counterpart of ``scripts/field_decomp.py``. Same four erasure
variants, same decomposition, same fits -- but these networks have no carry, so
one forward pass is the whole measurement and **depth replaces the DRC's
ticks**: a ResNet stage or a transformer block in place of a tick of the
recurrence. Reported per layer, in forward order.

1. **baseline distance** -- how far the objective-free observation sits from an
   ordinary one, which bounds how much the subtraction can be trusted.
2. **combination** ``D ~ alpha A_near + gamma A_far`` where both fields are
   present: ``gamma`` near 1 is superposition, near 0 winner-take-all.
3. **where value enters** -- ``dh(both) = dh(neither) + dA_rich + dA_poor +
   dI`` under the fitted axis and under a colour gain. The gain is a source
   change *by construction* (it scales one colour plane of the observation), so
   it calibrates where a source change lands. It does not calibrate how large a
   threshold move that is -- on the ViT a colour-1 gain moves the threshold the
   wrong way, and on the ResNet brightening barely moves it at all.
4. **field shape and propagation** -- the field's size against BFS distance
   from its own objective, per layer, fitted as an exponential and as a
   straight line, with the linear fit's R^2 beside the log fit's. The BFS
   against grid comparison is restricted to cells where the two distances
   differ by at least ``--differ``: everywhere else they agree, and a fit over
   all cells cannot separate routing from blurring.
5. **bridge** -- the two fields' sizes at the agent's own cell against the
   distance gap the level presents, with the model's measured threshold beside
   the crossing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.feedforward_fields import (
    depth_grids,
    differing_cells,
    linear_fit_r2,
    observations_for,
)
from goalmisgen.analysis.field_decomp import (
    NEITHER,
    cell_norms,
    chebyshev_distance,
    combination_fit,
    decompose,
    energies,
    erasure_statics,
    field_profile,
)
from goalmisgen.analysis.geometry import free_cells
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.provenance import header

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-levels", type=int, default=256)
    parser.add_argument("--offset", type=float, default=0.30, help="Axis dose for the value manipulation.")
    parser.add_argument(
        "--gain", type=float, default=0.65, help="Colour-1 gain for the source-change control; 1/x is used for the other sign."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float("nan"),
        help="The model's measured threshold, printed beside the bridge crossing.",
    )
    parser.add_argument(
        "--differ", type=int, default=3, help="Keep cells where |BFS - grid| >= this for the routing-against-blur comparison."
    )
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--batch", type=int, default=128, help="Levels per forward pass.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def grids_in_batches(reader, params, statics, positions, gain, batch: int, seed: int) -> list[np.ndarray]:
    """``depth_grids`` over level batches, concatenated: ``[(4, N, S, S, K)]``."""
    chunks: list[list[np.ndarray]] = []
    for start in range(0, statics.shape[1], batch):
        stop = min(start + batch, statics.shape[1])
        observations = observations_for(statics[:, start:stop], positions[start:stop], gain)
        chunks.append(depth_grids(reader, params, observations, seed=seed))
    return [np.concatenate([chunk[layer] for chunk in chunks], axis=1) for layer in range(len(chunks[0]))]


def main() -> None:
    args = parse_args()
    print(header())

    from goalmisgen.analysis.route_targets import route_targets, usable
    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.axis import load_base, load_sweep
    from goalmisgen.nets.readers import state_reader_for

    base = load_base(args.run, args.levels, split=args.split, num_envs=2, seed=args.seed)
    checkpoint, values, config, policy, unravel = base.checkpoint, base.values, base.config, base.policy, base.unravel
    base_flat = np.asarray(base.raveled, dtype=np.float32)
    reader = state_reader_for(policy)
    print(f"base   {checkpoint}, values {values}")
    print(f"reader {type(reader).__name__}: {len(reader.layer_names)} layers {list(reader.layer_names)}")

    sampler = config.sampler()
    rng = np.random.default_rng(args.seed)
    levels = []
    while len(levels) < args.n_levels:
        level = sampler.sample(rng)
        if usable(level):
            levels.append(level)
    targets = route_targets(levels, size=11)
    print(f"levels {len(targets)} from {args.levels} [{args.split}]; depth replaces the DRC's ticks\n")

    offsets, diffs = load_sweep(base, args.sweep, arm_steps=args.arm_steps, fit_window=args.fit_window)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    axis = np.asarray(axis, dtype=np.float32)
    print(f"sweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}\n")

    statics = erasure_statics(targets)
    positions = np.asarray(targets.positions[:, 0, 0], dtype=np.int32)
    mask = free_cells(targets.static)
    bfs = np.asarray(targets.fields)  # (N, 2, S, S): 0 = rich, 1 = poor
    grid_distance = chebyshev_distance(targets)
    layers = list(reader.layer_names)

    def gains(colour_gain: float) -> np.ndarray:
        out = np.ones(statics.shape[-1], dtype=np.float32)
        out[COLOUR[1]] = colour_gain
        return out

    ones = np.ones(statics.shape[-1], dtype=np.float32)
    variants = {
        "base": (base_flat, ones),
        "axis +": (base_flat + args.offset * axis, ones),
        "axis -": (base_flat - args.offset * axis, ones),
        "gain +": (base_flat, gains(args.gain)),
        "gain -": (base_flat, gains(1.0 / args.gain)),
    }
    captured = {}
    for name, (flat, gain) in variants.items():
        captured[name] = grids_in_batches(reader, unravel(flat), statics, positions, gain, args.batch, args.seed)
        print(f"  ran {name}", flush=True)

    report: dict = {
        "run": str(args.run),
        "checkpoint": str(checkpoint),
        "values": list(values),
        "levels": len(targets),
        "offset": args.offset,
        "gain": args.gain,
        "threshold": args.threshold,
        "layers": layers,
        "rows": [],
    }

    print("\n=== how off-distribution is the 'neither' baseline? rms(h_neither - h_both) / rms(h_both) ===\n")
    for k, name in enumerate(layers):
        site = captured["base"][k]
        both, none = site[0], site[NEITHER]
        ratio = float(np.sqrt(((none - both) ** 2).mean()) / np.sqrt((both**2).mean()))
        print(f"  {name:>18}   {ratio:6.3f}")
        report.setdefault("baseline_distance", {})[name] = ratio

    print("\n=== combination: D ~ alpha A_near + gamma A_far ===")
    print("    superposition is (1, 1); winner-take-all is (1, 0)\n")
    print(f"  {'layer':>18} {'alpha':>7} {'gamma':>7} {'R^2':>6} {'cells':>8}  {'|I|/|D|':>8}")
    near_is_rich = bfs[:, 0] <= bfs[:, 1]
    for k, name in enumerate(layers):
        a_rich, a_poor, total, inter = decompose(captured["base"][k])
        a_near = np.where(near_is_rich[..., None], a_rich, a_poor)
        a_far = np.where(near_is_rich[..., None], a_poor, a_rich)
        both_present = mask & (cell_norms(a_rich) > 0) & (cell_norms(a_poor) > 0)
        fit = combination_fit(a_near, a_far, total, both_present)
        share_terms = energies(a_rich, a_poor, total, inter, both_present)
        share = float(np.sqrt(share_terms["interaction"] / max(share_terms["total"], 1e-30)))
        print(f"  {name:>18} {fit['alpha']:7.3f} {fit['gamma']:7.3f} {fit['r2']:6.3f} " f"{fit['cells']:8.0f}  {share:8.3f}")
        report["rows"].append({"analysis": "combination", "layer": name, **fit, "interaction_share": share})

    print("\n=== where a value manipulation lands: share of its energy in each term ===")
    print("    dh(both) = dh(neither) + dA_rich + dA_poor + dI")
    print("    'gain' scales colour 1's plane, so it is a source change by construction\n")
    print(f"  {'layer':>18} {'variant':>8} {'neither':>8} {'dA_rich':>8} {'dA_poor':>8} {'dI':>8} {'rms dh':>8}")
    for k, name in enumerate(layers):
        for variant in ("axis +", "axis -", "gain +", "gain -"):
            base_site, moved = captured["base"][k], captured[variant][k]
            base_parts, moved_parts = decompose(base_site), decompose(moved)
            d_none = moved[NEITHER] - base_site[NEITHER]
            d_rich = moved_parts[0] - base_parts[0]
            d_poor = moved_parts[1] - base_parts[1]
            d_inter = moved_parts[3] - base_parts[3]
            total_change = moved[0] - base_site[0]
            weigh = lambda x: float(((x**2).sum(-1) * mask).sum())  # noqa: E731
            denominator = max(weigh(d_none) + weigh(d_rich) + weigh(d_poor) + weigh(d_inter), 1e-30)
            row = {
                "analysis": "value",
                "layer": name,
                "variant": variant,
                "neither": weigh(d_none) / denominator,
                "rich": weigh(d_rich) / denominator,
                "poor": weigh(d_poor) / denominator,
                "interaction": weigh(d_inter) / denominator,
                "rms": float(np.sqrt((total_change**2).mean())),
            }
            print(
                f"  {name:>18} {variant:>8} {row['neither']:8.3f} {row['rich']:8.3f} "
                f"{row['poor']:8.3f} {row['interaction']:8.3f} {row['rms']:8.4f}"
            )
            report["rows"].append(row)

    print("\n=== field shape and propagation: A_rich against distance from the rich objective ===")
    print("    log slope fits exp(-lambda d); linear slope fits a cost-subtracted utility")
    print(f"    'differ' columns keep only cells where |BFS - grid| >= {args.differ}\n")
    print(
        f"  {'layer':>18} {'log slope':>10} {'log R^2':>8} {'lin slope':>10} {'lin R^2':>8} {'half d':>7} "
        f"{'BFS R^2':>8} {'grid R^2':>8} {'cells':>7}"
    )
    differ = differing_cells(bfs[:, 0], grid_distance[:, 0], args.differ) & mask
    for k, name in enumerate(layers):
        a_rich = decompose(captured["base"][k])[0]
        by_bfs = field_profile(a_rich, bfs[:, 0], mask)
        linear_r2 = linear_fit_r2(by_bfs)
        on_bfs = field_profile(a_rich, bfs[:, 0], differ)
        on_grid = field_profile(a_rich, grid_distance[:, 0], differ)
        print(
            f"  {name:>18} {by_bfs['log_slope']:10.3f} {by_bfs['log_r2']:8.3f} "
            f"{by_bfs['linear_slope']:10.4f} {linear_r2:8.3f} {by_bfs['half_distance']:7.1f} "
            f"{on_bfs['log_r2']:8.3f} {on_grid['log_r2']:8.3f} {int(differ.sum()):7d}"
        )
        report["rows"].append(
            {
                "analysis": "profile",
                "layer": name,
                "bfs": by_bfs,
                "linear_r2": linear_r2,
                "bfs_differ": on_bfs,
                "grid_differ": on_grid,
                "differ_cells": int(differ.sum()),
            }
        )

    print("\n=== bridge: the two fields at the agent's own cell, against the level's gap ===\n")
    start = positions
    rows = np.arange(len(targets))
    for k, name in enumerate(layers):
        a_rich, a_poor, _, _ = decompose(captured["base"][k])
        size_rich = cell_norms(a_rich)[rows, start[:, 0], start[:, 1]]
        size_poor = cell_norms(a_poor)[rows, start[:, 0], start[:, 1]]
        good = (size_rich > 0) & (size_poor > 0)
        if good.sum() < 20:
            print(f"  {name:>18}   fields reach the agent on {int(good.sum())} levels; skipped")
            continue
        delta = np.log(size_rich[good]) - np.log(size_poor[good])
        gap = np.asarray(targets.gap)[good]
        slope, intercept = np.polyfit(gap, delta, 1)
        correlation = float(np.corrcoef(gap, delta)[0, 1])
        crossing = float(-intercept / slope) if abs(slope) > 1e-9 else float("nan")
        print(
            f"  {name:>18}   corr {correlation:+.3f}   slope {slope:+.4f} per step   "
            f"zero at gap {crossing:+.2f}   n {int(good.sum())}   (threshold {args.threshold:.2f})"
        )
        report["rows"].append(
            {
                "analysis": "bridge",
                "layer": name,
                "corr": correlation,
                "slope": float(slope),
                "crossing": crossing,
                "n": int(good.sum()),
                "threshold": args.threshold,
            }
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    print("FIELD_DECOMP_DONE")


if __name__ == "__main__":
    main()
