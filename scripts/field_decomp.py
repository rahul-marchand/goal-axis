"""What the DRC propagates, and where value enters it.

Runs the four erasure variants -- both objectives, only the richer, only the
poorer, neither -- on the same teacher-forced positions, and reports:

1. **combination** ``D ~ alpha A_near + gamma A_far``, fitted where both fields
   are present. ``gamma`` near 1 is superposition, near 0 winner-take-all.
2. **where value enters**: the same decomposition under a value manipulation,
   ``dh(both) = dh(neither) + dA_rich + dA_poor + dI``. Weight in ``dA_c`` means
   value rides the field; weight in ``dI`` means it acts at the comparison.
3. **field shape**: the field's size against BFS distance from its own
   objective, fitted as an exponential and as a straight line.
4. **propagation**: the same per tick, and against grid distance rather than
   BFS distance -- a convolution blurs in grid distance whatever the walls do,
   so a field that tracks the maze is routing rather than blurring.
5. **bridge**: the two fields' sizes at the agent's own cell against the
   distance gap the level presents.

The "neither" baseline is off distribution -- training never showed a maze with
no objective at all -- so the report also states how far those activations sit
from ordinary ones, which bounds how much the subtraction can be trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.field_decomp import (  # noqa: E402
    NEITHER,
    cell_norms,
    chebyshev_distance,
    combination_fit,
    decompose,
    energies,
    erasure_statics,
    field_profile,
    make_grid_fn,
)
from goalmisgen.analysis.gain_alignment import sites_of  # noqa: E402
from goalmisgen.analysis.geometry import free_cells  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-levels", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=4, help="Teacher-forced steps read, from the start.")
    parser.add_argument("--offset", type=float, default=0.30, help="Axis dose for the value manipulation.")
    parser.add_argument("--bright", type=float, default=1.54,
                        help="Colour-1 gain standing in for +offset of the axis; 1/x is used for -offset.")
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    payload = json.loads((args.run / "BASE.json").read_text())
    checkpoint = args.run / payload["checkpoint"]
    values = tuple(payload["values"])

    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis.route_targets import route_targets, usable
    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.configs.env import MazeConfig

    config = MazeConfig(
        max_episode_steps=120, num_envs=2, min_size=11, max_size=11, n_objectives=len(values),
        objective_values=values, feature_value_correlation=1.0, value_encoding="none",
        colour_is_the_only_value_cue=True, level_dataset=args.levels, dataset_split=args.split,
        asynchronous=False, seed=args.seed,
    )
    policy, _, _, base_state, _ = load_train_state(checkpoint, env_cfg=config)
    base_flat, unravel = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float32)

    sampler = config.sampler()
    rng = np.random.default_rng(args.seed)
    levels = []
    while len(levels) < args.n_levels:
        level = sampler.sample(rng)
        if usable(level):
            levels.append(level)
    targets = route_targets(levels, size=11)
    print(f"base   {checkpoint}, values {values}")
    print(f"levels {len(targets)} from {args.levels} [{args.split}], {args.n_steps} teacher-forced steps\n")

    from scripts.restricted_writes import load_sweep

    offsets, diffs = load_sweep(args.run, args.sweep, values, args, config, base_flat)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    axis = np.asarray(axis, dtype=np.float32)
    print(f"sweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}\n")

    sites = sites_of(policy.cfg)
    spatial = tuple(i for i, s in enumerate(sites) if s.group not in ("dense", "logits"))
    grids = make_grid_fn(policy, unravel, n_steps=args.n_steps, site_index=spatial)
    statics = erasure_statics(targets)
    positions = np.asarray(targets.positions[:, 0], dtype=np.int32)
    mask = free_cells(targets.static)
    bfs = np.asarray(targets.fields)          # (N, 2, S, S): 0 = rich, 1 = poor
    grid_distance = chebyshev_distance(targets)
    ones = np.ones(statics.shape[-1], dtype=np.float32)

    def gains(colour_gain: float) -> np.ndarray:
        out = np.ones(statics.shape[-1], dtype=np.float32)
        out[COLOUR[1]] = colour_gain
        return out

    variants = {
        "base": (base_flat, ones),
        "axis +": (base_flat + args.offset * axis, ones),
        "axis -": (base_flat - args.offset * axis, ones),
        "bright +": (base_flat, gains(args.bright)),
        "bright -": (base_flat, gains(1.0 / args.bright)),
    }
    captured = {}
    for name, (flat, gain) in variants.items():
        out = grids(flat, gain, statics, positions)
        captured[name] = [np.asarray(site, dtype=np.float32) for site in out]
        print(f"  ran {name}", flush=True)

    report: dict = {"run": str(args.run), "checkpoint": str(checkpoint), "values": list(values),
                    "levels": len(targets), "n_steps": args.n_steps, "offset": args.offset,
                    "bright": args.bright, "sites": [sites[i].name for i in spatial], "rows": []}

    # How far the objective-free baseline sits from an ordinary observation.
    print("=== how off-distribution is the 'neither' baseline? rms(h_neither - h_both) / rms(h_both) ===\n")
    for k, index in enumerate(spatial):
        site = captured["base"][k]
        both, none = site[:, 0], site[:, NEITHER]
        ratio = float(np.sqrt(((none - both) ** 2).mean()) / np.sqrt((both ** 2).mean()))
        print(f"  {sites[index].name:>18}   {ratio:6.3f}")
        report.setdefault("baseline_distance", {})[sites[index].name] = ratio

    print("\n=== combination: D ~ alpha A_near + gamma A_far, last step ===")
    print("    superposition is (1, 1); winner-take-all is (1, 0)\n")
    print(f"  {'site':>18} {'alpha':>7} {'gamma':>7} {'R^2':>6} {'cells':>8}  {'|I|/|D|':>8}")
    step = args.n_steps - 1
    for k, index in enumerate(spatial):
        a_rich, a_poor, total, inter = decompose(captured["base"][k][step])
        near_is_rich = bfs[:, 0] <= bfs[:, 1]
        a_near = np.where(near_is_rich[..., None], a_rich, a_poor)
        a_far = np.where(near_is_rich[..., None], a_poor, a_rich)
        both_present = mask & (cell_norms(a_rich) > 0) & (cell_norms(a_poor) > 0)
        fit = combination_fit(a_near, a_far, total, both_present)
        share = float(np.sqrt(energies(a_rich, a_poor, total, inter, both_present)["interaction"]
                              / max(energies(a_rich, a_poor, total, inter, both_present)["total"], 1e-30)))
        print(f"  {sites[index].name:>18} {fit['alpha']:7.3f} {fit['gamma']:7.3f} {fit['r2']:6.3f} "
              f"{fit['cells']:8.0f}  {share:8.3f}")
        report["rows"].append({"analysis": "combination", "site": sites[index].name, **fit, "interaction_share": share})

    print("\n=== where a value manipulation lands: share of its energy in each term ===")
    print("    dh(both) = dh(neither) + dA_rich + dA_poor + dI\n")
    print(f"  {'site':>18} {'variant':>9} {'neither':>8} {'dA_rich':>8} {'dA_poor':>8} {'dI':>8} {'rms dh':>8}")
    for k, index in enumerate(spatial):
        for name in ("axis +", "axis -", "bright +", "bright -"):
            base_site, moved = captured["base"][k][step], captured[name][k][step]
            base_parts, moved_parts = decompose(base_site), decompose(moved)
            d_none = moved[NEITHER] - base_site[NEITHER]
            d_rich = moved_parts[0] - base_parts[0]
            d_poor = moved_parts[1] - base_parts[1]
            d_inter = moved_parts[3] - base_parts[3]
            total_change = moved[0] - base_site[0]
            weigh = lambda x: float(((x ** 2).sum(-1) * mask).sum())  # noqa: E731
            denominator = max(weigh(d_none) + weigh(d_rich) + weigh(d_poor) + weigh(d_inter), 1e-30)
            row = {"analysis": "value", "site": sites[index].name, "variant": name,
                   "neither": weigh(d_none) / denominator, "rich": weigh(d_rich) / denominator,
                   "poor": weigh(d_poor) / denominator, "interaction": weigh(d_inter) / denominator,
                   "rms": float(np.sqrt((total_change ** 2).mean()))}
            print(f"  {sites[index].name:>18} {name:>9} {row['neither']:8.3f} {row['rich']:8.3f} "
                  f"{row['poor']:8.3f} {row['interaction']:8.3f} {row['rms']:8.4f}")
            report["rows"].append(row)

    print("\n=== field shape and propagation: A_rich against distance from the rich objective ===")
    print("    log slope fits exp(-lambda d); linear slope fits a cost-subtracted utility\n")
    print(f"  {'site':>18} {'step':>4} {'log slope':>10} {'log R^2':>8} {'lin slope':>10} {'half d':>7} "
          f"{'BFS R^2':>8} {'grid R^2':>8}")
    for k, index in enumerate(spatial):
        for t in range(args.n_steps):
            a_rich = decompose(captured["base"][k][t])[0]
            by_bfs = field_profile(a_rich, bfs[:, 0], mask)
            by_grid = field_profile(a_rich, grid_distance[:, 0], mask)
            print(f"  {sites[index].name:>18} {t:4d} {by_bfs['log_slope']:10.3f} {by_bfs['log_r2']:8.3f} "
                  f"{by_bfs['linear_slope']:10.4f} {by_bfs['half_distance']:7.1f} {by_bfs['log_r2']:8.3f} "
                  f"{by_grid['log_r2']:8.3f}")
            report["rows"].append({"analysis": "profile", "site": sites[index].name, "step": t,
                                   "bfs": by_bfs, "grid": by_grid})

    print("\n=== bridge: the two fields at the agent's own cell, against the level's gap ===\n")
    start = positions[:, 0]
    rows = np.arange(len(targets))
    for k, index in enumerate(spatial):
        a_rich, a_poor, _, _ = decompose(captured["base"][k][step])
        size_rich = cell_norms(a_rich)[rows, start[:, 0], start[:, 1]]
        size_poor = cell_norms(a_poor)[rows, start[:, 0], start[:, 1]]
        good = (size_rich > 0) & (size_poor > 0)
        if good.sum() < 20:
            print(f"  {sites[index].name:>18}   fields reach the agent on {int(good.sum())} levels; skipped")
            continue
        delta = np.log(size_rich[good]) - np.log(size_poor[good])
        gap = np.asarray(targets.gap)[good]
        slope, intercept = np.polyfit(gap, delta, 1)
        correlation = float(np.corrcoef(gap, delta)[0, 1])
        crossing = float(-intercept / slope) if abs(slope) > 1e-9 else float("nan")
        print(f"  {sites[index].name:>18}   corr {correlation:+.3f}   slope {slope:+.4f} per step   "
              f"zero at gap {crossing:+.2f}   n {int(good.sum())}")
        report["rows"].append({"analysis": "bridge", "site": sites[index].name, "corr": correlation,
                               "slope": float(slope), "crossing": crossing, "n": int(good.sum())})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    print("FIELD_DECOMP_DONE")


if __name__ == "__main__":
    main()
