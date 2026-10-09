"""What the route model propagates, and where value enters it.

E12 ran this decomposition on the DRC. This is the same five measurements on a
route model, so the two architectures can be read side by side:

1. **combination** ``D ~ alpha A_near + gamma A_far`` -- superposition is
   ``(1, 1)``, winner-take-all ``(1, 0)``.
2. **where a value manipulation lands**, ``dh(both) = dh(neither) + dA_rich +
   dA_poor + dI``. Weight on ``dA_c`` means value rides the objective's own
   field; weight on ``dI`` means it acts where the two meet.
3. **field shape**, the field's size against BFS distance from its objective,
   fitted as an exponential *and* as a straight line, both R^2 reported.
4. **propagation**, the same per block, and BFS against grid distance on the
   cells where the two disagree -- that is where routing and blur differ.
5. **bridge**, the two fields at the agent's own cell against the level's gap.

The value manipulation is the fitted axis written into the weights. Its
calibration is objective scaling: multiplying the poorer objective's own cell,
which is a source change by construction. A route model answers that
non-monotonically in threshold terms, so it cannot be dose-matched to the axis
the way E12 matched brightness -- it calibrates *where* a source change lands,
not how big the threshold move is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.field_decomp import (
    NEITHER,
    cell_norms,
    combination_fit,
    decompose,
    energies,
)
from goalmisgen.analysis.geometry import free_cells
from goalmisgen.analysis.route_fields import (
    cell_grids,
    erasure_observations,
    grid_distance_from,
    objective_channels,
    profile,
    scale_objective,
)
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test.rho100")
    parser.add_argument("--n-levels", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=64, help="Levels per forward pass; d512 OOMs on large batches.")
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--steps-tag", type=int, default=1000)
    parser.add_argument("--offset", type=float, default=0.30)
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Objective-scaling control: the poorer objective's cell times this, and its reciprocal.")
    parser.add_argument("--differ", type=int, default=3,
                        help="Routing vs blur: keep cells where |BFS - grid| is at least this.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())

    import jax.numpy as jnp

    from goalmisgen.analysis.route_targets import route_targets, usable
    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
    from goalmisgen.offline.decode import evaluate
    from goalmisgen.offline.demos import DemoSet
    from goalmisgen.offline.gap_probe import _prefix_residuals_fn

    base = load_base(args.run)
    cfg = base.model.config
    print(f"base   {args.run.name} @ step {base.step:,} ({base.flat.size:,} parameters), values {base.values}")

    arms = arm_dirs(args.run, args.sweep, args.steps_tag)
    if not arms:
        raise SystemExit(f"no finished {args.sweep} arms at {args.steps_tag} steps under {args.run}/arms")
    diffs = load_diffs(base, arms)
    offsets = np.array(sorted(diffs), dtype=np.float64)
    axis, _ = fit_axis_and_drift(offsets, np.stack([diffs[o] for o in offsets]))
    del diffs
    axis = np.asarray(axis, dtype=np.float32)
    print(f"axis   {args.sweep}: {len(offsets)} arms, |v| {np.linalg.norm(axis):.3f} per unit value")

    demos = DemoSet.load(args.demos / args.split, hide_values=base.hide_values)
    chosen = [i for i in range(len(demos)) if usable(demos.level(i))][: args.n_levels]
    indices = np.asarray(chosen)
    if len(indices) < args.n_levels:
        print(f"note   only {len(indices)} usable levels in {args.split}")
    targets = route_targets([demos.level(int(i)) for i in indices], size=cfg.size, n_channels=demos.n_channels)
    bfs = np.asarray(targets.fields)                    # (B, 2, S, S), 0 = rich
    gap = np.asarray(targets.gap)
    mask = free_cells(demos.observations(np.asarray(indices)))
    rich, poor = objective_channels(demos, indices)
    grid = np.stack([grid_distance_from(rich[:, 1:], cfg.size), grid_distance_from(poor[:, 1:], cfg.size)], axis=1)
    agent = np.asarray(demos.agent)[indices].astype(np.int64)
    print(f"levels {len(indices)} from {args.split}; {cfg.size}x{cfg.size} cell tokens + SEP\n")

    summary, _, _ = evaluate(base.model, base.params, demos, indices)
    tau_base = float(summary.indifference)
    print(f"threshold {tau_base:.2f} steps on these levels\n")

    residuals_fn = _prefix_residuals_fn(cfg)
    statics = erasure_observations(demos, indices)       # (4, B, S, S, C)

    def grids(params, observations: np.ndarray) -> np.ndarray:
        """``(4, B, S, S, C) -> (depths, 4, B, S, S, W)``, in chunks."""
        per_variant = []
        for variant in range(observations.shape[0]):
            parts = []
            for start in range(0, observations.shape[1], args.chunk):
                window = jnp.asarray(observations[variant, start : start + args.chunk])
                parts.append(np.asarray(residuals_fn(params, window), dtype=np.float32))
            streams = np.concatenate(parts, axis=1)
            per_variant.append(cell_grids(streams, cfg.size))
        return np.stack(per_variant, axis=1)

    written = {sign: base.unravel(np.asarray(base.flat + sign * args.offset * axis, dtype=np.float32))
               for sign in (+1.0, -1.0)}
    variants: dict[str, np.ndarray] = {}
    variants["base"] = grids(base.params, statics)
    print("  ran base", flush=True)
    for sign, name in ((+1.0, "axis +"), (-1.0, "axis -")):
        variants[name] = grids(written[sign], statics)
        print(f"  ran {name}", flush=True)
    for factor, name in ((args.scale, "scale +"), (1.0 / args.scale, "scale -")):
        variants[name] = grids(base.params, scale_objective(statics, poor, factor))
        print(f"  ran {name}", flush=True)

    depths = variants["base"].shape[0]
    names = ["embed"] + [f"block {i}" for i in range(depths - 1)]
    report: dict = {"run": str(args.run), "levels": int(len(indices)), "offset": args.offset,
                    "scale": args.scale, "axis_norm": float(np.linalg.norm(axis)), "tau_base": tau_base,
                    "depths": names, "rows": []}

    print("=== how off-distribution is the 'neither' baseline? rms(h_neither - h_both) / rms(h_both) ===\n")
    for d, name in enumerate(names):
        site = variants["base"][d]
        ratio = float(np.sqrt(((site[NEITHER] - site[0]) ** 2).mean()) / np.sqrt((site[0] ** 2).mean()))
        print(f"  {name:>10}   {ratio:6.3f}")
        report.setdefault("baseline_distance", {})[name] = ratio

    print("\n=== combination: D ~ alpha A_near + gamma A_far ===")
    print("    superposition is (1, 1); winner-take-all is (1, 0)\n")
    print(f"  {'depth':>10} {'alpha':>7} {'gamma':>7} {'R^2':>6} {'cells':>8}  {'|I|/|D|':>8}")
    near_is_rich = bfs[:, 0] <= bfs[:, 1]
    for d, name in enumerate(names):
        a_rich, a_poor, total, inter = decompose(variants["base"][d])
        a_near = np.where(near_is_rich[..., None], a_rich, a_poor)
        a_far = np.where(near_is_rich[..., None], a_poor, a_rich)
        both_present = mask & (cell_norms(a_rich) > 0) & (cell_norms(a_poor) > 0)
        fit = combination_fit(a_near, a_far, total, both_present)
        packet = energies(a_rich, a_poor, total, inter, both_present)
        share = float(np.sqrt(packet["interaction"] / max(packet["total"], 1e-30)))
        print(f"  {name:>10} {fit['alpha']:7.3f} {fit['gamma']:7.3f} {fit['r2']:6.3f} {fit['cells']:8.0f}  {share:8.3f}")
        report["rows"].append({"analysis": "combination", "depth": name, **fit, "interaction_share": share})

    print("\n=== where a value manipulation lands: share of its energy in each term ===")
    print("    dh(both) = dh(neither) + dA_rich + dA_poor + dI\n")
    print(f"  {'depth':>10} {'variant':>9} {'neither':>8} {'dA_rich':>8} {'dA_poor':>8} {'dI':>8} {'rms dh':>8}")
    weigh = lambda x: float(((np.asarray(x, dtype=np.float64) ** 2).sum(-1) * mask).sum())  # noqa: E731
    for d, name in enumerate(names):
        for variant in ("axis +", "axis -", "scale +", "scale -"):
            here, moved = variants["base"][d], variants[variant][d]
            base_parts, moved_parts = decompose(here), decompose(moved)
            terms = {"neither": moved[NEITHER] - here[NEITHER],
                     "rich": moved_parts[0] - base_parts[0],
                     "poor": moved_parts[1] - base_parts[1],
                     "interaction": moved_parts[3] - base_parts[3]}
            shares = {k: weigh(v) for k, v in terms.items()}
            denominator = max(sum(shares.values()), 1e-30)
            row = {"analysis": "value", "depth": name, "variant": variant,
                   **{k: v / denominator for k, v in shares.items()},
                   "rms": float(np.sqrt(((moved[0] - here[0]) ** 2).mean()))}
            print(f"  {name:>10} {variant:>9} {row['neither']:8.3f} {row['rich']:8.3f} {row['poor']:8.3f} "
                  f"{row['interaction']:8.3f} {row['rms']:8.4f}")
            report["rows"].append(row)

    print("\n=== field shape and propagation: A_rich against distance from the rich objective ===")
    print("    'differ' columns keep only cells where |BFS - grid| >= "
          f"{args.differ}, where routing and blur disagree\n")
    print(f"  {'depth':>10} {'log slope':>10} {'log R^2':>8} {'lin slope':>10} {'lin R^2':>8} {'half d':>7} "
          f"{'BFS R^2':>8} {'grid R^2':>8} {'cells':>7}")
    differ = mask & (np.abs(bfs[:, 0].astype(np.int64) - grid[:, 0].astype(np.int64)) >= args.differ) & (bfs[:, 0] < 10_000)
    for d, name in enumerate(names):
        a_rich = decompose(variants["base"][d])[0]
        by_bfs = profile(a_rich, bfs[:, 0], mask)
        bfs_differ = profile(a_rich, bfs[:, 0], differ)
        grid_differ = profile(a_rich, grid[:, 0], differ)
        print(f"  {name:>10} {by_bfs['log_slope']:10.3f} {by_bfs['log_r2']:8.3f} {by_bfs['linear_slope']:10.4f} "
              f"{by_bfs['linear_r2']:8.3f} {by_bfs['half_distance']:7.1f} {bfs_differ['log_r2']:8.3f} "
              f"{grid_differ['log_r2']:8.3f} {bfs_differ['cells']:7d}")
        report["rows"].append({"analysis": "profile", "depth": name, "bfs": by_bfs,
                               "bfs_differ": bfs_differ, "grid_differ": grid_differ})

    print("\n=== bridge: the two fields at the agent's own cell, against the level's gap ===\n")
    rows = np.arange(len(indices))
    for d, name in enumerate(names):
        a_rich, a_poor, _, _ = decompose(variants["base"][d])
        size_rich = cell_norms(a_rich)[rows, agent[:, 0], agent[:, 1]]
        size_poor = cell_norms(a_poor)[rows, agent[:, 0], agent[:, 1]]
        good = (size_rich > 0) & (size_poor > 0)
        if good.sum() < 20:
            print(f"  {name:>10}   fields reach the agent on {int(good.sum())} levels; skipped")
            continue
        delta = np.log(size_rich[good]) - np.log(size_poor[good])
        slope, intercept = np.polyfit(gap[good], delta, 1)
        correlation = float(np.corrcoef(gap[good], delta)[0, 1])
        crossing = float(-intercept / slope) if abs(slope) > 1e-9 else float("nan")
        print(f"  {name:>10}   corr {correlation:+.3f}   slope {slope:+.4f} per step   "
              f"zero at gap {crossing:+.2f}   n {int(good.sum())}   (threshold {tau_base:.2f})")
        report["rows"].append({"analysis": "bridge", "depth": name, "corr": correlation, "slope": float(slope),
                               "crossing": crossing, "n": int(good.sum())})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    print("ROUTE_FIELD_DECOMP_DONE")


if __name__ == "__main__":
    main()
