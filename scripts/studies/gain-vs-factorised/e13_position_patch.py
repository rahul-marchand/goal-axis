"""Where in the sequence does the route model's value axis do its work?

Every BC intervention so far has been DRC-shaped: edit a module's weights and
see what happens. A transformer offers a handle a convolution does not -- token
position. The fitted axis `v` is a weight direction, but writing it changes the
residual stream by a definite amount at every token, and that change can be
replayed into the UNEDITED model one subset of tokens at a time:

* **objectives** -- the two cells the objectives sit on. Value written into the
  objective's own representation, before any comparison.
* **sep** -- the decision token alone. Value applied at the comparison, which is
  the factorised story.
* **other cells** -- the rest of the maze.
* **all** -- the control. It must reproduce the weight write's own threshold
  shift; if it does not, the decomposition below means nothing and the run says
  so rather than reporting subsets.

Three earlier steering attempts on this model were null because the *linearly
decodable* gap direction at SEP is not load-bearing. This patches the axis's own
activation difference instead of a probe direction, so it is not the same
experiment -- but that is exactly why the `all` control is mandatory.

The patch is applied at `--depth` (0 the embedding, k after block k). An edit at
the maze tokens after the last block cannot reach the decoder, so depths at or
near the end are expected to be inert for cell subsets and are skipped by
default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test.rho100")
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--steps-tag", type=int, default=1000)
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument("--depths", type=int, nargs="+", default=None, help="Default: 0 and every block but the last.")
    parser.add_argument("--doses", type=float, nargs="+", default=[-0.3, 0.3], help="Value offsets to write.")
    parser.add_argument(
        "--chunk", type=int, default=256, help="Levels per forward pass when reading residuals; d512l4 OOMs at 2048."
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())

    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
    from goalmisgen.offline.decode import evaluate
    from goalmisgen.offline.demos import DemoSet
    from goalmisgen.offline.gap_probe import _prefix_residuals_fn
    from goalmisgen.offline.rewrite import objective_cells

    base = load_base(args.run)
    cfg = base.model.config
    print(f"base   {args.run.name} @ step {base.step:,} ({base.flat.size:,} parameters), values {base.values}")

    arms = arm_dirs(args.run, args.sweep, args.steps_tag)
    if not arms:
        raise SystemExit(f"no finished {args.sweep} arms at {args.steps_tag} steps under {args.run}/arms")
    diffs = load_diffs(base, arms)
    offsets = np.array(sorted(diffs), dtype=np.float64)
    stack = np.stack([diffs[o] for o in offsets])
    axis, _ = fit_axis_and_drift(offsets, stack)
    print(f"axis   {args.sweep}: {len(offsets)} arms, |v| {np.linalg.norm(axis):.3f} per unit value")

    demos = DemoSet.load(args.demos / args.split, hide_values=base.hide_values)
    indices = np.arange(min(args.levels, len(demos)))
    observations = demos.observations(indices)
    n_cells = cfg.size * cfg.size
    prefix = cfg.prefix_length
    print(f"levels {len(indices)} from {args.split}; {n_cells} cell tokens + SEP = {prefix}\n")

    residuals_fn = _prefix_residuals_fn(cfg)
    depths = args.depths if args.depths is not None else [0, *range(1, cfg.n_layers)]

    # Token masks, (levels, prefix, 1) so they multiply the residual difference.
    objective_mask = np.zeros((len(indices), prefix), dtype=bool)
    for row, index in enumerate(indices):
        flat = objective_cells(observations[row]).reshape(-1)
        objective_mask[row, :n_cells] = flat
    sep_mask = np.zeros_like(objective_mask)
    sep_mask[:, -1] = True
    other_mask = np.zeros_like(objective_mask)
    other_mask[:, :n_cells] = ~objective_mask[:, :n_cells]
    all_mask = np.ones_like(objective_mask)
    masks = {"objectives": objective_mask, "sep": sep_mask, "other cells": other_mask, "all": all_mask}

    def threshold(params, edit=None, depth=0) -> tuple[float, float, float]:
        summary, _, _ = evaluate(base.model, params, demos, indices, edit=edit, edit_depth=depth)
        behaviour = summary.behaviour
        return float(summary.indifference), float(behaviour.reached_objective), float(behaviour.chose_optimal)

    out: dict = {
        "run": str(args.run),
        "sweep": args.sweep,
        "levels": int(len(indices)),
        "axis_norm": float(np.linalg.norm(axis)),
        "rows": [],
    }

    tau_base, reached_base, route_base = threshold(base.params)
    print(f"  {'condition':>26}{'depth':>7}{'tau':>9}{'reached':>9}{'route':>9}{'share':>8}")
    print(f"  {'base':>26}{'-':>7}{tau_base:9.2f}{reached_base:9.1%}{route_base:9.1%}{'-':>8}")
    out["rows"].append({"condition": "base", "tau": tau_base, "reached": reached_base, "on_route": route_base})

    for dose in args.doses:
        written = base.unravel(np.asarray(base.flat + dose * axis, dtype=np.float32))
        tau_w, reached_w, route_w = threshold(written)
        shift = tau_w - tau_base
        print(f"  {f'weight write @ {dose:+.2f}':>26}{'-':>7}{tau_w:9.2f}{reached_w:9.1%}{route_w:9.1%}{1.0:8.2f}")
        out["rows"].append(
            {
                "condition": "weight write",
                "dose": dose,
                "tau": tau_w,
                "reached": reached_w,
                "on_route": route_w,
                "shift": shift,
                "share": 1.0,
            }
        )
        for depth in depths:
            parts = []
            for start in range(0, len(observations), args.chunk):
                window = observations[start : start + args.chunk]
                streams_base = np.asarray(residuals_fn(base.params, window))[depth]
                streams_written = np.asarray(residuals_fn(written, window))[depth]
                parts.append(streams_written - streams_base)
            delta = np.concatenate(parts, axis=0)
            for name, mask in masks.items():
                edit = delta * mask[..., None]
                tau, reached, route = threshold(base.params, edit=edit.astype(np.float32), depth=depth)
                share = (tau - tau_base) / shift if abs(shift) > 1e-6 else float("nan")
                print(f"  {f'{name} @ {dose:+.2f}':>26}{depth:7d}{tau:9.2f}{reached:9.1%}{route:9.1%}{share:8.2f}", flush=True)
                out["rows"].append(
                    {
                        "condition": name,
                        "dose": dose,
                        "depth": int(depth),
                        "tau": tau,
                        "reached": reached,
                        "on_route": route,
                        "shift": tau - tau_base,
                        "share": share,
                    }
                )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
