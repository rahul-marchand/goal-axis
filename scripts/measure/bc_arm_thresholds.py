"""What one family of the route model's arms does, without fitting an axis.

    uv run python scripts/measure/bc_arm_thresholds.py RUN --family k1 --base-value 0.5 \
        --demos /workspace/data/offline/demos/test.rho100

The offline twin of ``scripts/measure/arm_thresholds.py``, and it exists for the same
reason: ``027`` measures arms and fits an axis in one pass, and the held-out
families are one-sided handfuls that cannot identify a slope and an intercept.

Every arm is decoded on the *base* demonstrations, so an arm fine-tuned past
preference parity is still scored on the task the figure's axis is measured on.
Rows print in the shared table format, so the figure's parser reads a route
model's arms exactly as it reads an agent's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from goalmisgen.provenance import header
from goalmisgen.volume import iter_arms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Route-model run directory, holding checkpoints/ and arms/.")
    parser.add_argument("--family", type=str, required=True, help="Sweep tag, e.g. k1.")
    parser.add_argument("--base-value", type=float, required=True)
    parser.add_argument("--demos", type=Path, required=True, help="Held-out demonstrations at the base values.")
    parser.add_argument("--n", type=int, default=4096, help="Levels to decode per arm.")
    parser.add_argument("--steps", type=int, default=1000, help="Arm budget, to pick the right directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    from goalmisgen.analysis.behaviour import indifference_point
    from goalmisgen.analysis.metrics import bootstrap_episodes
    from goalmisgen.offline.decode import evaluate
    from goalmisgen.offline.demos import DemoSet
    from goalmisgen.offline.fast_decode import greedy_decode_cached
    from goalmisgen.offline.train import list_checkpoints, load_checkpoint

    demos = DemoSet.load(args.demos, hide_values=True)
    indices = np.arange(min(args.n, len(demos)))
    values = np.asarray(demos.values)[indices]
    distances = np.asarray(demos.distances)[indices].astype(int)
    feature_ids = np.asarray(demos.feature_ids)[indices].astype(int)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(indices))
    d_rich = distances[rows, richer]
    d_poor = distances[rows, 1 - richer]
    colour_of_rich = feature_ids[rows, richer]

    def threshold(checkpoint: Path, label: str) -> None:
        model, params = load_checkpoint(checkpoint)
        _, _, outcomes = evaluate(model, params, demos, indices, decoder=greedy_decode_cached, indifference=False)
        reached = np.array([bool(o.get("reached_objective")) for o in outcomes])
        reached_fid = np.array([-1 if o.get("reached_feature_id") is None else int(o["reached_feature_id"]) for o in outcomes])
        gaps = (d_rich - d_poor)[reached]
        took_richer = (reached_fid == colour_of_rich)[reached].astype(float)
        point = indifference_point(gaps, took_richer)
        low, high = bootstrap_episodes(
            lambda pick: indifference_point(gaps[pick], took_richer[pick]),
            np.arange(len(gaps)),
            resamples=200,
            seed=0,
        )
        print(f"  {label:>30}{point:>8.1f}  [{low:5.1f}, {high:5.1f}]{reached.mean():>11.1%}")

    arms = {}
    for parsed, directory in iter_arms(args.run / "arms", args.family, steps=args.steps):
        checkpoints = list_checkpoints(directory)
        if checkpoints:
            arms[round(args.base_value + parsed.offset, 10)] = checkpoints[-1][1]
    if not arms:
        sys.exit(f"no {args.family} arms at {args.steps} steps under {args.run / 'arms'}")

    print(f"run        {args.run}")
    print(f"family     {args.family}, {len(arms)} arms, decoded on {args.demos.name}\n")
    print("=== what the arms themselves do ===\n")
    print(f"  {'':>30}{'steps':>8}  {'95% interval':^14}{'reached':>11}")
    threshold(list_checkpoints(args.run)[-1][1], "base, untouched")
    for value, checkpoint in sorted(arms.items()):
        threshold(checkpoint, f"v={value:.2f} fine-tuned")


if __name__ == "__main__":
    main()
