"""What one family of arms does, without fitting an axis through them.

    uv run python scripts/measure/arm_thresholds.py RUN --family k1 --base-value 0.5 \
        --levels /workspace/data/levels/values/1.00-0.50@500k

``014`` measures arms and fits an axis in the same pass, which is right for the
symmetric grid the axis comes from and wrong for the held-out families. Those
exist precisely because they must stay out of the fit, and a one-sided handful
of them cannot identify an intercept and a slope at all: with three arms the
leave-one-out step is left with two and the run dies before printing a single
threshold.

So this measures and stops. Every arm is decoded at the *base* values, like
everything else in the project, so an arm trained past preference parity is
still scored on the task the figure's axis is measured on. Rows are printed in
``014``'s own table format, so the figure's parser reads them without knowing
which script produced them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from goalmisgen.provenance import header
from goalmisgen.volume import read_base_marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Agent directory holding BASE.json and arms/.")
    parser.add_argument("--family", type=str, required=True, help="Sweep tag, e.g. k1 or x0.")
    parser.add_argument("--base-value", type=float, required=True, help="What the swept objective pays at the base.")
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--at", type=int, default=-1, help="Which checkpoint of each arm, by index.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    from goalmisgen.axis import Greedy, arm_checkpoints, load_base

    marker = read_base_marker(args.run)
    checkpoint, values = marker.checkpoint, marker.values
    arms = arm_checkpoints(args.run / "arms", args.family, args.base_value, steps=args.arm_steps, at=args.at)
    if not arms:
        sys.exit(f"no {args.family} arms at {args.arm_steps} steps under {args.run / 'arms'}")
    print(f"base       {checkpoint}")
    print(f"family     {args.family}, {len(arms)} arms, decoded at the base values {values}\n")

    base = load_base(args.run, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed)
    greedy = Greedy.of(base)

    def measure(params, label: str) -> None:
        print(greedy.threshold(params, args.episodes, args.seed).line(label))

    print("=== what the arms themselves do ===\n")
    print(f"  {'':>30}{'steps':>8}  {'95% interval':^14}{'reached':>11}")
    measure(base.params, "base, untouched")
    for value, arm in sorted(arms.items()):
        measure(base.other_params(arm), f"v={value:.2f} fine-tuned")


if __name__ == "__main__":
    main()
