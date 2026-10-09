"""Fit and cache a value axis npz for written_scurves.py, nothing else.

    uv run python scripts/fit_axis_cache.py --agent /workspace/data/runs/resnet11nvg1.s1234 \
        --sweep o1 --arm-steps 400000 --out /workspace/data/figures-data/resnet

The same fit and cache format as scripts/basin_map.py (goalmisgen.axis.cached_axis),
without the basin rollouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from goalmisgen.volume import read_base_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    print(f"checkpoint {read_base_marker(args.agent).checkpoint}")

    from goalmisgen.axis import cached_axis, load_base

    # Weights only: live levels and no seed, built to load and never stepped.
    base = load_base(args.agent, None, split="train", num_envs=64, seed=None)
    print(f"loaded base (update {base.update}, {base.raveled.size:,} parameters)")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / f"axis.{args.agent.name}.{args.sweep}@{args.arm_steps}.npz"
    cached_axis(base, args.sweep, args.arm_steps, cache)
    print(f"cached {cache}")


if __name__ == "__main__":
    main()
