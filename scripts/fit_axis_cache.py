"""Fit and cache a value axis npz for written_scurves.py, nothing else.

    uv run python /workspace/fit_axis_cache.py --agent /workspace/data/runs/resnet11nvg1.s1234 \
        --sweep o1 --arm-steps 400000 --out /workspace/data/figures-data/resnet

Reuses basin_map.fitted_axis so the fit and the cache format are identical to
the DRC's; skips the basin rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/GoalMisgeneralisation")
sys.path.insert(0, "/workspace/GoalMisgeneralisation/scripts")

from basin_map import fitted_axis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads((args.agent / "BASE.json").read_text())
    checkpoint = args.agent / payload["checkpoint"]
    values = tuple(payload["values"])
    print(f"checkpoint {checkpoint}")

    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.configs.env import MazeConfig

    config = MazeConfig(
        max_episode_steps=120,
        num_envs=64,
        min_size=11,
        max_size=11,
        n_objectives=len(values),
        objective_values=values,
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        asynchronous=False,
    )
    _, _, _, base_state, update = load_train_state(checkpoint, env_cfg=config)
    base_flat, _ = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)
    print(f"loaded base (update {update}, {base_flat.size:,} parameters)")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / f"axis.{args.agent.name}.{args.sweep}@{args.arm_steps}.npz"
    fitted_axis(args.agent, args.sweep, args.arm_steps, values, config, base_flat, cache)
    print(f"cached {cache}")


if __name__ == "__main__":
    main()
