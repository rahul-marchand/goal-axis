"""Dump per-cell probe scores for the BC route model, mirroring plan_heatmap.py.

    uv run python scripts/measure/bc_plan_heatmap.py RUN_DIR \
        --demos /workspace/data/offline/demos/scaling/test.rho100 --out bc_plan_examples.npz

Same npz schema as ``scripts/measure/plan_heatmap.py`` so one drawing script serves
both: the probe is fit on the residual stream at the maze tokens before any
action token exists — the closest analogue of the DRC's state at t=0 — and the
labels are the model's own greedy route, replayed under env rules.

Levels come from the run's demo pool, not from a ``MazeConfig`` dataset, so the
examples cannot be the same levels as the RL models'. State that in a caption
rather than papering over it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.analysis import cell_dataset
from goalmisgen.analysis.probes import apply_logistic, fit_logistic
from goalmisgen.envs.observation import WALL_CHANNEL
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.probe import capture
from goalmisgen.offline.train import list_checkpoints, load_checkpoint, load_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--examples", type=int, default=12, help="Episodes to save. Draw the clearest few.")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Residual depth to read (0 = embedding). Default: every depth concatenated, as the DRC probe reads.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    step, directory = list_checkpoints(args.run)[-1]
    model, params = load_checkpoint(directory)
    hide_values = bool(load_run_config(args.run).get("demos", {}).get("hide_values", False))
    demos = DemoSet.load(args.demos, hide_values=hide_values)
    print(f"{args.run.name} @ step {step:,}; {args.demos.name} (hide_values={hide_values})")

    train_idx = np.arange(args.train_episodes)
    example_idx = np.arange(args.train_episodes, args.train_episodes + args.examples)
    train = capture(model, params, demos, train_idx, args.layer)
    examples = capture(model, params, demos, example_idx, args.layer)
    weights, mean, std = fit_logistic(*cell_dataset(train))

    scores, visited, visit_step, observations, distances = [], [], [], [], []
    for rollout in examples:
        free = rollout.observation[:, :, WALL_CHANNEL] < 0.5
        grid = np.full(free.shape, np.nan)
        grid[free] = apply_logistic(rollout.features[free].astype(np.float64), weights, mean, std)
        scores.append(grid)
        visited.append(rollout.visited)
        visit_step.append(rollout.visit_step)
        observations.append(rollout.observation)
        distances.append(rollout.distance)

    def info_column(key, default):
        return np.array([rollout.info.get(key, default) for rollout in examples])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        scores=np.stack(scores),
        visited=np.stack(visited),
        visit_step=np.stack(visit_step),
        observations=np.stack(observations),
        distances=np.stack(distances),
        step_penalty=demos.meta.get("step_penalty", 0.05) if isinstance(demos.meta, dict) else 0.05,
        optimal_feature_id=info_column("optimal_feature_id", -1),
        reached_feature_id=info_column("reached_feature_id", -1),
        chose_optimal=info_column("chose_optimal", False),
        checkpoint=str(directory),
    )
    print(f"wrote {args.out}  ({len(scores)} episodes)")


if __name__ == "__main__":
    main()
