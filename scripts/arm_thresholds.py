"""What one family of arms does, without fitting an axis through them.

    uv run python scripts/arm_thresholds.py RUN --family k1 --base-value 0.5 \
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
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.provenance import header  # noqa: E402
from goalmisgen.volume import discover_arms, sweep_family, sweep_index  # noqa: E402


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

    payload = json.loads((args.run / "BASE.json").read_text())
    checkpoint = args.run / payload["checkpoint"]
    values = tuple(payload["values"])
    arms = discover_arms(
        args.run / "arms",
        objective=sweep_index(args.family),
        base_value=args.base_value,
        steps=args.arm_steps,
        at=args.at,
        family=sweep_family(args.family),
    )
    if not arms:
        sys.exit(f"no {args.family} arms at {args.arm_steps} steps under {args.run / 'arms'}")
    print(f"base       {checkpoint}")
    print(f"family     {args.family}, {len(arms)} arms, decoded at the base values {values}\n")

    import jax
    from cleanba.cleanba_impala import load_train_state

    from goalmisgen.analysis import collect_episode_outcomes, summarise
    from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
    from goalmisgen.analysis.metrics import bootstrap_episodes
    from goalmisgen.configs.env import MazeConfig

    config = MazeConfig(
        max_episode_steps=120,
        num_envs=args.num_envs,
        min_size=11,
        max_size=11,
        n_objectives=len(values),
        objective_values=values,
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split=args.split,
        asynchronous=False,
        seed=args.seed,
    )
    policy, _, _, base_state, _ = load_train_state(checkpoint, env_cfg=config)
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    envs = config.make()

    def measure(params, label: str) -> None:
        carry = policy.apply(
            params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry
        )
        state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

        def act(observations, starts):
            state["carry"], action, _, state["key"] = get_action(
                params, state["carry"], observations, starts, state["key"], temperature=0.0
            )
            return np.asarray(action)

        outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
        gaps, took_richer, _ = value_distance_decisions(outcomes)
        point = indifference_point(gaps, took_richer)
        low, high = bootstrap_episodes(
            lambda rows: indifference_point(gaps[rows], took_richer[rows]),
            np.arange(len(gaps)),
            resamples=200,
            seed=args.seed,
        )
        reached = summarise(outcomes).reached_objective
        print(f"  {label:>30}{point:>8.1f}  [{low:5.1f}, {high:5.1f}]{reached:>11.1%}")

    print("=== what the arms themselves do ===\n")
    print(f"  {'':>30}{'steps':>8}  {'95% interval':^14}{'reached':>11}")
    measure(base_state.params, "base, untouched")
    for value, arm in sorted(arms.items()):
        _, _, _, state, _ = load_train_state(arm, env_cfg=config)
        measure(state.params, f"v={value:.2f} fine-tuned")


if __name__ == "__main__":
    main()
