"""Per-gap choice curves for axis-written models: the s-curve panel's data.

    uv run python scripts/written_scurves.py --agent /workspace/data/runs/novalue11.s1234 \
        --axis /workspace/data/figures-data/drc/basin/axis.novalue11.s1234.o1@400000.npz \
        --alphas -0.2 0 0.2 --out /workspace/data/figures-data/drc/scurves

The threshold tables say where the written threshold sits; this dumps the whole
psychometric curve around it. For each alpha (in units of the swept objective's
value) the base weights get ``theta + alpha * axis`` and are decoded over the
same episodes — one env, one seed, so the levels are paired across alphas and
the curves differ only by the write. Saves decode_h1_drc's five arrays per
alpha, plus a printed binned curve as a sanity check.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", type=Path, default=Path("/workspace/data/runs/novalue11.s1234"))
    parser.add_argument("--axis", type=Path, required=True, help="Cached axis npz from scripts/basin_map.py.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[-0.2, 0.0, 0.2])
    parser.add_argument("--levels", type=str, default="/workspace/data/levels/values/1.00-0.50@1M")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--episodes", type=int, default=8192)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    payload = json.loads((args.agent / "BASE.json").read_text())
    checkpoint = args.agent / payload["checkpoint"]
    values = tuple(payload["values"])
    axis = np.load(args.axis)["axis"].astype(np.float64)
    print(f"checkpoint {checkpoint}")
    print(f"axis       {args.axis.name}  |axis| {np.linalg.norm(axis):.1f} per unit value")

    import jax
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis import collect_episode_outcomes
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
    base_flat, unravel = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    envs = config.make()

    args.out.mkdir(parents=True, exist_ok=True)
    for alpha in args.alphas:
        params = unravel(np.asarray(base_flat + alpha * axis, dtype=np.float32))
        carry0 = policy.apply(
            params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry
        )
        state = {"carry": carry0, "key": jax.random.PRNGKey(args.seed)}

        def act(observations, episode_starts):
            state["carry"], action, _, state["key"] = get_action(
                params, state["carry"], observations, episode_starts, state["key"], temperature=0.0
            )
            return np.asarray(action)

        outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
        fids = sorted(int(k.split("_")[1]) for k in outcomes[0] if k.startswith("feature_") and k.endswith("_value"))
        v = np.array([[o[f"feature_{fid}_value"] for fid in fids] for o in outcomes])
        d = np.array([[o[f"feature_{fid}_distance"] for fid in fids] for o in outcomes], dtype=int)
        richer = np.argmax(v, axis=1)
        rows = np.arange(len(outcomes))
        reached = np.array([bool(o.get("reached_objective")) for o in outcomes])
        arrays = dict(
            d_rich=d[rows, richer],
            d_poor=d[rows, 1 - richer],
            colour_of_rich=np.array(fids)[richer],
            reached=reached,
            reached_fid=np.array([int(o["reached_feature_id"]) if r else -1 for o, r in zip(outcomes, reached)]),
        )
        tag = f"{'+' if alpha >= 0 else '-'}{abs(alpha):.2f}"
        out = args.out / f"scurve.{args.agent.name}.a{tag}.npz"
        np.savez_compressed(out, alpha=alpha, **arrays)

        gap = arrays["d_rich"] - arrays["d_poor"]
        took = (arrays["reached_fid"] == arrays["colour_of_rich"]) & arrays["reached"]
        line = []
        for g in range(-4, 21, 4):
            m = (gap >= g) & (gap < g + 4) & arrays["reached"]
            if m.sum() >= 50:
                line.append(f"gap {g:>3}..{g+3}: {took[m].mean():.2f}")
        print(f"alpha {alpha:+.2f}  reached {reached.mean():.1%}  " + "  ".join(line))
    print("\nSCURVES_DONE")


if __name__ == "__main__":
    main()
