"""Basin map of the value axis: theta + alpha*axis, rolled out from every free cell.

    uv run python scripts/basin_map.py --agent /workspace/data/runs/novalue11.s1234 \
        --sweep o1 --arm-steps 400000 --out /workspace/data/figures-data/drc/basin

For the figure-1 prototype: one fixed level, a grid of alphas along the fitted
weight-space value axis, and a deterministic rollout from every free starting
cell of each edited model. Each cell is coloured by which objective the agent
reaches from there, so the alpha sweep shows the decision boundary moving.

Three outputs, all under --out:

* ``axis.<agent>.<sweep>@<steps>.npz`` — the fitted axis, cached. No fitted DRC
  axis is saved anywhere else on the volume; every experiment refits it from
  the ~25 arm checkpoints, which costs minutes. The cache holds the axis, the
  drift, and the arm values it was fitted from.
* ``basin.<agent>.<sweep>.json`` — everything the figure needs: the level, the
  per-alpha per-start outcomes with full trajectories, the per-objective
  distance fields (for the utility-rule overlay), and a paired population
  dose-response with a competence line.
* ``level_candidates/`` — renders of the top-scoring levels from the scan, so
  the archetypal level is chosen by eye from a shortlist rather than trusted
  to a score.

Alpha is in units of the swept objective's reward value, exactly as in
``experiments/014_value_axis_analysis.py``: the axis is fitted per unit of
value and only ``alpha * axis`` is ever written — never the drift.

The population panel reuses one vector env and one seed for every alpha, so
its episodes are paired across the curve: every model faces the same levels.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from goalmisgen.envs.level import Level
from goalmisgen.paths import figures_data_dir, run_dir
from goalmisgen.provenance import header
from goalmisgen.volume import read_base_marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", type=Path, default=run_dir("novalue11.s1234"))
    parser.add_argument("--sweep", type=str, default="o1", help="Which fitted sweep the axis comes from.")
    parser.add_argument("--arm-steps", type=int, default=400_000, help="Arm budget; arms/ mixes @3M and @400k.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs=3,
        default=(-0.3, 1.3, 33),
        metavar=("LO", "HI", "N"),
        help="Alpha grid, linspace(LO, HI, N). Units of the swept objective's value.",
    )
    parser.add_argument(
        "--level-seed",
        type=int,
        default=None,
        help="Sample the level from this seed directly. Default: scan --scan seeds and take the best score.",
    )
    parser.add_argument("--scan", type=int, default=4000, help="Seeds to scan for an archetypal level.")
    parser.add_argument("--episodes", type=int, default=512, help="Population episodes per alpha.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=figures_data_dir() / "drc" / "basin")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Level choice
# ----------------------------------------------------------------------

# The behavioural indifference point of the base agent, in steps of extra
# distance to the richer objective (docs/Experiment2.md). Used only to score
# candidate levels for a balanced starting split; nothing downstream trusts it.
BASE_THRESHOLD = 7.7


def distance_fields(level) -> list[np.ndarray]:
    """Per-objective distance from every cell, routing around the other objective."""
    from goalmisgen.envs.solver import distance_field, walls_blocking_other_objectives

    return [
        distance_field(walls_blocking_other_objectives(level, i), objective.position)
        for i, objective in enumerate(level.objectives)
    ]


def usable_starts(level, fields, step_limit: int) -> list[tuple[int, int]]:
    """Free cells from which at least one objective is reachable in time."""
    from goalmisgen.envs.solver import UNREACHABLE

    objective_cells = {o.position for o in level.objectives}
    starts = []
    for position in level.free_positions():
        if position in objective_cells:
            continue
        if any(f[position] != UNREACHABLE and f[position] <= step_limit for f in fields):
            starts.append(position)
    return starts


def score_level(level, step_limit: int) -> tuple[float, dict]:
    """Higher is better: most cells inside the band the dial can actually sweep.

    The axis moves the behavioural threshold over roughly 0..16 extra steps
    before it saturates at preference parity (the writes refuse to reorder the
    objectives), so a start's colour can only flip if its distance gap lies in
    that band. A level chosen for *wide* gap coverage puts nearly every cell
    outside it and the boundary barely moves on screen; what the figure wants
    is most cells sweepable, with small stable anchors on both sides.
    """
    from goalmisgen.envs.solver import UNREACHABLE

    fields = distance_fields(level)
    starts = usable_starts(level, fields, step_limit)
    if len(starts) < 30:
        return -1e9, {}
    rich = int(np.argmax([o.value for o in level.objectives]))
    reachable = [s for s in starts if fields[rich][s] != UNREACHABLE and fields[1 - rich][s] != UNREACHABLE]
    if len(reachable) < len(starts):  # dead cells make the map lie
        return -1e9, {}
    gaps = np.array([int(fields[rich][s]) - int(fields[1 - rich][s]) for s in starts])
    sweepable = float(((gaps >= 1) & (gaps <= 15)).mean())
    anchor_rich = float((gaps <= -1).mean())
    anchor_poor = float((gaps >= 18).mean())
    distinct = len(set(gaps[(gaps >= 1) & (gaps <= 15)])) / 15.0
    score = 2.0 * sweepable + 0.3 * min(anchor_rich, 0.15) / 0.15 + 0.3 * min(anchor_poor, 0.15) / 0.15 + 0.5 * distinct
    return score, {
        "gap_min": int(gaps.min()),
        "gap_max": int(gaps.max()),
        "sweepable": round(sweepable, 2),
        "anchors": (round(anchor_rich, 2), round(anchor_poor, 2)),
        "distinct": round(distinct, 2),
        "n_starts": len(starts),
    }


def pick_level(values: tuple[float, ...], scan: int, level_seed: int | None, step_limit: int, out: Path):
    from goalmisgen.envs.features import CorrelatedFeatures
    from goalmisgen.envs.rendering import render
    from goalmisgen.envs.sampling import MazeLevelSampler
    from goalmisgen.envs.values import FixedValues

    sampler = MazeLevelSampler(
        size_range=(11, 11),
        n_objectives=len(values),
        values=FixedValues(values),
        features=CorrelatedFeatures(1.0),
    )

    def sample(seed: int):
        return sampler.sample(np.random.default_rng(seed))

    if level_seed is not None:
        level = sample(level_seed)
        print(f"level: seed {level_seed} (given)  score {score_level(level, step_limit)[1]}")
        return level, level_seed

    scored = []
    for seed in range(scan):
        score, detail = score_level(sample(seed), step_limit)
        if score > -1e9:
            scored.append((score, seed, detail))
    scored.sort(reverse=True)
    gallery = out / "level_candidates"
    gallery.mkdir(parents=True, exist_ok=True)
    print(f"level scan: {len(scored)} usable of {scan} seeds; top five:")
    for score, seed, detail in scored[:5]:
        level = sample(seed)
        image = render(level, level.agent_start, cell_pixels=16)
        try:
            import imageio.v3 as imageio

            imageio.imwrite(gallery / f"seed{seed:05d}.png", image)
        except ImportError:
            np.save(gallery / f"seed{seed:05d}.npy", image)
        print(f"  seed {seed:>5}  score {score:.3f}  {detail}")
    best = scored[0][1]
    return sample(best), best


# ----------------------------------------------------------------------
# The axis
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Rollouts
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FixedLevelSampler:
    level: "Level"

    def sample(self, rng: np.random.Generator) -> "Level":
        del rng
        return self.level


def rollout_from_everywhere(level, starts, params, policy, get_action, config, seed: int):
    """One deterministic episode per start, all starts stepped in lockstep.

    Vector-env autoreset is the wrong tool here — each slot must hold its own
    fixed start and stop after one episode — so the envs are plain ``MazeEnv``s
    stepped by hand, sharing one batched policy call. Finished slots keep
    feeding their last observation; their carries drift but nothing reads them.
    """
    import jax

    from goalmisgen.envs.maze import MazeEnv

    step_limit = config.max_episode_steps
    envs, observations = [], []
    for start in starts:
        variant = dataclasses.replace(level, agent_start=start)
        env = MazeEnv(
            sampler=FixedLevelSampler(variant),
            encoder=config.encoder(),
            step_penalty=config.step_penalty,
            step_limit=step_limit,
        )
        obs, _ = env.reset(seed=seed)
        envs.append(env)
        observations.append(obs.transpose(2, 0, 1))

    n = len(envs)
    obs = np.stack(observations)
    carry = policy.apply(params, jax.random.PRNGKey(seed), obs.shape, method=policy.initialize_carry)
    key = jax.random.PRNGKey(seed)
    episode_starts = np.ones(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    outcome = np.full(n, -1, dtype=int)
    steps = np.zeros(n, dtype=int)
    paths = [[tuple(map(int, start))] for start in starts]

    for _ in range(step_limit):
        carry, action, _, key = get_action(params, carry, obs, episode_starts, key, temperature=0.0)
        episode_starts = np.zeros(n, dtype=bool)
        action = np.asarray(action)
        for i, env in enumerate(envs):
            if done[i]:
                continue
            observation, _, terminated, truncated, info = env.step(int(action[i]))
            paths[i].append(tuple(map(int, env.agent_position)))
            obs[i] = observation.transpose(2, 0, 1)
            if terminated or truncated:
                done[i] = True
                steps[i] = info["episode_steps"]
                if terminated:
                    outcome[i] = int(info["reached_feature_id"])
        if done.all():
            break
    return outcome, steps, paths


def population_stats(outcomes) -> dict:
    fids = sorted(int(k.split("_")[1]) for k in outcomes[0] if k.startswith("feature_") and k.endswith("_value"))
    values = np.array([[o[f"feature_{fid}_value"] for fid in fids] for o in outcomes])
    distances = np.array([[o[f"feature_{fid}_distance"] for fid in fids] for o in outcomes], dtype=int)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(outcomes))
    reached = np.array([bool(o.get("reached_objective")) for o in outcomes])
    reached_fid = np.array([int(o["reached_feature_id"]) if r else -1 for o, r in zip(outcomes, reached)])
    steps = np.array([int(o["episode_steps"]) for o in outcomes])
    colour_of_rich = np.array(fids)[richer]
    took_rich = reached & (reached_fid == colour_of_rich)
    # Competence: extra steps beyond the shortest path to whichever objective
    # the agent itself chose. Near zero at every alpha = the edit moves the
    # preference, not the ability to execute it.
    fid_column = {fid: i for i, fid in enumerate(fids)}
    chosen_distance = np.array(
        [distances[row, fid_column[fid]] if fid >= 0 else -1 for row, fid in zip(rows, reached_fid)]
    )
    valid = reached & (chosen_distance >= 0)
    return {
        "n": len(outcomes),
        "reached": float(reached.mean()),
        "took_rich": float(took_rich[reached].mean()) if reached.any() else float("nan"),
        "excess_steps": float((steps[valid] - chosen_distance[valid]).mean()) if valid.any() else float("nan"),
    }


def main() -> None:
    args = parse_args()
    print(header())
    print()

    marker = read_base_marker(args.agent)
    checkpoint, values = marker.checkpoint, marker.values
    print(f"checkpoint {checkpoint}")
    print(f"values     {values}")

    from goalmisgen.axis import Greedy, cached_axis, load_base
    from goalmisgen.envs.solver import UNREACHABLE

    # Live levels (no dataset): the population statistics sample their own.
    base = load_base(args.agent, None, split="train", num_envs=args.num_envs, seed=args.seed)
    config, policy, unravel = base.config, base.policy, base.unravel
    base_flat = base.flat
    print(f"loaded base (update {base.update}, {base_flat.size:,} parameters)")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / f"axis.{args.agent.name}.{args.sweep}@{args.arm_steps}.npz"
    axis = cached_axis(base, args.sweep, args.arm_steps, cache)

    level, level_seed = pick_level(values, args.scan, args.level_seed, config.max_episode_steps, args.out)
    fields = distance_fields(level)
    starts = usable_starts(level, fields, config.max_episode_steps)
    rich = int(np.argmax([o.value for o in level.objectives]))
    print(f"level seed {level_seed}: {len(starts)} starts, rich objective {rich} "
          f"(colour {level.objectives[rich].feature_id})")

    greedy = Greedy.of(base)
    get_action = greedy.get_action

    low, high, count = args.alphas
    alphas = np.linspace(low, high, int(count))
    per_alpha = []
    for alpha in alphas:
        tick = time.perf_counter()
        params = unravel(np.asarray(base_flat + alpha * axis, dtype=np.float32))
        outcome, steps, paths = rollout_from_everywhere(level, starts, params, policy, get_action, config, args.seed)

        stats = population_stats(greedy.outcomes(params, args.episodes, args.seed))
        per_alpha.append(
            {
                "alpha": round(float(alpha), 4),
                "outcome": outcome.tolist(),
                "steps": steps.tolist(),
                "paths": [[list(p) for p in path] for path in paths],
                "population": stats,
            }
        )
        reached_any = (outcome >= 0).mean()
        took_rich_here = (outcome == level.objectives[rich].feature_id).mean()
        print(
            f"alpha {alpha:+.2f}  basin rich {took_rich_here:.0%}  reached {reached_any:.0%}  "
            f"population rich {stats['took_rich']:.0%}  excess {stats['excess_steps']:.2f}  "
            f"({time.perf_counter() - tick:.0f}s)"
        )

    payload = {
        "agent": args.agent.name,
        "checkpoint": str(checkpoint),
        "sweep": args.sweep,
        "arm_steps": args.arm_steps,
        "base_values": list(values),
        "step_penalty": config.step_penalty,
        "level_seed": level_seed,
        "walls": level.walls.astype(int).tolist(),
        "objectives": [
            {"position": list(o.position), "value": o.value, "feature_id": o.feature_id} for o in level.objectives
        ],
        "distance_fields": [np.where(f == UNREACHABLE, -1, f).astype(int).tolist() for f in fields],
        "starts": [list(s) for s in starts],
        "alphas": alphas.round(4).tolist(),
        "runs": per_alpha,
    }
    out = args.out / f"basin.{args.agent.name}.{args.sweep}.seed{level_seed}.json"
    out.write_text(json.dumps(payload))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
