"""Decode choices over the whole (d_rich, d_poor) plane for axis writes and threshold-matched brightness.

    uv run python scripts/separability_decode.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k \
        --out /workspace/data/figures-data/separability/novalue11g1.s1234

The data for :mod:`goalmisgen.analysis.separability` (the report is
``scripts/separability_report.py``). Per base:

1. **coarse brightness** -- colour 1 and colour 0 each scaled on their own over a
   grid of gains, a short decode each, to learn threshold against log gain;
2. **axis writes** -- the o-sweep axis written at a grid of offsets, full decode;
3. **matched brightness** -- for every write, the colour-1 gain and the colour-0
   gain whose threshold equals the write's, interpolated from (1), full decode;
4. **below zero** -- brightness taken past parity (threshold < 0), where no
   written axis reaches; full decode.

Every full decode uses the same seed and number of environments, so rows are the
same levels in the same order across conditions (checked and recorded, not
assumed). Each condition is one ``.npz`` plus a row in ``manifest.json``; an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.input_gain import scale_planes  # noqa: E402
from goalmisgen.analysis.weights import fit_axis_and_drift  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.7])
    parser.add_argument("--coarse-gains-1", type=float, nargs="+", default=[0.3, 0.45, 0.65, 0.8, 1.25, 1.6, 2.0, 3.0, 4.5])
    parser.add_argument("--coarse-gains-0", type=float, nargs="+", default=[0.15, 0.25, 0.4, 0.6, 0.8, 1.25, 1.6, 2.2])
    parser.add_argument("--below-zero", type=float, nargs="+", default=[-2.0, -5.0], help="Target thresholds past parity.")
    parser.add_argument("--episodes", type=int, default=16_384)
    parser.add_argument("--coarse-episodes", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-base", action="store_true", help="Base plus one axis write, for a throughput/pairing check.")
    return parser.parse_args()


def interpolate_gain(coarse: list[dict], target: float) -> float | None:
    """Log gain at which threshold equals ``target``, from a monotone-ish coarse curve; None if out of range."""
    points = sorted((np.log(r["gain"]), r["tau"]) for r in coarse if np.isfinite(r["tau"]))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            return float(np.exp(x0 + (target - y0) / (y1 - y0) * (x1 - x0)))
    return None


def main() -> None:
    args = parse_args()
    if args.episodes % args.num_envs or args.coarse_episodes % args.num_envs:
        sys.exit("episode counts must be multiples of --num-envs, or the per-slot level order breaks")
    print(header())
    print()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"conditions": {}}

    payload = json.loads((args.run / "BASE.json").read_text())
    checkpoint = args.run / payload["checkpoint"]
    values = tuple(payload["values"])

    import jax
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis import collect_episode_outcomes, summarise
    from goalmisgen.analysis.behaviour import indifference_point, on_shortest_route, value_distance_decisions
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
    base_flat, unravel = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)
    manifest.update(run=str(args.run), checkpoint=str(checkpoint), values=list(values), levels=args.levels,
                    split=args.split, seed=args.seed, num_envs=args.num_envs, episodes=args.episodes, sweep=args.sweep)
    print(f"base {checkpoint}, values {values}")
    envs = config.make()
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def decode(name: str, flat: np.ndarray, gains: dict[int, float] | None, episodes: int, **tags) -> dict:
        if name in manifest["conditions"] and (args.out / f"{name}.npz").exists():
            row = manifest["conditions"][name]
            print(f"  {name:>28}  (cached) tau {row['tau']:6.2f}", flush=True)
            return row
        params = unravel(np.asarray(flat, dtype=np.float32))
        carry = policy.apply(params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
        state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

        def act(observations, starts):
            if gains:
                observations = scale_planes(observations, gains, channel_axis=1)
            state["carry"], action, _, state["key"] = get_action(params, state["carry"], observations, starts, state["key"], temperature=0.0)
            return np.asarray(action)

        start = time.perf_counter()
        outcomes = collect_episode_outcomes(envs, act, episodes, seed=args.seed)
        seconds = time.perf_counter() - start
        fids = sorted(int(k.split("_")[1]) for k in outcomes[0] if k.startswith("feature_") and k.endswith("_value"))
        v = np.array([[o[f"feature_{f}_value"] for f in fids] for o in outcomes])
        d = np.array([[o[f"feature_{f}_distance"] for f in fids] for o in outcomes], dtype=int)
        richer = np.argmax(v, axis=1)
        rows = np.arange(len(outcomes))
        np.savez(
            args.out / f"{name}.npz",
            d_rich=d[rows, richer],
            d_poor=d[rows, 1 - richer],
            colour_of_rich=np.array(fids)[richer],
            reached=np.array([bool(o.get("reached_objective")) for o in outcomes]),
            reached_fid=np.array([o.get("reached_feature_id", -1) for o in outcomes], dtype=int),
            steps=np.array([int(o.get("episode_steps", -1)) for o in outcomes], dtype=int),
            value_tie=v[:, 0] == v[:, 1],
        )
        gaps, took, _ = value_distance_decisions(outcomes)
        tau = indifference_point(gaps, took)
        # A quick interval only; the report bootstraps properly. Full decodes are big enough not to need one here.
        low, high = (float("nan"), float("nan")) if episodes > args.coarse_episodes else bootstrap_episodes(
            lambda r: indifference_point(gaps[r], took[r]), np.arange(len(gaps)), resamples=20, seed=args.seed)
        row = {"name": name, "tau": float(tau), "low": float(low), "high": float(high), "episodes": episodes,
               "reached": float(summarise(outcomes).reached_objective), "on_route": float(on_shortest_route(outcomes)),
               "seconds": seconds, "gains": {str(k - FIRST_FEATURE_CHANNEL): g for k, g in (gains or {}).items()}, **tags}
        manifest["conditions"][name] = row
        save()
        print(f"  {name:>28}  tau {tau:6.2f} [{low:5.1f},{high:5.1f}]  reached {row['reached']:.1%}  route {row['on_route']:.1%}"
              f"  {episodes / seconds:,.0f} eps/s", flush=True)
        return row

    print("\n=== base ===")
    decode("base", base_flat, None, args.episodes, kind="base")

    from scripts.restricted_writes import load_sweep

    offsets, diffs = load_sweep(args.run, args.sweep, values, args, config, base_flat)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    manifest.update(axis_norm=float(np.linalg.norm(axis)), offsets_fitted=offsets.tolist())
    save()

    if args.only_base:
        decode("axis+0.30", base_flat + 0.3 * axis, None, args.episodes, kind="axis", offset=0.3)
        return

    print("\n=== coarse brightness ===")
    coarse = {0: [], 1: []}
    for plane, gains in ((1, args.coarse_gains_1), (0, args.coarse_gains_0)):
        for gain in gains:
            row = decode(f"coarse-c{plane}x{gain:g}", base_flat, {COLOUR[plane]: gain}, args.coarse_episodes,
                         kind="coarse", plane=plane, gain=gain)
            coarse[plane].append(row)
        base_tau = manifest["conditions"]["base"]["tau"]
        coarse[plane].append({"gain": 1.0, "tau": base_tau})

    print("\n=== axis writes ===")
    writes = []
    for offset in args.offsets:
        writes.append(decode(f"axis{offset:+.2f}", base_flat + offset * axis, None, args.episodes, kind="axis", offset=offset))

    print("\n=== brightness matched to each write's threshold ===")
    for write in writes:
        for plane in (1, 0):
            gain = interpolate_gain(coarse[plane], write["tau"])
            if gain is None:
                print(f"  axis{write['offset']:+.2f}: colour {plane} brightness never reaches tau {write['tau']:.2f}; skipped")
                continue
            decode(f"match{write['offset']:+.2f}-c{plane}", base_flat, {COLOUR[plane]: gain}, args.episodes,
                   kind="matched", plane=plane, gain=gain, matches=write["name"], target_tau=write["tau"])

    print("\n=== brightness past parity ===")
    for target in args.below_zero:
        for plane in (0, 1):
            gain = interpolate_gain(coarse[plane], target)
            if gain is None:
                print(f"  colour {plane} brightness never reaches tau {target:g} on the coarse grid; skipped")
                continue
            decode(f"below{target:+g}-c{plane}", base_flat, {COLOUR[plane]: gain}, args.episodes,
                   kind="below", plane=plane, gain=gain, target_tau=target)
    print("\nSEPARABILITY_DECODE_DONE")


if __name__ == "__main__":
    main()
