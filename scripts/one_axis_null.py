"""Is it one axis? cos(axis_0, axis_1) against its noise ceiling and a permutation null.

    JAX_PLATFORMS=cpu uv run python scripts/one_axis_null.py \
        --rl novalue11g1.s1234=cp_149995520 resnet11nvg1.s1234=cp_120176640 \
        --bc bcnv11.s1 bcnv11.s2 bcnv11.s3 \
        --json figures/data/one_axis.json

Weight space only, no rollouts. For every base it fits one axis per swept
objective (``o0`` and ``o1``, common component absorbed by an intercept) and
reports three things the appendix table needs beside the raw cosine:

* the split-half reliability of each axis, Spearman-Brown corrected, and the
  ceiling ``sqrt(r0 * r1)`` two independent fits of one direction could reach;
* a permutation null: offsets shuffled among one sweep's diffs against the
  other sweep's axis, both ways round, which keeps whatever structure diffs of
  one network on one task share and destroys only the value signal;
* the cross-readback: each arm of one sweep, drift removed, projected onto the
  *other* sweep's axis and read as an offset. One knob on the difference in
  values predicts a slope of -1 against the offset the arm was trained at.

The RL loaders are ``014``'s and the route-model loaders are ``027``'s, so the
diffs here are the same objects those scripts fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from goalmisgen import provenance
from goalmisgen.analysis.weights import (
    cosine,
    fit_axis_and_drift,
    permutation_cosines,
    permutation_p_value,
    projected_offset,
    split_half_reliability,
)

RUNS = Path("/workspace/data/runs")
OFFLINE_RUNS = Path("/workspace/data/offline/runs")
LEVELS = "/workspace/data/levels/values/1.00-0.50@500k"
BASE_VALUES = {0: 1.0, 1: 0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rl", nargs="*", default=[], help="agent=checkpoint pairs, e.g. novalue11g1.s1234=cp_149995520")
    parser.add_argument("--bc", nargs="*", default=[], help="route-model run names under the offline runs directory")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--bc-steps", type=int, default=1000)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


# --- loading -----------------------------------------------------------------


def load_rl(agent: str, checkpoint: str, arm_steps: int) -> tuple[int, dict[int, dict[float, np.ndarray]]]:
    from jax.flatten_util import ravel_pytree
    from cleanba.cleanba_impala import load_train_state

    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.volume import discover_arms

    config = MazeConfig(
        max_episode_steps=120,
        num_envs=8,
        min_size=11,
        max_size=11,
        feature_value_correlation=1.0,
        objective_values=(1.0, 0.5),
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=LEVELS,
        dataset_split="test",
        asynchronous=False,
        seed=0,
    )
    root = RUNS / agent
    _, _, _, base_state, _ = load_train_state(root / "local-files" / checkpoint, env_cfg=config)
    base_flat, _ = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)
    sweeps: dict[int, dict[float, np.ndarray]] = {}
    for objective in (0, 1):
        selected = discover_arms(root / "arms", objective, BASE_VALUES[objective], steps=arm_steps, at=-1, family="o")
        diffs: dict[float, np.ndarray] = {}
        for value, path in sorted(selected.items()):
            _, _, _, state, _ = load_train_state(path, env_cfg=config)
            flat, _ = ravel_pytree(state.params)
            diffs[round(value - BASE_VALUES[objective], 10)] = np.asarray(flat, dtype=np.float64) - base_flat
        sweeps[objective] = diffs
    return int(base_flat.size), sweeps


def load_bc(run: str, steps: int) -> tuple[int, dict[int, dict[float, np.ndarray]]]:
    from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs

    base = load_base(OFFLINE_RUNS / run)
    sweeps = {}
    for objective in (0, 1):
        arms = arm_dirs(base.run_dir, f"o{objective}", steps)
        sweeps[objective] = {o: np.asarray(d, dtype=np.float64) for o, d in load_diffs(base, arms).items()}
    return int(base.flat.size), sweeps


# --- the statistics ----------------------------------------------------------


def fit(diffs: dict[float, np.ndarray], seed: int) -> dict:
    offsets = np.array(sorted(o for o in diffs if abs(o) > 1e-9))
    stacked = np.stack([diffs[o] for o in offsets])
    axis, drift = fit_axis_and_drift(offsets, stacked)
    null_arm = next((diffs[o] for o in diffs if abs(o) <= 1e-9), None)
    return dict(
        offsets=offsets,
        stacked=stacked,
        axis=axis,
        drift=drift,
        reliability=split_half_reliability(offsets, stacked, seed=seed),
        null_arm=null_arm,
    )


def analyse(name: str, parameters: int, sweeps: dict[int, dict[float, np.ndarray]], args: argparse.Namespace) -> dict:
    fits = {k: fit(v, args.seed) for k, v in sweeps.items()}
    a0, a1 = fits[0]["axis"], fits[1]["axis"]
    observed = cosine(a0, a1)
    r0, r1 = fits[0]["reliability"], fits[1]["reliability"]
    ceiling = float(np.sqrt(r0 * r1)) if r0 > 0 and r1 > 0 else float("nan")

    # Permutation null, each way round: shuffle sweep k's offsets, refit its
    # axis, cosine against the other sweep's (unshuffled) axis.
    nulls = {}
    for k, other in ((0, a1), (1, a0)):
        drawn = permutation_cosines(fits[k]["offsets"], fits[k]["stacked"], other, resamples=args.resamples, seed=args.seed)
        nulls[k] = dict(
            mean=float(drawn.mean()),
            sd=float(drawn.std()),
            p5=float(np.percentile(drawn, 5)),
            p=permutation_p_value(observed, drawn, alternative="less"),
        )

    # Cross-readback: sweep k's arms, drift removed, read through the other
    # sweep's axis as an offset in the other objective's units.
    readback = {}
    for k, other in ((0, a1), (1, a0)):
        f = fits[k]
        implied = np.array([projected_offset(f["stacked"][i] - f["drift"], other) for i in range(len(f["offsets"]))])
        slope = float(np.polyfit(f["offsets"], implied, 1)[0])
        own = np.array([projected_offset(f["stacked"][i] - f["drift"], f["axis"]) for i in range(len(f["offsets"]))])
        readback[k] = dict(
            trained=f["offsets"].tolist(),
            implied_other=implied.tolist(),
            slope_other=slope,
            slope_own=float(np.polyfit(f["offsets"], own, 1)[0]),
            corr_other=float(np.corrcoef(f["offsets"], implied)[0, 1]),
        )

    print(f"\n=== {name}  ({parameters:,} parameters) ===")
    for k in (0, 1):
        f = fits[k]
        print(
            f"  o{k}: {len(f['offsets'])} arms at {f['offsets'].min():+.2f}..{f['offsets'].max():+.2f}"
            + ("  + null arm" if f["null_arm"] is not None else "")
            + f"   |drift| {np.linalg.norm(f['drift']):.4g}  |axis| {np.linalg.norm(f['axis']):.4g}"
            f"  split-half reliability (SB) {f['reliability']:+.3f}"
        )
    print(f"  cos(axis_0, axis_1)          {observed:+.3f}")
    print(f"  ceiling sqrt(r0 r1)          {ceiling:+.3f}    cos/ceiling {observed / ceiling:+.2f}")
    for k in (0, 1):
        n = nulls[k]
        print(
            f"  null, shuffling o{k}:  mean {n['mean']:+.3f}  sd {n['sd']:.3f}  5th pct {n['p5']:+.3f}"
            f"  p(null <= observed) {n['p']:.4f}"
        )
    for k in (0, 1):
        r = readback[k]
        print(
            f"  o{k} arms read through axis_{1 - k}:  slope {r['slope_other']:+.2f} (one knob: -1)"
            f"   corr {r['corr_other']:+.3f}   own-axis slope {r['slope_own']:+.2f}"
        )

    return dict(
        parameters=parameters,
        cos=observed,
        reliability={f"o{k}": fits[k]["reliability"] for k in (0, 1)},
        ceiling=ceiling,
        cos_over_ceiling=observed / ceiling,
        arms={f"o{k}": len(fits[k]["offsets"]) for k in (0, 1)},
        leverage={f"o{k}": float(((fits[k]["offsets"] - fits[k]["offsets"].mean()) ** 2).sum()) for k in (0, 1)},
        drift_norm={f"o{k}": float(np.linalg.norm(fits[k]["drift"])) for k in (0, 1)},
        axis_norm={f"o{k}": float(np.linalg.norm(fits[k]["axis"])) for k in (0, 1)},
        null={f"shuffle_o{k}": nulls[k] for k in (0, 1)},
        readback={f"o{k}_through_axis_{1 - k}": readback[k] for k in (0, 1)},
    )


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")
    out = {}
    for spec in args.rl:
        agent, _, checkpoint = spec.partition("=")
        if not checkpoint:
            sys.exit(f"--rl wants agent=checkpoint, got {spec!r}")
        parameters, sweeps = load_rl(agent, checkpoint, args.arm_steps)
        out[agent] = dict(base=checkpoint, **analyse(agent, parameters, sweeps, args))
        sys.stdout.flush()
    for run in args.bc:
        parameters, sweeps = load_bc(run, args.bc_steps)
        out[run] = analyse(run, parameters, sweeps, args)
        sys.stdout.flush()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
