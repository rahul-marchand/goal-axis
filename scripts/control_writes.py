"""Build threshold-matched writes for knobs that are not a gain, and read what they do.

    uv run python scripts/control_writes.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k \
        --edits late=figures/data/gain_vs_factorised/e4/novalue11g1.s1234.late.npz:seed0 \
        --rotation figures/data/gain_vs_factorised/e7/novalue11g1.s1234.rotation.json \
        --pairs /workspace/data/figures-data/control/novalue11g1.s1234.npz

The positive control the study is missing: E5's factorised network never learned
to use value, so nothing has yet shown that the gain tests would say *no* to a
mechanism that is not a gain. This script prepares the knobs that supply it
(:mod:`goalmisgen.analysis.control_writes`) and reports the cheap behavioural
half of the comparison; ``scripts/gain_alignment.py`` and
``scripts/leftover_patch.py`` read the same pairs through ``--extra-writes``, so
all three tests see identical writes.

Each knob is scaled until its own antisymmetric threshold shift equals the
axis's, because gain alignment and the object-independent share both grow with
the size of a write and a knob at its own arbitrary dose would not be comparable.

The table at the end is one test in itself. Alongside the threshold it reports

* **reached / on route** -- the competence collateral, where the axis behaves
  like the input-conv edit and the late edit does not (E4);
* **tau above / below** -- the threshold measured separately on levels where the
  richer objective is above the agent and where it is below. A value knob moves
  both together; the contingent edit is trained to split them, so this is the
  statistic that should expose it.

Every default is pinned here rather than taken from another script, so an edit
elsewhere cannot change a run that is already going.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.control_writes import linear_pair, match_scale, rotation_pair, save_pairs  # noqa: E402
from goalmisgen.analysis.input_gain import input_kernel  # noqa: E402
from goalmisgen.analysis.weights import fit_axis_and_drift  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--offset", type=float, default=0.3, help="Dose the axis is written at; sets the target shift.")
    parser.add_argument("--edits", nargs="*", default=[], help="NAME=PATH.npz:KEY, a direction in per-step units.")
    parser.add_argument("--split-match", nargs="*", default=[],
                        help="Knobs scaled by their contingent split rather than their pooled shift: a contingent "
                             "edit moves the two thresholds opposite ways, so its pooled shift is ~0 at any dose.")
    parser.add_argument("--rotation", type=Path, default=None, help="A cheapest_edit rotation json; read turns_per_step.")
    parser.add_argument("--rotation-name", type=str, default="rotation")
    parser.add_argument("--n-eval", type=int, default=2048)
    parser.add_argument("--eval-chunk", type=int, default=1024)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=2, help="Matching rounds per knob.")
    parser.add_argument("--match-gains", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                        help="Colour-1 brightness grid the tau-matched brightness is chosen from.")
    parser.add_argument("--pairs", type=Path, default=None, help="npz of the matched writes, for the other two tests.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    import jax.numpy as jnp
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis import closed_loop
    from goalmisgen.analysis.closed_loop import read_outcomes
    from goalmisgen.analysis.route_targets import route_targets, usable
    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.edits.rotation import RotationEdit, colour_planes, partner_map
    from goalmisgen.envs.features import CorrelatedFeatures
    from scripts.restricted_writes import load_sweep

    payload = json.loads((args.run / "BASE.json").read_text())
    checkpoint = args.run / payload["checkpoint"]
    values = tuple(payload["values"])

    def config_for(split):
        return MazeConfig(
            max_episode_steps=args.step_limit, num_envs=1, min_size=11, max_size=11, n_objectives=len(values),
            objective_values=values, feature_value_correlation=1.0, value_encoding="none",
            colour_is_the_only_value_cue=True, level_dataset=args.levels, dataset_split=split, asynchronous=False,
        )

    config = config_for("test")
    policy, _, _, state, _ = load_train_state(checkpoint, env_cfg=config)
    flat32, unravel = ravel_pytree(state.params)
    base = np.asarray(flat32, dtype=np.float64)
    kernel = input_kernel(state.params)
    n_inputs = kernel.n_inputs
    print(f"base   {checkpoint}  ({base.size:,} parameters), values {values}")

    def levels_of(split, count):
        sampler = config_for(split).sampler()
        scheme = CorrelatedFeatures(1.0)
        out = []
        for index in sampler.indices:
            here = tuple(float(v) for v in sampler.dataset.values[index])
            level = sampler.dataset.level(int(index), scheme.assign(here, np.random.default_rng(int(index))))
            if usable(level):
                out.append(level)
            if len(out) == count:
                break
        return out

    offsets, diffs = load_sweep(args.run, args.sweep, values, args, config, base)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    print(f"sweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}")

    eval_targets = route_targets(levels_of("test", args.n_eval), size=11, n_channels=n_inputs)
    eval_static = jnp.asarray(eval_targets.static)
    eval_start = jnp.asarray(eval_targets.positions[:, 0, 0])
    plain = closed_loop.make_rollout_fn(policy, unravel, step_limit=args.step_limit)
    ones = np.ones(n_inputs, np.float32)

    def readout(flat, gains=None):
        reached, steps = [], []
        for start in range(0, len(eval_targets), args.eval_chunk):
            rows = slice(start, start + args.eval_chunk)
            r, s = plain(jnp.asarray(flat, dtype=jnp.float32), eval_static[rows], eval_start[rows],
                         jnp.asarray(ones if gains is None else gains, dtype=jnp.float32))
            reached.append(np.asarray(r))
            steps.append(np.asarray(s))
        return read_outcomes(eval_targets, np.concatenate(reached), np.concatenate(steps), resamples=50)

    started = time.time()
    base_row = readout(base)
    axis_plus, axis_minus = linear_pair(base, axis, args.offset)
    plus_row, minus_row = readout(axis_plus), readout(axis_minus)
    target = plus_row.tau - minus_row.tau
    print(f"\nbase threshold {base_row.tau:.2f} (reached {base_row.reached:.1%}, on route {base_row.on_route:.1%}; "
          f"{time.time() - started:.0f}s incl. compile)")
    print(f"axis  @ +-{args.offset}: {plus_row.tau:.2f} vs {minus_row.tau:.2f}  -> target shift {target:+.2f} steps")

    def shift_fn(plus, minus):
        return readout(plus).tau - readout(minus).tau

    def split_fn(plus, minus):
        """How far the two contingent thresholds are pulled apart, antisymmetrically.

        A contingent edit raises the threshold where the richer objective is above
        the agent and lowers it where it is below, so its pooled shift stays near
        zero however hard it is written and cannot fix its dose.
        """
        a, b = readout(plus), readout(minus)
        return ((a.tau_rich_above - a.tau_rich_below) - (b.tau_rich_above - b.tau_rich_below)) / 2

    base_split = base_row.tau_rich_above - base_row.tau_rich_below
    matched = []
    per_step = abs(target) / 2 if target else 1.0  # each side carries half the shift
    for spec in args.edits:
        name, _, source = spec.partition("=")
        path, _, key = source.partition(":")
        with np.load(path) as z:
            direction = np.asarray(z[key or "seed0"], dtype=np.float64)
        on_split = name in args.split_match
        # Saved edits are per step of threshold shift, so half the target is the first guess.
        m = match_scale(lambda s, d=direction: linear_pair(base, d, s), split_fn if on_split else shift_fn,
                        target=target, scale0=per_step, name=name, rounds=args.rounds)
        matched.append(m)
        print(f"match  {name}{' (on split)' if on_split else ''}: "
              + ", ".join(f"scale {s:.3g} -> {v:+.2f}" for s, v in m.history)
              + f"  (kept {m.scale:.3g}, {m.shift:+.2f}, error {m.error:.2f})", flush=True)

    if args.rotation:
        turns = np.asarray(json.loads(args.rotation.read_text())["runs"][0]["turns_per_step"], dtype=np.float64)
        family = RotationEdit(size=base.size, planes=colour_planes(kernel, COLOUR), partners=partner_map(COLOUR),
                              base_flat=base.astype(np.float32))
        m = match_scale(lambda s: rotation_pair(family, base, turns, s), shift_fn, target=target, scale0=per_step,
                        name=args.rotation_name, rounds=args.rounds)
        matched.append(m)
        print(f"match  {args.rotation_name}: " + ", ".join(f"scale {s:.3g} -> {v:+.2f}" for s, v in m.history)
              + f"  (kept {m.scale:.3g}, {m.shift:+.2f}, error {m.error:.2f})", flush=True)

    # The gain reference at the same threshold shift, chosen the way E1 chooses it.
    shifts = {}
    for g in args.match_gains:
        up, down = ones.copy(), ones.copy()
        up[COLOUR[1]], down[COLOUR[1]] = 1 + g, 1 - g
        shifts[g] = readout(base, up).tau - readout(base, down).tau
    g_match = min(shifts, key=lambda g: abs(abs(shifts[g]) - abs(target)))
    sign = 1.0 if np.sign(shifts[g_match]) == np.sign(target) else -1.0
    bright_up, bright_down = ones.copy(), ones.copy()
    bright_up[COLOUR[1]] = 1 + sign * g_match
    bright_down[COLOUR[1]] = 1 - sign * g_match
    print("match  brightness colour 1: " + ", ".join(f"{g:g}: {v:+.2f}" for g, v in shifts.items())
          + f"  -> {sign * g_match:+g}")

    rows = {
        "base": (base_row, None),
        "axis +": (plus_row, None),
        "axis -": (minus_row, None),
        "brightness +": (readout(base, bright_up), None),
        "brightness -": (readout(base, bright_down), None),
    }
    for m in matched:
        rows[f"{m.name} +"] = (readout(m.plus), None)
        rows[f"{m.name} -"] = (readout(m.minus), None)

    print("\n=== what each knob does, every knob at the axis's own threshold shift ===\n")
    print(f"  {'write':<16}{'tau':>8}{'reached':>10}{'on route':>10}{'took rich':>11}{'tau above':>11}{'tau below':>11}{'split':>8}")
    table = {}
    for name, (r, _) in rows.items():
        split = r.tau_rich_above - r.tau_rich_below
        table[name] = {**r.as_row(), "split": split}
        print(f"  {name:<16}{r.tau:>8.2f}{r.reached:>10.1%}{r.on_route:>10.1%}{r.took_rich:>11.1%}"
              f"{r.tau_rich_above:>11.2f}{r.tau_rich_below:>11.2f}{split:>8.2f}")
    print(f"\n  split = tau(rich above) - tau(rich below); the base's is {base_split:+.2f}. A value knob keeps it,")
    print("  a contingent edit pulls it apart, which is what makes it the cleanest distractor.")

    out = {
        "run": str(args.run), "checkpoint": str(checkpoint), "values": list(values), "sweep": args.sweep,
        "offset": args.offset, "target_shift": target, "axis_norm": float(np.linalg.norm(axis)),
        "matched_gain": sign * g_match, "brightness_shifts": {str(g): v for g, v in shifts.items()},
        "behaviour": table, "argv": sys.argv[1:], "base_split": base_split, "split_matched": list(args.split_match),
        "matching": {m.name: {"scale": m.scale, "shift": m.shift, "error": m.error, "history": m.history} for m in matched},
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float))
    if args.pairs:
        save_pairs(args.pairs, matched, meta={"run": str(args.run), "target_shift": target, "offset": args.offset})
        print(f"\nwrote {args.pairs} ({len(matched)} knobs) and {args.json or 'no json'}")


if __name__ == "__main__":
    main()
