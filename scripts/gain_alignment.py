"""Is the value axis a gain on the colours at every depth? Gain-alignment profile and mediation.

    uv run python scripts/gain_alignment.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --json figures/data/gain_alignment/novalue11g1.s1234.json

Two parts, both explained in :mod:`goalmisgen.analysis.gain_alignment`.

**Profile.** For each write (the full axis, each module's slice alone, the gain
part and what is left without it, and controls), at every site of the DRC --
input conv, conv 1, each cell at each tick, the dense layer, the logits -- the
fraction of the write's activation change explained by what turning up each
colour's brightness does there. Teacher-forced along both solver routes of
``--n-fit`` levels from the ``valid`` split.

**Mediation.** For the writes in ``--mediate``, closed-loop rollouts on
``--n-eval`` ``test`` levels, the base network with one site group set to the
base's value plus the write's change there: in full, only its gain component,
or only the residual,
with gain coefficients fitted in the profile (so on disjoint levels). Reported
as the fraction of the write's own antisymmetric threshold shift each patch
carries.

Controls: the axis's gain part written as brightness and into the weights
(small, so R^2 = 1 everywhere -- the plumbing check), a colour-1 brightness
change matched to the axis's threshold shift (the gain ceiling at the axis's own
dose, since a finite gain is not exactly its derivative), and random directions
with the axis's norm in every module.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.input_gain import gain_coefficients, gain_part, input_kernel  # noqa: E402
from goalmisgen.analysis.layers import module_groups, restrict  # noqa: E402
from goalmisgen.analysis.weights import fit_axis_and_drift  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)
MODES = {"full": (1.0, 0.0, None), "gain": (0.0, 1.0, "pc"), "residual": (1.0, -1.0, "pc"),
         "gain, global": (0.0, 1.0, "global"), "residual, global": (1.0, -1.0, "global")}
"""Mediation patches: (weight on the write's change, weight on its gain component, which coefficients)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--offset", type=float, default=0.3)
    parser.add_argument("--eps", type=float, default=0.05, help="Brightness step for the gain reference C.")
    parser.add_argument("--n-fit", type=int, default=1024, help="valid-split levels for the profile and coefficients")
    parser.add_argument("--n-eval", type=int, default=2048, help="test-split levels for the mediation rollouts")
    parser.add_argument("--chunk", type=int, default=32, help="levels per teacher-forced batch")
    parser.add_argument("--eval-chunk", type=int, default=1024)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--mediate", nargs="*", default=["full axis", "all but input conv", "brightness colour 1, tau-matched"])
    parser.add_argument("--match-gains", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-writes", type=Path, default=None,
                        help="npz of threshold-matched non-gain knobs from scripts/control_writes.py.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    import jax.numpy as jnp
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis import gain_alignment as ga
    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.edits import rl as rl_edit
    from goalmisgen.edits.rl import read_outcomes
    from goalmisgen.edits.targets import PAD, route_targets, usable
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
    groups, order = module_groups(state.params)
    sites = ga.sites_of(policy.cfg)
    print(f"base   {checkpoint}  ({base.size:,} parameters), values {values}, {n_inputs} input channels")

    def levels_of(split, count):
        sampler = config_for(split).sampler()
        scheme = CorrelatedFeatures(1.0)
        out = []
        for index in sampler.indices:
            values_here = tuple(float(v) for v in sampler.dataset.values[index])
            level = sampler.dataset.level(int(index), scheme.assign(values_here, np.random.default_rng(int(index))))
            if usable(level):
                out.append(level)
            if len(out) == count:
                break
        return out

    offsets, diffs = load_sweep(args.run, args.sweep, values, args, config, base)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    input_slice = restrict(axis, groups, "input conv")
    gain = gain_part(axis, base, kernel, COLOUR)
    betas = gain_coefficients(axis, base, kernel, COLOUR)
    print(f"sweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}; gain per unit offset "
          + ", ".join(f"colour {c - FIRST_FEATURE_CHANNEL}: {b:+.3f}" for c, b in betas.items()))

    rng = np.random.default_rng(args.seed)
    random_matched = np.zeros_like(axis)
    for name in order:
        idx = groups[name]
        r = rng.normal(size=len(idx))
        random_matched[idx] = r / np.linalg.norm(r) * np.linalg.norm(axis[idx])
    r = rng.normal(size=len(groups["input conv"]))
    random_input = np.zeros_like(axis)
    random_input[groups["input conv"]] = r / np.linalg.norm(r) * np.linalg.norm(input_slice)

    ones = np.ones(n_inputs)
    as_brightness = ones.copy()
    for channel, beta in betas.items():
        as_brightness[channel] = beta  # gain per unit offset; scaled below
    zero_gain = np.zeros(n_inputs)
    contrast = np.zeros(n_inputs)
    contrast[COLOUR[0]], contrast[COLOUR[1]] = 0.25 / args.offset, -0.25 / args.offset

    # name -> (parameter direction per unit offset, log-free gain slope per unit offset)
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "full axis": (axis, zero_gain),
        "gain part, in weights": (gain, zero_gain),
        "gain part, as brightness": (np.zeros_like(axis), np.where(np.isin(np.arange(n_inputs), list(betas)), as_brightness, 0.0)),
        "input conv minus gain": (input_slice - gain, zero_gain),
        "full minus gain": (axis - gain, zero_gain),
        "all but input conv": (axis - input_slice, zero_gain),
    }
    for name in order:
        variants[f"only {name}"] = (restrict(axis, groups, name), zero_gain)
    variants["random, module-matched"] = (random_matched, zero_gain)
    variants["random input conv, matched"] = (random_input, zero_gain)
    variants["brightness contrast +-25%"] = (np.zeros_like(axis), contrast)

    # A brightness change the size of the axis's own effect: the ceiling on every
    # gain-alignment number, since a finite gain is not exactly its derivative.
    eval_targets = route_targets(levels_of("test", args.n_eval), size=11, n_channels=n_inputs)
    plain = rl_edit.make_rollout_fn(policy, unravel, step_limit=args.step_limit)

    def plain_tau(flat, gains):
        reached, steps = [], []
        for start in range(0, len(eval_targets), args.eval_chunk):
            rows = slice(start, start + args.eval_chunk)
            r, s = plain(jnp.asarray(flat, dtype=jnp.float32), jnp.asarray(eval_targets.static[rows]),
                         jnp.asarray(eval_targets.positions[rows, 0, 0]), jnp.asarray(gains, dtype=jnp.float32))
            reached.append(np.asarray(r))
            steps.append(np.asarray(s))
        return read_outcomes(eval_targets, np.concatenate(reached), np.concatenate(steps), resamples=20).tau

    unit = np.ones(n_inputs)
    axis_shift = plain_tau(base + args.offset * axis, unit) - plain_tau(base - args.offset * axis, unit)
    shifts = {}
    for g in args.match_gains:
        up, down = unit.copy(), unit.copy()
        up[COLOUR[1]], down[COLOUR[1]] = 1 + g, 1 - g
        shifts[g] = plain_tau(base, up) - plain_tau(base, down)
    g_match = min(shifts, key=lambda g: abs(abs(shifts[g]) - abs(axis_shift)))
    sign = 1.0 if np.sign(shifts[g_match]) == np.sign(axis_shift) else -1.0
    matched = np.zeros(n_inputs)
    matched[COLOUR[1]] = sign * g_match / args.offset
    variants["brightness colour 1, tau-matched"] = (np.zeros_like(axis), matched)
    print(f"match  axis threshold shift over +-{args.offset}: {axis_shift:+.2f}; colour-1 brightness +-g: "
          + ", ".join(f"{g:g}: {v:+.2f}" for g, v in shifts.items()) + f"  -> {sign * g_match:+g}")

    # Knobs that are not a gain, already scaled to the axis's own threshold shift
    # (scripts/control_writes.py). They are stored as the two writes rather than as
    # a direction, because a rotation is not linear in its angle.
    extra_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if args.extra_writes:
        from goalmisgen.analysis.control_writes import load_pairs

        extra_pairs = load_pairs(args.extra_writes)
        for name in extra_pairs:
            variants[name] = (np.zeros_like(axis), zero_gain)  # placeholder; write_of uses the pair
        print(f"extra  {len(extra_pairs)} matched non-gain knobs from {args.extra_writes}: {', '.join(extra_pairs)}")
    names = list(variants)

    def write_of(name, sign):
        if name in extra_pairs:
            plus, minus = extra_pairs[name]
            return (plus if sign > 0 else minus).astype(np.float32), np.ones(n_inputs, dtype=np.float32)
        direction, slope = variants[name]
        return (base + sign * args.offset * direction).astype(np.float32), (1.0 + sign * args.offset * slope).astype(np.float32)

    out: dict = {
        "axis_shift": axis_shift, "brightness_shifts": {str(g): v for g, v in shifts.items()}, "matched_gain": sign * g_match,
        "run": str(args.run), "checkpoint": str(checkpoint), "values": list(values), "sweep": args.sweep,
        "offset": args.offset, "eps": args.eps, "betas": {str(c - FIRST_FEATURE_CHANNEL): b for c, b in betas.items()},
        "sites": [s.name for s in sites], "variants": names, "profile": {}, "mediation": [],
    }

    # --- profile ------------------------------------------------------------------
    fit_targets = route_targets(levels_of("valid", args.n_fit), size=11, n_channels=n_inputs)
    moments_fn = ga.make_moments_fn(policy, unravel, eps=args.eps)
    plus = [write_of(n, +1) for n in names]
    minus = [write_of(n, -1) for n in names]
    flats = jnp.asarray(np.stack([f for f, _ in plus] + [f for f, _ in minus]))
    gains = jnp.asarray(np.stack([g for _, g in plus] + [g for _, g in minus]))
    acc, started = None, time.time()
    n_fit = len(fit_targets)
    for start in range(0, n_fit, args.chunk):
        rows = np.arange(start, min(start + args.chunk, n_fit))
        sub = fit_targets.subset(rows)
        steps = sub.steps
        acc = ga.add_moments(acc, moments_fn(
            jnp.asarray(flat32), flats, gains,
            jnp.asarray(np.repeat(sub.static, 2, axis=0)),
            jnp.asarray(sub.positions.reshape(-1, steps, 2)),
            jnp.asarray(sub.role.reshape(-1, steps) != PAD),
        ))
        if start == 0:
            print(f"\nprofile: first chunk {time.time() - started:.0f}s (incl. compile)", flush=True)
    print(f"profile: {n_fit} levels in {time.time() - started:.0f}s")
    del flats

    fits = {name: [ga.fit_site(acc[k], v) for k in range(len(sites))] for v, name in enumerate(names)}
    for name in names:
        out["profile"][name] = [
            {"site": s.name, "r2_per_channel": f.r2_per_channel, "r2_global": f.r2_global, "change": f.change,
             "coef_global": f.coef_global.tolist()}
            for s, f in zip(sites, fits[name])
        ]
    for key, label in (("r2_per_channel", "R^2, two coefficients per channel"), ("r2_global", "R^2, two coefficients overall"),
                       ("change", "RMS change / RMS base activation")):
        print(f"\n=== gain alignment: {label}; rows = sites in forward order ===\n")
        width = 9
        print(f"  {'site':<18}" + "".join(f"{i:>{width}}" for i in range(len(names))))
        for k, s in enumerate(sites):
            print(f"  {s.name:<18}" + "".join(f"{getattr(fits[n][k], key):>{width}.3f}" for n in names))
        print("  columns: " + "; ".join(f"{i} {n}" for i, n in enumerate(names)))
    ceiling = names.index("brightness colour 1, tau-matched")
    print("\n=== R^2 per channel as a fraction of the tau-matched brightness's own (the gain ceiling at this dose) ===\n")
    print(f"  {'site':<18}" + "".join(f"{i:>9}" for i in range(len(names))))
    for k, s_ in enumerate(sites):
        top = fits[names[ceiling]][k].r2_per_channel
        print(f"  {s_.name:<18}" + "".join(f"{fits[n][k].r2_per_channel / top:>9.2f}" for n in names))

    # --- mediation ----------------------------------------------------------------
    if not args.mediate:
        finish(args, out)
        return
    rollout = ga.make_mediation_rollout(policy, unravel, step_limit=args.step_limit, eps=args.eps)
    group_names = ga.groups_of(sites)
    n_sites = len(sites)
    widths = [int(np.asarray(acc[k]["hh"]).shape[0]) for k in range(n_sites)]
    zero_coef = [jnp.zeros(w) for w in widths]

    def tau(main, write, write_gains, mix, full_w, gain_w, coef_a, coef_b):
        reached, steps = [], []
        for start in range(0, len(eval_targets), args.eval_chunk):
            rows = slice(start, start + args.eval_chunk)
            r, s = rollout(
                jnp.asarray(main), jnp.asarray(flat32), jnp.asarray(write), jnp.asarray(write_gains),
                jnp.asarray(eval_targets.static[rows]), jnp.asarray(eval_targets.positions[rows, 0, 0]),
                jnp.asarray(mix), jnp.asarray(full_w), jnp.asarray(gain_w), coef_a, coef_b,
            )
            reached.append(np.asarray(r))
            steps.append(np.asarray(s))
        readout = read_outcomes(eval_targets, np.concatenate(reached), np.concatenate(steps), resamples=50)
        return readout

    zeros = np.zeros(n_sites, dtype=np.float32)
    started = time.time()
    base_readout = tau(flat32, flat32, np.ones(n_inputs, np.float32), zeros, zeros, zeros, zero_coef, zero_coef)
    print(f"\nmediation on {len(eval_targets)} test levels; base threshold {base_readout.tau:.2f} "
          f"(reached {base_readout.reached:.1%}; {time.time() - started:.0f}s incl. compile)")
    out["base_tau"] = base_readout.tau
    for name in args.mediate:
        if name not in names:
            raise SystemExit(f"--mediate {name!r} is not one of the writes: {', '.join(names)}")
        per_sign = {}
        for sign in (+1, -1):
            write, write_gains = write_of(name, sign)
            row = {"write": name, "sign": sign}
            # A full patch at the logits *is* the write (tested), including a write made as brightness, which
            # the main network's own observation would not carry.
            at_logits = np.array([s.group == "logits" for s in sites], dtype=np.float32)
            row["write_tau"] = tau(flat32, write, write_gains, at_logits, at_logits, zeros, zero_coef, zero_coef).tau
            for group in group_names:
                in_group = np.array([s.group == group for s in sites], dtype=np.float32)
                for mode, (w_full, w_gain, which) in MODES.items():
                    if which == "pc":
                        coef_a = [jnp.asarray(sign * fits[name][k].coef_per_channel[0], dtype=jnp.float32) for k in range(n_sites)]
                        coef_b = [jnp.asarray(sign * fits[name][k].coef_per_channel[1], dtype=jnp.float32) for k in range(n_sites)]
                    elif which == "global":
                        coef_a = [jnp.full(widths[k], sign * fits[name][k].coef_global[0], dtype=jnp.float32) for k in range(n_sites)]
                        coef_b = [jnp.full(widths[k], sign * fits[name][k].coef_global[1], dtype=jnp.float32) for k in range(n_sites)]
                    else:
                        coef_a = coef_b = zero_coef
                    coef_a = [jnp.nan_to_num(c) for c in coef_a]
                    coef_b = [jnp.nan_to_num(c) for c in coef_b]
                    readout = tau(flat32, write, write_gains, in_group, in_group * w_full, in_group * w_gain, coef_a, coef_b)
                    row[f"{group}|{mode}"] = readout.tau
            per_sign[sign] = row
            out["mediation"].append(row)
            print(f"  {name} @ {sign * args.offset:+.2f}: write threshold {row['write_tau']:.2f} ({time.time() - started:.0f}s)", flush=True)
        plus_row, minus_row = per_sign[+1], per_sign[-1]
        print(f"\n=== mediation: {name}; fraction of its threshold shift ({plus_row['write_tau']:.2f} vs "
              f"{minus_row['write_tau']:.2f}) carried by a patch at each site group ===\n")
        print(f"  {'site group':<14}" + "".join(f"{mode:>18}" for mode in MODES))
        summary = {}
        for group in group_names:
            fractions = {mode: ga.fraction_carried(plus_row[f"{group}|{mode}"], minus_row[f"{group}|{mode}"],
                                                   plus_row["write_tau"], minus_row["write_tau"]) for mode in MODES}
            summary[group] = fractions
            print(f"  {group:<14}" + "".join(f"{fractions[mode]:>18.2f}" for mode in MODES))
        out.setdefault("fractions", {})[name] = summary
        finish(args, out)
    finish(args, out)


def finish(args: argparse.Namespace, out: dict) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
