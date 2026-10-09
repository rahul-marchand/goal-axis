"""Three loose ends in the DRC gain story: the leftover, the lever's ceiling, and the axis's own mix.

    uv run python scripts/leftover_patch.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k \
        --json figures/data/leftover/novalue11g1.s1234.json

**leftover.** With the colour-1 objective erased a gain does nothing at all, yet
the fitted axis still moves activations. Per site, the size of that
object-independent part (``nn / ff``), and then the part patched into the base on
ordinary levels, alone and taken away, as a fraction of the write's own threshold
shift. A random module-matched direction is the noise reference and the axis's
own gain part is the plumbing check -- its object-independent share must be zero.

**signal.** ``S_c(g)``, how much of the activation is there because objective ``c``
is, measured by erasing it, against the threshold on the same levels and the same
gain grid. If brightening saturates while dimming does not, ``S_1`` flattens where
the threshold does and ``S_0`` falls without limit.

**mix.** The axis's gain slice is about 3.6:1 towards brightening colour 1. Two
composites keep the axis but swap that slice for a pure dim of colour 0 or a pure
brighten of colour 1, matched at first order, to see how much of the axis's floor
is its mix rather than the damage its other parts do.

Every default is pinned here rather than read from another script, so edits
elsewhere cannot change a run mid-flight.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--offset", type=float, default=0.3)
    parser.add_argument("--parts", nargs="+", default=["leftover", "signal", "mix"])
    parser.add_argument("--mediate", nargs="*", default=["full axis", "random, module-matched"])
    parser.add_argument("--groups", nargs="*", default=["cell 0", "cell 1", "cell 2", "all"])
    parser.add_argument("--n-fit", type=int, default=768, help="valid-split levels for the teacher-forced sizes")
    parser.add_argument("--n-eval", type=int, default=2048, help="test-split levels for rollouts")
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--eval-chunk", type=int, default=1024)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--signal-chunk", type=int, default=16)
    parser.add_argument("--n-signal", type=int, default=512)
    parser.add_argument("--bright-1", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0])
    parser.add_argument("--dim-0", type=float, nargs="+", default=[0.8, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12])
    parser.add_argument("--mix-offsets", type=float, nargs="+", default=[0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
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
    from goalmisgen.analysis import leftover as lo
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
    gain = gain_part(axis, base, kernel, COLOUR)
    betas = gain_coefficients(axis, base, kernel, COLOUR)
    beta0, beta1 = betas[COLOUR[0]], betas[COLOUR[1]]
    print(f"sweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}; "
          f"gain per unit offset colour 0 {beta0:+.3f}, colour 1 {beta1:+.3f}")

    rng = np.random.default_rng(args.seed)
    random_matched = np.zeros_like(axis)
    for name in order:
        idx = groups[name]
        r = rng.normal(size=len(idx))
        random_matched[idx] = r / np.linalg.norm(r) * np.linalg.norm(axis[idx])

    def plane_gain(channel: int, beta: float) -> np.ndarray:
        """A full-length direction scaling ``channel``'s input kernel by ``1 + offset * beta``."""
        out = np.zeros_like(axis)
        rows = kernel.plane(channel)
        out[rows] = beta * base[rows]
        return out

    variants: dict[str, np.ndarray] = {
        "full axis": axis,
        "gain part, in weights": gain,
        "random, module-matched": random_matched,
        "all but input conv": axis - restrict(axis, groups, "input conv"),
    }

    # Knobs that are not a gain, already scaled to the axis's own threshold shift
    # (scripts/control_writes.py), stored as the two writes because a rotation is
    # not linear in its angle. ``write_of`` below is the only place they differ.
    extra_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if args.extra_writes:
        from goalmisgen.analysis.control_writes import load_pairs

        extra_pairs = load_pairs(args.extra_writes)
        for name in extra_pairs:
            variants[name] = np.zeros_like(axis)  # placeholder; write_of uses the pair
        print(f"extra  {len(extra_pairs)} matched non-gain knobs from {args.extra_writes}: {', '.join(extra_pairs)}")

    def write_of(name: str, sign: int) -> np.ndarray:
        if name in extra_pairs:
            plus, minus = extra_pairs[name]
            return (plus if sign > 0 else minus).astype(np.float32)
        return (base + sign * args.offset * variants[name]).astype(np.float32)
    out: dict = {
        "run": str(args.run), "checkpoint": str(checkpoint), "values": list(values), "sweep": args.sweep,
        "offset": args.offset, "betas": {"0": beta0, "1": beta1}, "sites": [s.name for s in sites],
        "axis_norm": float(np.linalg.norm(axis)), "argv": sys.argv[1:],
    }

    def finish():
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(out, indent=2, default=float))

    eval_targets = route_targets(levels_of("test", args.n_eval), size=11, n_channels=n_inputs)
    eval_static = jnp.asarray(eval_targets.static)
    eval_null = jnp.asarray(lo.erase(eval_targets.static, [COLOUR[1]]))
    eval_start = jnp.asarray(eval_targets.positions[:, 0, 0])
    plain = rl_edit.make_rollout_fn(policy, unravel, step_limit=args.step_limit)

    def plain_readout(flat, gains):
        reached, steps = [], []
        for start in range(0, len(eval_targets), args.eval_chunk):
            rows = slice(start, start + args.eval_chunk)
            r, s = plain(jnp.asarray(flat, dtype=jnp.float32), eval_static[rows], eval_start[rows],
                         jnp.asarray(gains, dtype=jnp.float32))
            reached.append(np.asarray(r))
            steps.append(np.asarray(s))
        return read_outcomes(eval_targets, np.concatenate(reached), np.concatenate(steps), resamples=50)

    ones = np.ones(n_inputs, np.float32)

    # --- 1. the leftover ------------------------------------------------------------
    if "leftover" in args.parts:
        names = list(variants)
        fit_targets = route_targets(levels_of("valid", args.n_fit), size=11, n_channels=n_inputs)
        moments_fn = lo.make_change_moments_fn(policy, unravel)
        writes = jnp.asarray(np.stack([write_of(n, +1) for n in names]))
        acc, started = None, time.time()
        for start in range(0, len(fit_targets), args.chunk):
            sub = fit_targets.subset(np.arange(start, min(start + args.chunk, len(fit_targets))))
            steps = sub.steps
            static = np.repeat(sub.static, 2, axis=0)
            acc = lo.add_moments(acc, moments_fn(
                jnp.asarray(flat32), writes, jnp.asarray(static), jnp.asarray(lo.erase(static, [COLOUR[1]])),
                jnp.asarray(sub.positions.reshape(-1, steps, 2)), jnp.asarray(sub.role.reshape(-1, steps) != PAD),
            ))
        print(f"\nsizes: {len(fit_targets)} valid levels in {time.time() - started:.0f}s")
        print("\n=== share of the write's activation change that survives erasing the colour-1 objective ===\n")
        print(f"  {'site':<18}" + "".join(f"{i:>10}" for i in range(len(names))))
        shares = {n: [] for n in names}
        for k, site in enumerate(sites):
            row = []
            for v, n in enumerate(names):
                ff, nn = float(acc[k]["ff"][v]), float(acc[k]["nn"][v])
                share = nn / ff if ff > 0 else float("nan")
                shares[n].append(share)
                row.append(share)
            print(f"  {site.name:<18}" + "".join(f"{x:>10.3f}" for x in row))
        print("  columns: " + "; ".join(f"{i} {n}" for i, n in enumerate(names)))
        out["shares"] = {n: [{"site": s.name, "share": x} for s, x in zip(sites, shares[n])] for n in names}
        finish()

        rollout = lo.make_leftover_rollout(policy, unravel, step_limit=args.step_limit)

        def patched_readout(write, mix, w, w_null):
            reached, steps = [], []
            for start in range(0, len(eval_targets), args.eval_chunk):
                rows = slice(start, start + args.eval_chunk)
                r, s = rollout(jnp.asarray(flat32), jnp.asarray(flat32), jnp.asarray(write, dtype=jnp.float32),
                               eval_static[rows], eval_null[rows], eval_start[rows],
                               jnp.asarray(mix), jnp.asarray(w), jnp.asarray(w_null))
                reached.append(np.asarray(r))
                steps.append(np.asarray(s))
            return read_outcomes(eval_targets, np.concatenate(reached), np.concatenate(steps), resamples=50)

        base_readout = plain_readout(base, ones)
        print(f"\nmediation on {len(eval_targets)} test levels; base threshold {base_readout.tau:.2f} "
              f"(reached {base_readout.reached:.1%})")
        out["base_tau"] = base_readout.tau
        out["mediation"] = []
        for name in args.mediate:
            rows = {}
            for sign in (+1, -1):
                write = write_of(name, sign)
                row = {"write": name, "sign": sign, "write_tau": plain_readout(write, ones).tau}
                for group in args.groups:
                    in_group = np.array([group == "all" or s.group == group for s in sites], np.float32)
                    for mode, (w, w_null) in lo.MODES.items():
                        readout = patched_readout(write, in_group, in_group * w, in_group * w_null)
                        row[f"{group}|{mode}"] = readout.tau
                        row[f"{group}|{mode}|reached"] = readout.reached
                rows[sign] = row
                out["mediation"].append(row)
                print(f"  {name} @ {sign * args.offset:+.2f}: write threshold {row['write_tau']:.2f}", flush=True)
            plus, minus = rows[+1], rows[-1]
            print(f"\n=== {name}: fraction of its threshold shift ({plus['write_tau']:.2f} vs {minus['write_tau']:.2f}) "
                  "carried by each patch ===\n")
            print(f"  {'site group':<14}" + "".join(f"{mode:>22}" for mode in lo.MODES))
            summary = {}
            for group in args.groups:
                fractions = {mode: ga.fraction_carried(plus[f"{group}|{mode}"], minus[f"{group}|{mode}"],
                                                       plus["write_tau"], minus["write_tau"]) for mode in lo.MODES}
                summary[group] = fractions
                print(f"  {group:<14}" + "".join(f"{fractions[mode]:>22.2f}" for mode in lo.MODES))
            out.setdefault("fractions", {})[name] = summary
            finish()

    # --- 2. the lever's ceiling -----------------------------------------------------
    if "signal" in args.parts:
        grid = [("base", ones.copy())]
        for g in args.bright_1:
            row = ones.copy()
            row[COLOUR[1]] = g
            grid.append((f"c1 x{g:g}", row))
        for g in args.dim_0:
            row = ones.copy()
            row[COLOUR[0]] = g
            grid.append((f"c0 x{g:g}", row))
        gains = np.stack([g for _, g in grid])
        signal_targets = route_targets(levels_of("valid", args.n_signal), size=11, n_channels=n_inputs)
        signals_fn = lo.make_signal_fn(policy, unravel)
        acc, started = None, time.time()
        for start in range(0, len(signal_targets), args.signal_chunk):
            sub = signal_targets.subset(np.arange(start, min(start + args.signal_chunk, len(signal_targets))))
            steps = sub.steps
            static = np.repeat(sub.static, 2, axis=0)
            acc = lo.add_moments(acc, signals_fn(
                jnp.asarray(flat32), jnp.asarray(gains), jnp.asarray(static),
                jnp.asarray(lo.erase(static, [COLOUR[0]])), jnp.asarray(lo.erase(static, [COLOUR[1]])),
                jnp.asarray(sub.positions.reshape(-1, steps, 2)), jnp.asarray(sub.role.reshape(-1, steps) != PAD),
            ))
        print(f"\nsignals: {len(signal_targets)} valid levels in {time.time() - started:.0f}s")
        taus = [plain_readout(base, g) for _, g in grid]
        rows = []
        for i, (label, g) in enumerate(grid):
            row = {"label": label, "gain_c0": float(g[COLOUR[0]]), "gain_c1": float(g[COLOUR[1]]),
                   "tau": taus[i].tau, "reached": taus[i].reached}
            for k, site in enumerate(sites):
                row[f"s0|{site.name}"] = float(np.sqrt(acc[k]["s0"][i]))
                row[f"s1|{site.name}"] = float(np.sqrt(acc[k]["s1"][i]))
            rows.append(row)
        out["signal"] = rows
        for site in ("input conv", sites[-3].name, "dense"):
            print(f"\n=== objective signal at {site}, relative to the base, against the threshold ===\n")
            print(f"  {'condition':<12}{'tau':>8}{'reached':>9}{'S0':>9}{'S1':>9}{'ln S1/S0':>10}")
            ref0, ref1 = rows[0][f"s0|{site}"], rows[0][f"s1|{site}"]
            for row in rows:
                s0, s1 = row[f"s0|{site}"] / ref0, row[f"s1|{site}"] / ref1
                ratio = np.log(s1 / s0) if s0 > 0 and s1 > 0 else float("nan")
                print(f"  {row['label']:<12}{row['tau']:>8.2f}{row['reached']:>9.1%}{s0:>9.3f}{s1:>9.3f}{ratio:>10.3f}")
        finish()

    # --- 3. the axis's own mix ------------------------------------------------------
    if "mix" in args.parts:
        # beta1 - beta0 is the axis's log-gain difference per unit offset. Both pure levers
        # below carry that same slope at first order, one by dimming and one by brightening,
        # so they differ only in which end of the lever they use.
        composites = {
            "axis": axis,
            "axis, dimming mix": axis - gain + plane_gain(COLOUR[0], beta0 - beta1),
            "axis, brightening mix": axis - gain + plane_gain(COLOUR[1], beta1 - beta0),
            "pure dim c0, matched": plane_gain(COLOUR[0], beta0 - beta1),
            "pure brighten c1, matched": plane_gain(COLOUR[1], beta1 - beta0),
            "gain part, in weights": gain,
        }
        print("\n=== the axis with its gain slice swapped, and the two pure levers matched at first order ===\n")
        for name, direction in composites.items():
            print(f"  |{name}| = {np.linalg.norm(direction):.3g}")
        print()
        print(f"  {'write':<26}{'offset':>8}{'tau':>8}{'reached':>9}{'route':>8}")
        out["mix"] = []
        for name, direction in composites.items():
            for dv in args.mix_offsets:
                readout = plain_readout(base + dv * direction, ones)
                out["mix"].append({"write": name, "offset": dv, "tau": readout.tau, "reached": readout.reached,
                                   "on_route": readout.on_route})
                print(f"  {name:<26}{dv:>8.2f}{readout.tau:>8.2f}{readout.reached:>9.1%}{readout.on_route:>8.1%}",
                      flush=True)
            finish()

    finish()
    print("\nLEFTOVER_DONE")


if __name__ == "__main__":
    main()
