"""Does the value axis change the plan? Per-cell distance fields under axis writes and their brightness twins.

    uv run python scripts/plan_invariance.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --json figures/data/plan_invariance/novalue11g1.s1234.json

Separates two readings of the DRC's value axis (:mod:`goalmisgen.analysis.plan_invariance`):

**gain** -- value is how loudly a colour is seen, so the colour's distance field
changes with the write, from the first tick, along that field;
**factorised** -- the fields stay put and the write lands somewhere the distance
probes do not read, late, near the agent or the fork.

The gain reference is not a hypothesis but a manipulation: the untouched network
with the swept colour's plane scaled at the input, at the gain whose threshold
matches each axis dose. Every number is printed for it alongside the axis, so
"looks like a gain" means "looks like that".

**Noise in the axis.** A fitted axis is about 60% reliable in weight space: the
rest is fine-tuning noise, which moves activations without moving behaviour and
would dilute every activation-level number here. The o0 sweep's axis carries the
same value knob with the opposite sign and *independent* noise, so products of
the two writes' activation changes -- ``<dh(o1 @ +d), dh(o0 @ -d)>`` -- keep the
shared component and average the noise away. Those "reliable" columns are the
ones to read; the raw ones are printed so the dilution stays visible.

Phases:

1. **behaviour** -- threshold for each dose of both sweeps' axes, a brightness
   curve on the swept colour's plane, the matched gains, the input layer's slice
   alone, and a module-shaped random write scaled to change the activations as
   much as the axis does. Held-out levels, greedy.
2. **probes** -- ridge probes for the distance to each objective, one per tick and
   per layer (and all layers together), fitted once on the base network's state
   on the fit split, and the same on an untrained network of the same shape.
3. **the plan under each write** -- on test levels, every tick of the first two
   steps' worth of recurrence on the initial observation: the base probes' R^2
   under the write, how the decoded field moved (offset, slope, rms and by true
   distance, in field SDs), where the activation change sits, how much lies along
   the probes' readout directions, and how it lines up with the matched
   brightness's change. Scored on levels whose chosen objective is the same under
   every axis, brightness and slice write, so a plan that moved because the
   decision flipped cannot pass for one the write moved.
4. **null objects** -- the same writes with one or both objectives removed. A pure
   input gain on an empty plane changes nothing; a global value variable would
   still move the state.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis import plan_invariance as pi  # noqa: E402
from goalmisgen.analysis.weights import cosine  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402

N_FEATURES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--fit-split", type=str, default="valid")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sweep", type=str, default="o1", help="The written axis; the other sweep's axis is its independent twin.")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--doses", type=float, nargs="+", default=[-0.3, -0.15, 0.15, 0.3])
    parser.add_argument("--gains", type=float, nargs="+", default=[0.35, 0.5, 0.7, 0.85, 1.2, 1.5, 2.0, 3.0])
    parser.add_argument("--behaviour-episodes", type=int, default=2048)
    parser.add_argument("--fit-levels", type=int, default=1024)
    parser.add_argument("--test-levels", type=int, default=1024)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--state", choices=("h", "c"), default="h", help="Probe the ConvLSTM's hidden state or its cell state.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:  # noqa: C901 - one linear pipeline, read top to bottom
    args = parse_args()
    print(header())
    print()

    import jax
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis import collect_episode_outcomes, fields, metrics, targets
    from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
    from goalmisgen.analysis.input_gain import scale_planes
    from goalmisgen.analysis.layers import module_groups, restrict
    from goalmisgen.analysis.probes import apply_linear, fit_ridge
    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
    from goalmisgen.volume import sweep_index
    from scripts.restricted_writes import load_sweep

    payload = json.loads((args.run / "BASE.json").read_text())
    checkpoint = args.run / payload["checkpoint"]
    values = tuple(payload["values"])
    swept = sweep_index(args.sweep)
    twin_sweep = f"o{1 - swept}"
    plane = FIRST_FEATURE_CHANNEL + swept

    def config(split: str) -> MazeConfig:
        return MazeConfig(
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
            dataset_split=split,
            asynchronous=False,
            seed=args.seed,
        )

    policy, _, _, base_state, _ = load_train_state(checkpoint, env_cfg=config(args.split))
    base_flat, unravel = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)
    groups, _ = module_groups(base_state.params)
    axes = {}
    for sweep in (args.sweep, twin_sweep):
        offsets, diffs = load_sweep(args.run, sweep, values, args, config(args.split), base_flat)
        axes[sweep], _ = fit_axis_and_drift(offsets, diffs)
        print(f"axis   {sweep}: {len(offsets)} arms, |axis| {np.linalg.norm(axes[sweep]):.3g}")
        del diffs
    axis, twin = axes[args.sweep], -axes[twin_sweep]  # the twin's sign flipped so both lower the threshold at +d
    input_slice = restrict(axis, groups, "input conv")
    random_shape = pi.module_matched_random(axis, groups, seed=args.seed + 7)
    envs = config(args.split).make()
    _, _, untrained = policy.cfg.init_params(envs, jax.random.PRNGKey(12345))
    n_layers = int(policy.cfg.n_recurrent)
    layers = list(range(n_layers + 1))  # the last is every layer together
    layer_name = [str(i) for i in range(n_layers)] + ["all"]
    print(f"base   {checkpoint}  ({base_flat.size:,} parameters), values {values}")
    print(f"       cos(axis {args.sweep}, -axis {twin_sweep}) {cosine(axis, twin, nan_if_zero=True):+.3f}; brightness on input plane {plane}")

    def tree(flat):
        return unravel(np.asarray(flat, dtype=np.float32))

    out: dict = {"run": str(args.run), "checkpoint": str(checkpoint), "values": list(values), "sweep": args.sweep,
                 "twin": twin_sweep, "axis_norm": float(np.linalg.norm(axis)), "cos_axis_twin": cosine(axis, twin, nan_if_zero=True),
                 "ticks": args.ticks, "state": args.state, "layers": layer_name}
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    run = pi.make_tick_runner(policy, args.ticks, state=args.state)

    def states(params, observations, gains=None, batch: int = 256) -> np.ndarray:
        """(ticks, layers, n, H, W, C) float32."""
        parts = []
        for start in range(0, len(observations), batch):
            chunk = observations[start : start + batch]
            if gains:
                chunk = scale_planes(chunk, gains, channel_axis=1)
            parts.append(np.asarray(run(params, chunk)[0]))
        return np.concatenate(parts, axis=2)

    def view(h, t, layer):
        """(n, H, W, C) at one tick; ``layer == n_layers`` concatenates every layer."""
        return h[t, layer] if layer < n_layers else np.concatenate([h[t, i] for i in range(n_layers)], axis=-1)

    # ------------------------------------------------------------------ levels
    def initial_observations(split: str, n: int, seed0: int) -> list[np.ndarray]:
        split_envs = config(split).make()
        batches = []
        for b in range(-(-n // args.num_envs)):
            observations, _ = split_envs.reset(seed=seed0 + 1000 * b)
            batches.append(np.asarray(observations, dtype=np.float32))
        return batches

    test_seed = args.seed + 10_000
    test_batches = initial_observations(args.split, args.test_levels, test_seed)
    fit_obs = np.concatenate(initial_observations(args.fit_split, args.fit_levels, args.seed + 20_000))[: args.fit_levels]
    test_obs = np.concatenate(test_batches)[: args.test_levels]
    hwc_test = np.moveaxis(test_obs, 1, -1)
    hwc_fit = np.moveaxis(fit_obs, 1, -1)
    geos = [pi.level_geometry(o) for o in hwc_test]
    free = np.stack([g.free for g in geos])
    h_base = states(tree(base_flat), test_obs)

    # ------------------------------------------------------------------ 1. behaviour
    def threshold(flat, gains=None) -> float:
        params = tree(flat)
        carry = policy.apply(params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
        state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

        def act(observations, starts):
            if gains:
                observations = scale_planes(observations, gains, channel_axis=1)
            state["carry"], action, _, state["key"] = get_action(params, state["carry"], observations, starts, state["key"], temperature=0.0)
            return np.asarray(action)

        outcomes = collect_episode_outcomes(envs, act, args.behaviour_episodes, seed=args.seed)
        gaps, took_richer, _ = value_distance_decisions(outcomes)
        return float(indifference_point(gaps, took_richer))

    big = max(abs(d) for d in args.doses)
    conditions: dict[str, tuple[np.ndarray, dict | None]] = {"base": (base_flat, None)}
    for dose in args.doses:
        conditions[f"axis {dose:+.2f}"] = (base_flat + dose * axis, None)
        conditions[f"twin {dose:+.2f}"] = (base_flat + dose * twin, None)
    for sign in (-1, 1):
        conditions[f"input slice {sign * big:+.2f}"] = (base_flat + sign * big * input_slice, None)

    # the random control changes the activations as much as the axis does, not the weights: a module-shaped random
    # direction at the axis's own norm moves the state a hundred times further and breaks the agent
    probe_levels = test_obs[:128]
    h_base_small = states(tree(base_flat), probe_levels)

    def small_energy(flat):
        h = states(tree(flat), probe_levels)
        return float(np.mean(np.sum((h[-1] - h_base_small[-1]).astype(np.float64) ** 2, axis=-1)[:, free[:128]]))

    target = small_energy(base_flat + big * axis)
    low, high = 1e-4, 1.0
    for _ in range(30):
        mid = np.sqrt(low * high)
        if small_energy(base_flat + big * mid * random_shape) < target:
            low = mid
        else:
            high = mid
    random_scale = float(np.sqrt(low * high))
    print(f"random control: module-shaped, scaled x{random_scale:.4f} of the axis's per-module norms to match its activation change")
    for sign in (-1, 1):
        conditions[f"random {sign * big:+.2f}"] = (base_flat + sign * big * random_scale * random_shape, None)

    print(f"\n=== 1. behaviour: threshold (steps) on {args.behaviour_episodes} held-out levels ===\n")
    behaviour: dict = {}
    for name, (flat, _) in conditions.items():
        behaviour[name] = threshold(flat)
        print(f"  {name:<34} {behaviour[name]:7.2f}", flush=True)
    curve = {1.0: behaviour["base"]}
    for gain in args.gains:
        curve[gain] = threshold(base_flat, {plane: gain})
        print(f"  colour {swept} x{gain:<5g}                    {curve[gain]:7.2f}", flush=True)
    gains_sorted = sorted(curve)
    matched = {}
    for dose in args.doses:
        gain = pi.interpolate_gain(gains_sorted, [curve[g] for g in gains_sorted], behaviour[f"axis {dose:+.2f}"])
        matched[dose] = gain
        name = f"brightness {dose:+.2f}"
        conditions[name] = (base_flat, {plane: gain})
        behaviour[name] = threshold(base_flat, {plane: gain})
        print(f"  {name:<22} (x{gain:.3f})    {behaviour[name]:7.2f}   (axis {behaviour[f'axis {dose:+.2f}']:.2f})", flush=True)
    out["behaviour"] = behaviour
    out["brightness_curve"] = {str(g): curve[g] for g in gains_sorted}
    out["matched_gains"] = {f"{d:+.2f}": g for d, g in matched.items()}
    out["random_scale"] = random_scale
    trees = {name: tree(flat) for name, (flat, _) in conditions.items()}

    def choices(params, gains, seed0: int, expected) -> np.ndarray:
        """First episode per environment slot: which objective was reached, -1 on a timeout."""
        picked = []
        for b, first in enumerate(expected):
            observations, _ = envs.reset(seed=seed0 + 1000 * b)
            if not np.array_equal(np.asarray(observations, np.float32), first):
                raise RuntimeError("the same seed produced different levels; conditions would not be comparable")
            carry = policy.apply(params, jax.random.PRNGKey(0), envs.observation_space.shape, method=policy.initialize_carry)
            starts = np.zeros(envs.num_envs, dtype=bool)
            key = jax.random.PRNGKey(0)
            result = np.full(envs.num_envs, -2)
            while (result == -2).any():
                seen = scale_planes(observations, gains, channel_axis=1) if gains else observations
                carry, action, _, key = get_action(params, carry, seen, starts, key, temperature=0.0)
                observations, _, terminated, truncated, info = envs.step(np.asarray(action))
                done = np.logical_or(terminated, truncated)
                starts = done
                for index in np.flatnonzero(done & (result == -2)):
                    final = info["final_info"][index] or {}
                    result[index] = int(final["reached_feature_id"]) if final.get("reached_objective") else -1
            picked.append(result)
        return np.concatenate(picked)

    level_choices = {name: choices(trees[name], gains, test_seed, test_batches)[: args.test_levels]
                     for name, (_, gains) in conditions.items()}
    stable = pi.stable_levels({k: v for k, v in level_choices.items() if not k.startswith("random")})
    print(f"\nlevels: {len(test_obs)} test, {stable.sum()} ({np.mean(stable):.0%}) with the same choice under every axis, twin,"
          f" brightness and slice write; {len(fit_obs)} fit-split levels for the probes")
    flips = {k: float(np.mean(v != level_choices["base"])) for k, v in level_choices.items()}
    print("  flipped from base: " + ", ".join(f"{k} {v:.1%}" for k, v in flips.items() if k != "base"))
    out["levels"] = {"test": len(test_obs), "stable": int(stable.sum()), "flipped_from_base": flips}

    # ------------------------------------------------------------------ 2. probes
    def labelled(hwc):
        """Per objective: (level, row, col) of scoreable cells, their distance and Manhattan confound."""
        result = []
        for f in range(N_FEATURES):
            tgt = targets.DistanceToObjective(targets.fixed(f), name=f"d->f{f}", n_features=N_FEATURES)
            idx, ys, conf = [], [], []
            for i, o in enumerate(hwc):
                snap = type("Snap", (), {"observation": o, "info": {}})()
                y, c = tgt.labels(snap), tgt.confound(snap)[..., 0]
                rows, cols = np.nonzero(np.isfinite(y) & np.isfinite(c))
                idx.append(np.stack([np.full(len(rows), i), rows, cols], axis=1))
                ys.append(y[rows, cols])
                conf.append(c[rows, cols])
            result.append((np.concatenate(idx), np.concatenate(ys), np.concatenate(conf)))
        return result

    fit_labels, test_labels = labelled(hwc_fit), labelled(hwc_test)

    def at(grid, index):
        return grid[index[:, 0], index[:, 1], index[:, 2]]

    def fit_probes(h_fit):
        probes = {}
        for t in range(args.ticks):
            for layer in layers:
                grid = view(h_fit, t, layer)
                for f, (index, y, conf) in enumerate(fit_labels):
                    x = at(grid, index).astype(np.float64)
                    data = fields.CellData(x=x, y=y, confound=np.stack([conf, conf], 1), episode=index[:, 0],
                                           mask_fraction=1.0, dropped_columns=0)
                    l2, _ = fields.choose_l2(data)
                    w, mean, std = fit_ridge(x, y, l2=l2)
                    scale, offset = metrics.affine_fit(y, apply_linear(x, w, mean, std))
                    probes[(f, t, layer)] = (w, mean, std, scale, offset)
        return probes

    def predict(probe, x):
        w, mean, std, scale, offset = probe
        return scale * apply_linear(x.astype(np.float64), w, mean, std) + offset

    def probe_scores(h, probe_set):
        rows = {}
        for t in range(args.ticks):
            for layer in layers:
                grid = view(h, t, layer)
                for f, (index, y, conf) in enumerate(test_labels):
                    p = predict(probe_set[(f, t, layer)], at(grid, index))
                    hard = (y - conf) >= 4
                    rows[(f, t, layer)] = (metrics.r2(y, p), metrics.r2(y[hard], p[hard]), p)
        return rows

    print("\n=== 2. probes: distance to each objective, fitted on the base network's state ===\n", flush=True)
    probes = fit_probes(states(tree(base_flat), fit_obs))
    base_scores = probe_scores(h_base, probes)
    untrained_scores = probe_scores(states(untrained, test_obs), fit_probes(states(untrained, fit_obs)))
    print(f"  {'tick':>4} {'layer':>5}   {'trained R2 d->0 / d->1 (hard cells)':^36}   {'untrained R2 d->0 / d->1 (hard)':^36}")
    for t in range(args.ticks):
        for layer in layers:
            tr = [base_scores[(f, t, layer)] for f in range(2)]
            un = [untrained_scores[(f, t, layer)] for f in range(2)]
            print(f"  {t + 1:>4} {layer_name[layer]:>5}   {tr[0][0]:6.3f} ({tr[0][1]:6.3f}) / {tr[1][0]:6.3f} ({tr[1][1]:6.3f})"
                  f"     {un[0][0]:6.3f} ({un[0][1]:6.3f}) / {un[1][0]:6.3f} ({un[1][1]:6.3f})")
    out["probes"] = [{"feature": f, "tick": t + 1, "layer": layer_name[layer], "r2": base_scores[(f, t, layer)][0],
                      "hard_r2": base_scores[(f, t, layer)][1], "untrained_r2": untrained_scores[(f, t, layer)][0],
                      "untrained_hard_r2": untrained_scores[(f, t, layer)][1]}
                     for f in range(2) for t in range(args.ticks) for layer in layers]
    directions = {(t, layer): np.stack([pi.readout_directions(probes[(f, t, layer)][0][:-1], probes[(f, t, layer)][2]) for f in range(2)])
                  for t in range(args.ticks) for layer in layers}
    del untrained_scores

    # ------------------------------------------------------------------ 3. the plan under each write
    print("\n=== 3. the plan under each write ===", flush=True)
    deltas: dict[str, np.ndarray] = {}
    records = []
    for name, (_, gains) in conditions.items():
        if name == "base":
            continue
        h = states(trees[name], test_obs, gains)
        deltas[name] = (h - h_base).astype(np.float32)
        for t in range(args.ticks):
            for layer in layers:
                grid = view(h, t, layer)
                d = view(deltas[name], t, layer)
                sq = np.sum(d.astype(np.float64) ** 2, axis=-1)
                for subset_name, subset in (("stable", stable), ("all", np.ones_like(stable))):
                    row = {"condition": name, "subset": subset_name, "tick": t + 1, "layer": layer_name[layer]}
                    for f, (index, y, conf) in enumerate(test_labels):
                        keep = subset[index[:, 0]]
                        p = predict(probes[(f, t, layer)], at(grid, index[keep]))
                        row[f"r2_{f}"] = metrics.r2(y[keep], p)
                        row[f"shift_{f}"] = pi.field_shift(base_scores[(f, t, layer)][2][keep], p, y[keep])
                    chosen = [g for g, s in zip(geos, subset) if s]
                    base_cells = view(h_base, t, layer)[subset][free[subset]]
                    row["relative_change"] = float(sq[subset][free[subset]].mean() / np.mean(np.sum(base_cells.astype(np.float64) ** 2, -1)))
                    row["loci"] = pi.locus_enrichment(sq[subset], chosen)
                    row["nearness"] = pi.map_correlations(sq[subset], chosen)
                    row["probe_share"] = pi.subspace_share(d[subset][free[subset]], directions[(t, layer)])
                    records.append(row)
        del h
    out["plan"] = records

    # the reliable part: products of the axis's and its twin's changes, which share the knob and not the noise
    def dot_maps(a, b):
        return np.sum(a.astype(np.float64) * b.astype(np.float64), axis=-1)

    reliable = []
    for dose in args.doses:
        a, b = deltas[f"axis {dose:+.2f}"], deltas[f"twin {dose:+.2f}"]
        bright, rnd, sl = deltas[f"brightness {dose:+.2f}"], deltas[f"random {np.sign(dose) * big:+.2f}"], deltas[f"input slice {np.sign(dose) * big:+.2f}"]
        for t in range(args.ticks):
            for layer in layers:
                m = stable
                va, vb = view(a, t, layer)[m], view(b, t, layer)[m]
                cross = dot_maps(va, vb)
                chosen = [g for g, s in zip(geos, m) if s]
                fa, fb = va[free[m]], vb[free[m]]
                shared = float(np.sum(fa.astype(np.float64) * fb))
                reliability = shared / float(np.sqrt(np.sum(fa.astype(np.float64) ** 2) * np.sum(fb.astype(np.float64) ** 2)))
                q, _ = np.linalg.qr(directions[(t, layer)].T)
                row = {"dose": dose, "tick": t + 1, "layer": layer_name[layer], "reliability": reliability,
                       "loci": pi.locus_enrichment(cross, chosen), "nearness": pi.map_correlations(cross, chosen),
                       "probe_share": float(np.sum((fa @ q) * (fb @ q)) / shared) if shared > 0 else float("nan")}
                ceiling = np.sqrt(max(reliability, 1e-12))
                for label, other in (("brightness", bright), ("input slice", sl), ("random", rnd)):
                    fo = view(other, t, layer)[m][free[m]]
                    raw = 0.5 * (cosine(fa, fo, nan_if_zero=True) + cosine(fb, fo, nan_if_zero=True))
                    row[f"cos {label}"] = raw
                    row[f"cos {label} corrected"] = raw / ceiling
                reliable.append(row)
    out["reliable"] = reliable

    def fmt(v, width=6, sign=False):
        return f"{v:+{width}.2f}" if sign else f"{v:{width}.2f}"

    print("\n  Raw, stable levels. off/slp/rms: the base probe's decoded field under the write against under the base, in field SDs;")
    print("  chg: mean |dh|^2 / mean |h|^2 per free cell; prb%: share of dh on the two probes' readout directions;")
    print("  agent..obj1: mean |dh|^2 there over its mean on free cells; nr0/nr1/nrA: per-level corr of the |dh|^2 map with nearness.\n")
    shown = [f"axis {big:+.2f}", f"twin {big:+.2f}", f"brightness {big:+.2f}", f"input slice {big:+.2f}", f"random {big:+.2f}",
             f"axis {-big:+.2f}", f"twin {-big:+.2f}", f"brightness {-big:+.2f}", f"input slice {-big:+.2f}", f"random {-big:+.2f}"]
    print(f"  {'condition':<18}{'tk':>3}{'lay':>4} {'R2 d0':>6} {'R2 d1':>6} {'off0':>6} {'off1':>6} {'slp0':>5} {'slp1':>5} {'rms0':>5} {'rms1':>5}"
          f" {'chg':>7} {'prb%':>5} {'agent':>5} {'rt0':>5} {'rt1':>5} {'shrd':>5} {'obj0':>5} {'obj1':>5} {'nr0':>5} {'nr1':>5} {'nrA':>5}")
    for t in range(args.ticks):
        for layer in layers:
            b0, b1 = base_scores[(0, t, layer)][0], base_scores[(1, t, layer)][0]
            print(f"  {'base':<18}{t + 1:>3}{layer_name[layer]:>4} {b0:6.3f} {b1:6.3f}")
            for name in shown:
                r = next(x for x in records if x["condition"] == name and x["subset"] == "stable" and x["tick"] == t + 1 and x["layer"] == layer_name[layer])
                lo, nr, s0, s1 = r["loci"], r["nearness"], r["shift_0"], r["shift_1"]
                print(f"  {name:<18}{t + 1:>3}{layer_name[layer]:>4} {r['r2_0']:6.3f} {r['r2_1']:6.3f} {fmt(s0['offset'], sign=True)} {fmt(s1['offset'], sign=True)}"
                      f" {s0['slope']:5.2f} {s1['slope']:5.2f} {s0['rms']:5.2f} {s1['rms']:5.2f} {r['relative_change']:7.4f} {100 * r['probe_share']:5.1f}"
                      f" {lo['agent']:5.2f} {lo['route 0 only']:5.2f} {lo['route 1 only']:5.2f} {lo['shared route']:5.2f} {lo['objective 0']:5.2f}"
                      f" {lo['objective 1']:5.2f} {nr['objective 0']:+5.2f} {nr['objective 1']:+5.2f} {nr['agent']:+5.2f}")
        print()

    print("  Reliable part (<dh axis, dh twin>), stable levels. rel: cosine of the two changes (its square root is the ceiling);")
    print("  cos X: mean cosine of the two with X's change, raw and over the ceiling; prb%: reliable share on the probe directions")
    print(f"  (chance {200 / h_base.shape[-1]:.1f}% per layer, {200 / (n_layers * h_base.shape[-1]):.1f}% all); loci and nearness as above, on the product map.\n")
    print(f"  {'dose':>5}{'tk':>3}{'lay':>4} {'rel':>5} {'cosBr':>6} {'/ceil':>6} {'cosSl':>6} {'/ceil':>6} {'cosRn':>6} {'/ceil':>6} {'prb%':>6}"
          f" {'agent':>5} {'rt0':>5} {'rt1':>5} {'shrd':>5} {'obj0':>5} {'obj1':>5} {'off':>5} {'nr0':>5} {'nr1':>5} {'nrA':>5}")
    for dose in (big, -big):
        for r in (x for x in reliable if x["dose"] == dose):
            lo, nr = r["loci"], r["nearness"]
            print(f"  {dose:+5.2f}{r['tick']:>3}{r['layer']:>4} {r['reliability']:5.2f} {r['cos brightness']:+6.2f} {r['cos brightness corrected']:+6.2f}"
                  f" {r['cos input slice']:+6.2f} {r['cos input slice corrected']:+6.2f} {r['cos random']:+6.2f} {r['cos random corrected']:+6.2f}"
                  f" {100 * r['probe_share']:6.1f} {lo['agent']:5.2f} {lo['route 0 only']:5.2f} {lo['route 1 only']:5.2f} {lo['shared route']:5.2f}"
                  f" {lo['objective 0']:5.2f} {lo['objective 1']:5.2f} {lo['off route']:5.2f} {nr['objective 0']:+5.2f} {nr['objective 1']:+5.2f} {nr['agent']:+5.2f}")
        print()

    # ------------------------------------------------------------------ 4. null objects
    print("=== 4. null objects: activation change with objectives removed, over the same with both present (all layers) ===\n", flush=True)
    null_rows = []
    removals = {"no objective 0": [0], "no objective 1": [1], "no objectives": [0, 1]}
    null_names = [f"axis {big:+.2f}", f"twin {big:+.2f}", f"brightness {big:+.2f}", f"input slice {big:+.2f}", f"random {big:+.2f}",
                  f"axis {-big:+.2f}", f"twin {-big:+.2f}", f"brightness {-big:+.2f}", f"input slice {-big:+.2f}", f"random {-big:+.2f}"]
    for removal, ids in removals.items():
        emptied = pi.remove_objectives(test_obs, ids)
        h_base_e = states(tree(base_flat), emptied)
        empty_deltas = {}
        for name in null_names:
            _, gains = conditions[name]
            empty_deltas[name] = (states(trees[name], emptied, gains) - h_base_e).astype(np.float32)
        for t in range(args.ticks):
            layer = n_layers
            row = {"removal": removal, "tick": t + 1}
            for name in null_names:
                e, p = view(empty_deltas[name], t, layer)[free], view(deltas[name], t, layer)[free]
                row[name] = float(np.sum(e.astype(np.float64) ** 2) / np.sum(p.astype(np.float64) ** 2))
            for sign in (1, -1):
                ea, eb = view(empty_deltas[f"axis {sign * big:+.2f}"], t, layer)[free], view(empty_deltas[f"twin {sign * big:+.2f}"], t, layer)[free]
                pa, pb = view(deltas[f"axis {sign * big:+.2f}"], t, layer)[free], view(deltas[f"twin {sign * big:+.2f}"], t, layer)[free]
                row[f"reliable {sign * big:+.2f}"] = float(np.sum(ea.astype(np.float64) * eb) / np.sum(pa.astype(np.float64) * pb))
                # does the reliable change that survives look like the one with the objectives present?
                row[f"reliable pattern cos {sign * big:+.2f}"] = cosine(ea + eb, pa + pb, nan_if_zero=True)
            # the remaining objective's decoded field under the axis, at the layer-concatenated probe
            for f in range(N_FEATURES):
                if f in ids:
                    continue
                index, y, _ = test_labels[f]
                pb_ = predict(probes[(f, t, layer)], at(view(h_base_e, t, layer), index))
                for name in (f"axis {big:+.2f}", f"brightness {big:+.2f}", f"random {big:+.2f}"):
                    pc = predict(probes[(f, t, layer)], at(view(h_base_e + empty_deltas[name], t, layer), index))
                    row[f"shift_{f} {name}"] = pi.field_shift(pb_, pc, y)["offset"]
            null_rows.append(row)
        del h_base_e, empty_deltas
    out["null_objects"] = null_rows
    for removal in removals:
        print(f"  {removal}: |dh|^2 ratio per tick 1..{args.ticks}")
        for key in null_names + [f"reliable {big:+.2f}", f"reliable {-big:+.2f}", f"reliable pattern cos {big:+.2f}", f"reliable pattern cos {-big:+.2f}"]:
            vals = [r[key] for r in null_rows if r["removal"] == removal]
            print(f"    {key:<28} " + " ".join(f"{v:6.2f}" for v in vals))
        rows_here = [r for r in null_rows if r["removal"] == removal]
        for key in [k for k in rows_here[0] if k.startswith("shift_")]:
            vals = [r[key] for r in rows_here]
            print(f"    field offset {key:<18} " + " ".join(f"{v:+6.2f}" for v in vals))
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {args.json}")
    print("PLAN_INVARIANCE_DONE")


if __name__ == "__main__":
    main()
