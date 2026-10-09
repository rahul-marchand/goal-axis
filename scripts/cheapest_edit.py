"""The cheapest edit that moves an agent's exchange rate, and whether it is the value axis.

    uv run python scripts/cheapest_edit.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --family dense --seeds 0 1 2 \
        --json figures/data/cheapest_edit/novalue11g1.s1234.dense.json
    uv run python scripts/cheapest_edit.py /workspace/data/offline/runs/bcnv11.s1 --offline \
        --demos /workspace/data/offline/demos --family dense ...

The fine-tuned axis ``v`` is the direction SGD took when the reward changed. This
script asks the network-side question instead: with the base frozen and only an
edit trained, what is the smallest ``delta`` such that ``base + shift * delta``
chooses like a threshold of ``tau_base + shift`` -- for several shifts at once,
with the dose fixed in steps rather than learned (:mod:`goalmisgen.edits`).

Per base, in order:

1. **Targets.** The base's own threshold and logistic scale are measured on the
   edit levels; the search is trained toward ``tau_base + shift`` at that scale.
   Edit levels and evaluation levels are disjoint (``valid`` / ``test`` splits
   for the RL agents, ``valid`` / ``test`` demonstrations for the route model),
   and neither is the base's own training split.
2. **Search**, once per seed, in the chosen family: ``dense`` (all or
   ``--modules``), ``lowrank`` (``--rank``), ``sparse`` (dense, then the ``--k``
   largest coordinates refitted), ``gain`` (two scalars on the colour planes).
3. **Evaluation** on held-out levels at every ``--eval-shifts`` dose, including
   doses outside the trained ones and far enough down to ask for a threshold
   below zero: threshold with a bootstrap interval, reach, shortest-route rate.
   The fine-tuned axis is written through the same rollout, rescaled to steps,
   so the two are measured identically.
4. **Comparison** with ``v``: cosine overall and per module, split-half
   reliability of ``v`` and seed-to-seed agreement of ``u`` (the ceiling for the
   cosine), and the function-space cosine of ``J u`` against ``J v``.

``--rule contingent`` is the control target: shift up where the richer goal is
above, down where it is below. A value slot does not provide it; if it is as
cheap as the uniform shift, the budget is not doing its job.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.input_gain import input_kernel  # noqa: E402
from goalmisgen.analysis.layers import module_groups  # noqa: E402
from goalmisgen.analysis.weights import cosine, fit_axis_and_drift, split_half_reliability  # noqa: E402
from goalmisgen.edits.targets import route_targets, usable  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--offline", action="store_true", help="The base is a route model (bcnv11.*), not a cleanba agent.")
    parser.add_argument("--levels", type=str, help="RL: level dataset directory.")
    parser.add_argument("--demos", type=Path, help="Offline: directory holding valid.rho100 and test.rho100.")
    parser.add_argument("--n-edit", type=int, default=4096)
    parser.add_argument("--n-eval", type=int, default=2048)
    parser.add_argument("--family", choices=["dense", "lowrank", "sparse", "gain", "rotation"], default="dense")
    parser.add_argument("--modules", nargs="+", default=None, help="Restrict dense/lowrank/sparse to these modules.")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--rule", choices=["threshold", "contingent"], default="threshold")
    parser.add_argument("--shifts", type=float, nargs="+", default=[-2.0, 2.0, 4.0])
    parser.add_argument("--eval-shifts", type=float, nargs="+", default=[-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--init-scale", type=float, default=0.0, help="Norm of a random initial edit; 0 starts at zero.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--fork-weight", type=float, default=1.0)
    parser.add_argument("--target", choices=["curve", "sign", "margin", "soft"], default="curve")
    parser.add_argument("--nav-weight", type=float, default=10.0, help="Margin target: weight of the KL to the base off the fork.")
    parser.add_argument("--max-norm", type=float, default=None, help="Budget on |delta| per step; see --norm-units.")
    parser.add_argument("--norm-units", choices=["v", "abs"], default="v", help="--max-norm in multiples of |v| per step, or absolute.")
    parser.add_argument("--sweep", type=str, default="o1", help="Which fine-tuned axis to compare with.")
    parser.add_argument("--arm-steps", type=int, default=400_000, help="RL arm budget.")
    parser.add_argument("--steps-tag", type=int, default=1000, help="Offline arm budget, as in arm directory names.")
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--no-axis", action="store_true", help="Skip the comparison with a fine-tuned axis.")
    parser.add_argument("--on-policy", action="store_true", help="RL: read decisions where the edited agent commits in closed loop.")
    parser.add_argument("--commit-weight", type=float, default=1.0, help="On-policy: charge for committing less often than the base.")
    parser.add_argument("--soft-rates", action="store_true", help="Curve target: score soft decisions instead of straight-through greedy ones.")
    parser.add_argument("--init-from", type=str, default=None, help="Dense family: start from PATH.npz:KEY instead of zero.")
    parser.add_argument("--horizon", type=int, default=40, help="On-policy: steps of the agent's own rollout kept per level.")
    parser.add_argument("--loss-only", action="store_true", help="Score no edit and v under every target, then stop.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--save-deltas", type=Path, default=None, help="npz of every searched delta (full length).")
    return parser.parse_args()


# --- backends ---------------------------------------------------------------------


@dataclasses.dataclass
class Backend:
    """Everything that differs between a cleanba agent and the route model."""

    base_flat: np.ndarray
    unravel: object
    params: object
    logits_fn: object
    edit_arrays: dict
    eval_arrays: dict
    edit_targets: object
    eval_targets: object
    evaluate: object  # (flat, targets, gains | None) -> (reached, steps)
    load_axis: object  # () -> (offsets, diffs)
    onpolicy_fn: object = None


def rl_backend(args) -> Backend:
    import jax.numpy as jnp
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.edits import rl as rl_edit
    from goalmisgen.envs.features import CorrelatedFeatures

    payload = json.loads((args.run / "BASE.json").read_text())
    checkpoint = args.run / payload["checkpoint"]
    values = tuple(payload["values"])

    def config_for(split):
        return MazeConfig(
            max_episode_steps=120, num_envs=1, min_size=11, max_size=11, n_objectives=len(values),
            objective_values=values, feature_value_correlation=1.0, value_encoding="none",
            colour_is_the_only_value_cue=True, level_dataset=args.levels, dataset_split=split, asynchronous=False,
        )

    config = config_for("test")
    policy, _, _, state, _ = load_train_state(checkpoint, env_cfg=config)
    flat, unravel = ravel_pytree(state.params)
    print(f"base   {checkpoint}  ({flat.size:,} parameters), values {values}")

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

    edit_targets = route_targets(levels_of("valid", args.n_edit), size=11)
    eval_targets = route_targets(levels_of("test", args.n_eval), size=11)
    rollout = rl_edit.make_rollout_fn(policy, unravel)

    def evaluate(flat_params, targets, gains=None):
        g = np.ones(4, dtype=np.float32)
        for channel, gain in (gains or {}).items():
            g[channel] = gain
        reached, steps = [], []
        for start in range(0, len(targets), 1024):
            rows = slice(start, start + 1024)
            r, s = rollout(
                jnp.asarray(flat_params, dtype=jnp.float32),
                jnp.asarray(targets.static[rows]),
                jnp.asarray(targets.positions[rows, 0, 0]),
                jnp.asarray(g),
            )
            reached.append(np.asarray(r))
            steps.append(np.asarray(s))
        return np.concatenate(reached), np.concatenate(steps)

    def arrays(targets):
        out = {k: getattr(targets, k) for k in ("static", "positions", "toward_rich", "toward_poor", "role", "gap", "rich_above", "fields")}
        out["start"] = targets.positions[:, 0, 0]
        return out

    def load_axis():
        from scripts.restricted_writes import load_sweep

        return load_sweep(args.run, args.sweep, values, args, config, np.asarray(flat, dtype=np.float64))

    return Backend(
        base_flat=np.asarray(flat, dtype=np.float64), unravel=unravel, params=state.params,
        logits_fn=rl_edit.make_logits_fn(policy, unravel), edit_arrays=arrays(edit_targets), eval_arrays=arrays(eval_targets),
        edit_targets=edit_targets, eval_targets=eval_targets, evaluate=evaluate, load_axis=load_axis,
        onpolicy_fn=rl_edit.make_onpolicy_fn(policy, unravel, args.horizon),
    )


def offline_backend(args) -> Backend:
    from goalmisgen.edits import bc as bc_edit
    from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
    from goalmisgen.offline.decode import replay
    from goalmisgen.offline.demos import DemoSet
    from goalmisgen.offline.fast_decode import greedy_decode_cached

    base = load_base(args.run)
    print(f"base   {args.run.name} @ step {base.step:,}  ({base.flat.size:,} parameters), values {base.values}")
    max_actions = base.model.config.max_actions

    def levels_of(name, count):
        demos = DemoSet.load(args.demos / name, hide_values=base.hide_values)
        out = []
        for index in range(len(demos)):
            level = demos.level(index)
            if usable(level):
                out.append(level)
            if len(out) == count:
                break
        return out, demos

    edit_levels, _ = levels_of("valid.rho100", args.n_edit)
    eval_levels, eval_demos = levels_of("test.rho100", args.n_eval)
    channels = base.model.config.n_channels
    edit_targets = route_targets(edit_levels, size=base.model.config.size, n_channels=channels)
    eval_targets = route_targets(eval_levels, size=base.model.config.size, n_channels=channels)
    eval_level_list = {id(eval_targets): eval_levels, id(edit_targets): edit_levels}
    step_penalty = float(eval_demos.meta["step_penalty"])
    step_limit = int(eval_demos.meta["step_limit"])

    def evaluate(flat_params, targets, gains=None):
        import jax.numpy as jnp

        from goalmisgen.analysis.input_gain import scale_planes
        from goalmisgen.edits.rl import draw_agent

        # draw_agent writes with jnp's .at[], so both arguments have to be JAX
        # arrays: route targets arrive from numpy and would fail here otherwise.
        obs = np.asarray(draw_agent(jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0])))
        if gains:
            obs = scale_planes(obs, gains, channel_axis=-1)
        decoded = greedy_decode_cached(base.model, base.unravel(np.asarray(flat_params, dtype=np.float32)), obs)
        reached, steps = [], []
        for row, level in enumerate(eval_level_list[id(targets)]):
            info = replay(level, decoded.actions[row], step_penalty, step_limit, bool(decoded.emitted_eos[row]))
            reached.append(info["reached_feature_id"] if info["reached_objective"] else -1)
            steps.append(info["episode_steps"])
        return np.array(reached), np.array(steps)

    def load_axis():
        arms = arm_dirs(args.run, args.sweep, args.steps_tag)
        arms = {o: d for o, d in arms.items() if 1e-9 < abs(o) <= args.fit_window}
        diffs = load_diffs(base, arms, dtype=np.float64)
        offsets = np.array(sorted(diffs))
        return offsets, np.stack([diffs[o] for o in offsets])

    return Backend(
        base_flat=base.flat, unravel=base.unravel, params=base.params,
        logits_fn=bc_edit.make_logits_fn(base.model, base.unravel),
        edit_arrays=bc_edit.with_eos(edit_targets, max_actions), eval_arrays=bc_edit.with_eos(eval_targets, max_actions),
        edit_targets=edit_targets, eval_targets=eval_targets, evaluate=evaluate, load_axis=load_axis,
    )


# --- main ------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    print(header())
    print()
    started = time.time()

    from goalmisgen.edits import compare
    from goalmisgen.edits.core import (
        DenseEdit,
        GainEdit,
        LowRankEdit,
        SearchConfig,
        behaviour_loss,
        module_indices,
        search,
        top_k_indices,
        weight_kernels,
    )
    from goalmisgen.edits.rl import logistic_fit, read_outcomes

    backend = offline_backend(args) if args.offline else rl_backend(args)
    flat = backend.base_flat
    groups, order = module_groups(backend.params)
    print(f"levels {len(backend.edit_targets)} to edit on, {len(backend.eval_targets)} held out")

    # 1. the base, on both sets
    reached, _ = backend.evaluate(flat, backend.edit_targets)
    finished = reached >= 0
    tau_base, scale = logistic_fit(
        backend.edit_targets.gap[finished].astype(float),
        (reached[finished] == backend.edit_targets.rich_feature[finished]).astype(float),
    )
    base_eval = read_outcomes(backend.eval_targets, *backend.evaluate(flat, backend.eval_targets))
    print(f"base   threshold {tau_base:.2f} steps, logistic scale {scale:.2f} (edit levels); held out {base_eval.tau:.2f}")

    out: dict = {
        "run": str(args.run), "offline": args.offline, "family": args.family, "modules": args.modules, "rank": args.rank,
        "k": args.k, "rule": args.rule, "shifts": args.shifts, "eval_shifts": args.eval_shifts, "seeds": args.seeds,
        "optim": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "optimizer": args.optimizer, "l2": args.l2,
                  "fork_weight": args.fork_weight, "init_scale": args.init_scale, "target": args.target,
                  "nav_weight": args.nav_weight, "on_policy": args.on_policy, "horizon": args.horizon,
                  "commit_weight": args.commit_weight, "hard": not args.soft_rates, "init_from": args.init_from},
        "tau_base_edit": tau_base, "scale": scale, "base_eval": base_eval.as_row(), "parameters": int(flat.size), "runs": [],
    }

    def family_for(seed_indices=None):
        if args.family == "gain":
            kernel = input_kernel(backend.params)
            return GainEdit(size=flat.size, planes={c: kernel.plane(c) for c in COLOUR}, base_flat=flat)
        if args.family == "rotation":
            from goalmisgen.edits.rotation import RotationEdit, colour_planes, partner_map

            kernel = input_kernel(backend.params)
            return RotationEdit(
                size=flat.size, planes=colour_planes(kernel, COLOUR), partners=partner_map(COLOUR), base_flat=flat,
            )
        if args.family == "lowrank":
            return LowRankEdit(size=flat.size, kernels=weight_kernels(backend.params, args.modules), rank=args.rank)
        indices = seed_indices if seed_indices is not None else (module_indices(backend.params, args.modules) if args.modules else None)
        return DenseEdit(size=flat.size, indices=indices, init_scale=args.init_scale)

    def evaluate_direction(direction, label):
        rows = []
        print(f"\n  {label}")
        print(f"  {'shift':>7}{'target':>8}{'tau':>8}  {'95% interval':^14}{'reached':>9}{'route':>8}{'above':>8}{'below':>8}")
        for shift in sorted(set(args.eval_shifts) | set(args.shifts)):
            r = read_outcomes(backend.eval_targets, *backend.evaluate(flat + shift * direction, backend.eval_targets))
            rows.append({"shift": shift, "trained": shift in args.shifts, **r.as_row()})
            mark = "*" if shift in args.shifts else " "
            print(
                f"  {shift:>+6.1f}{mark}{base_eval.tau + shift:>8.2f}{r.tau:>8.2f}  [{r.low:5.1f}, {r.high:5.1f}]"
                f"{r.reached:>9.1%}{r.on_route:>8.1%}{r.tau_rich_above:>8.2f}{r.tau_rich_below:>8.2f}",
                flush=True,
            )
        return rows

    config = SearchConfig(
        shifts=tuple(args.shifts), tau_base=tau_base, scale=scale, rule=args.rule, steps=args.steps, batch=args.batch,
        lr=args.lr, optimizer=args.optimizer, l2=args.l2, fork_weight=args.fork_weight,
        target=args.target, nav_weight=args.nav_weight, hard=not args.soft_rates,
    )
    # The base's fork margin against the gap: its slope converts a threshold shift into a margin shift.
    import jax
    import jax.numpy as jnp

    from goalmisgen.edits.core import choice_curve, curve_targets, fork_margins, margin_curve, margin_offsets, rule_labels

    margin_of = jax.jit(lambda f, b: fork_margins(backend.logits_fn(f, b), b["toward_rich"], b["toward_poor"], b["role"]))
    margins = np.concatenate([
        np.asarray(margin_of(jnp.asarray(flat, dtype=jnp.float32), {k: jnp.asarray(v[i : i + 64]) for k, v in backend.edit_arrays.items()}))
        for i in range(0, len(backend.edit_targets), 64)
    ])
    gaps = backend.edit_targets.gap.astype(float)
    b, a = np.polyfit(gaps, margins, 1)
    agree = float(np.mean((margins > 0)[finished] == (reached[finished] == backend.edit_targets.rich_feature[finished])))
    print(f"base   fork margin = {a:+.3f} {b:+.3f} x gap (r {np.corrcoef(gaps, margins)[0, 1]:+.3f}); "
          f"margin-implied threshold {-a / b:.2f} vs behavioural {tau_base:.2f}; sign agrees with the rollout choice on {agree:.1%} of levels")
    curve = margin_curve(gaps, margins)
    backend.edit_arrays["margin_offset"] = margin_offsets(curve, gaps, backend.edit_targets.rich_above, args.shifts, args.rule)
    tau_m, rule_sign, rule_weight = rule_labels(gaps, margins, backend.edit_targets.rich_above, args.shifts, args.rule)
    backend.edit_arrays["rule_sign"], backend.edit_arrays["rule_weight"] = rule_sign, rule_weight
    flips = [float(np.mean((rule_sign[:, j] > 0) != (gaps < tau_m))) for j in range(len(args.shifts))]
    print(f"base   fork-sign threshold {tau_m:.1f}; {rule_weight.mean():.1%} of levels agree with it and are trained on; "
          f"doses ask to flip " + ", ".join(f"{100 * f:.1f}%" for f in flips))
    if args.on_policy:
        if backend.onpolicy_fn is None:
            sys.exit("--on-policy is implemented for cleanba agents only")
        # Closed-loop targets come from closed-loop behaviour: the base's own rollout choices.
        rollout_rich = np.where(finished, reached == backend.edit_targets.rich_feature, margins > 0)
        decisions = choice_curve(gaps, rollout_rich)
        config = dataclasses.replace(config, on_policy=True, commit_weight=args.commit_weight)
        committed_of = jax.jit(lambda f, bt: backend.onpolicy_fn(f, bt)["committed"])
        backend.edit_arrays["base_committed"] = np.concatenate([
            np.asarray(committed_of(jnp.asarray(flat, dtype=jnp.float32), {k: jnp.asarray(v[i : i + 256]) for k, v in backend.edit_arrays.items()}))
            for i in range(0, len(backend.edit_targets), 256)
        ])
        print(f"base   commits within {args.horizon} steps on {backend.edit_arrays['base_committed'].mean():.1%} of edit levels")
    else:
        decisions = choice_curve(gaps, margins > 0)
    gap_bin, n_bins, curve_rate = curve_targets(decisions, gaps, backend.edit_targets.rich_above, args.shifts, args.rule)
    backend.edit_arrays["gap_bin"], backend.edit_arrays["curve_rate"] = gap_bin, curve_rate
    grid = np.arange(-12, 13, 3)
    print("base   fork P(rich) by gap: " + "  ".join(f"{g:+d}:{float(decisions(g)):.2f}" for g in grid))
    print("base   mean fork margin by gap: " + "  ".join(f"{g:+d}:{float(curve(g)):+.2f}" for g in grid))
    out["margin_fit"] = {"intercept": float(a), "slope": float(b), "implied_tau": float(-a / b), "sign_agreement": agree, "tau_m": tau_m, "rule_weight": float(rule_weight.mean()), "flips": flips,
                         "curve": {int(g): float(curve(g)) for g in range(-20, 21)}}
    config = dataclasses.replace(config, margin_slope=float(abs(b)), n_bins=n_bins)
    if args.max_norm is not None and args.norm_units == "v" and args.no_axis:
        sys.exit("--max-norm in units of v needs the fine-tuned axis; drop --no-axis or use --norm-units abs")
    deltas = {}
    loss_rows = np.random.default_rng(1).choice(len(backend.edit_targets), size=min(512, len(backend.edit_targets)), replace=False)
    zero = behaviour_loss(np.zeros_like(flat), backend.logits_fn, flat, backend.edit_arrays, config, loss_rows, backend.onpolicy_fn)
    out["zero_edit_loss"] = zero
    print(f"no edit: fixed-level loss {zero['loss']:.4f} (nav {zero['nav']:.4f}, fork {zero['fork']:.4f}, fork error {zero['fork_error']:.3f})")

    v_step = None
    if not args.no_axis:
        offsets, diffs = backend.load_axis()
        axis, _ = fit_axis_and_drift(offsets, diffs)
        reliability = split_half_reliability(offsets, diffs)
        del diffs
        # Put v in steps: measure its threshold slope through the same rollout the edits used.
        probe = [read_outcomes(backend.eval_targets, *backend.evaluate(flat + o * axis, backend.eval_targets)).tau for o in (-0.3, 0.3)]
        slope = (probe[1] - probe[0]) / 0.6
        v_step = axis / slope
        print(f"\nfine-tuned axis {args.sweep}: {len(offsets)} arms, |v| {np.linalg.norm(axis):.4g} per unit value, "
              f"slope {slope:+.2f} steps per unit, |v| {np.linalg.norm(v_step):.4g} per step, split-half reliability {reliability:.3f}")
        v_loss = behaviour_loss(v_step, backend.logits_fn, flat, backend.edit_arrays, config, loss_rows, backend.onpolicy_fn)
        print(f"v per step: fixed-level loss {v_loss['loss']:.4f} (nav {v_loss['nav']:.4f}, fork {v_loss['fork']:.4f}, fork error {v_loss['fork_error']:.3f})")
        if args.loss_only:
            for target in (("curve",) if config.on_policy else ("curve", "sign", "margin", "soft")):
                alt = dataclasses.replace(config, target=target)
                z = behaviour_loss(np.zeros_like(flat), backend.logits_fn, flat, backend.edit_arrays, alt, loss_rows, backend.onpolicy_fn)
                w = behaviour_loss(v_step, backend.logits_fn, flat, backend.edit_arrays, alt, loss_rows, backend.onpolicy_fn)
                print(f"  target {target:<7} no edit {z['loss']:.4f} (fork {z['fork']:.4f}, error {z['fork_error']:.3f})   "
                      f"v {w['loss']:.4f} (nav {w['nav']:.4f}, fork {w['fork']:.4f}, error {w['fork_error']:.3f})")
            return
        v_rows = evaluate_direction(v_step, "fine-tuned axis, written per step")
        out["axis"] = {
            "sweep": args.sweep, "offsets": offsets.tolist(), "norm_per_value": float(np.linalg.norm(axis)), "slope": slope,
            "norm_per_step": float(np.linalg.norm(v_step)), "reliability": reliability, "eval": v_rows, "fixed_loss": v_loss,
        }
    if args.max_norm is not None:
        radius = args.max_norm * (float(np.linalg.norm(v_step)) if args.norm_units == "v" else 1.0)
        config = dataclasses.replace(config, max_norm=radius)
        out["max_norm"] = radius
        print(f"\nbudget: |delta| <= {radius:.4g} per step ({args.max_norm:g} {'x |v|' if args.norm_units == 'v' else 'absolute'})")
    for seed in args.seeds:
        print(f"\n=== search: family {args.family}, seed {seed} ===")
        family = family_for()
        init_phi = None
        if args.init_from:
            path, key = args.init_from.rsplit(":", 1)
            start = np.load(path)[key].astype(np.float32)
            if not isinstance(family, DenseEdit):
                sys.exit("--init-from is for the dense family")
            init_phi = {"u": jnp.asarray(start if family.indices is None else start[family.indices])}
            print(f"  starting from {args.init_from} (|delta| {np.linalg.norm(start):.4g})")
        result = search(family, backend.logits_fn, flat, backend.edit_arrays, dataclasses.replace(config, seed=seed),
                        onpolicy_fn=backend.onpolicy_fn, init_phi=init_phi)
        delta = result.delta
        if args.family == "sparse":
            support = top_k_indices(delta, args.k, within=family.indices)
            print(f"  refitting on the {len(support)} largest coordinates")
            result = search(family_for(support), backend.logits_fn, flat, backend.edit_arrays, dataclasses.replace(config, seed=seed), onpolicy_fn=backend.onpolicy_fn)
            delta = result.delta
        deltas[f"seed{seed}"] = delta
        fixed = behaviour_loss(delta, backend.logits_fn, flat, backend.edit_arrays, config, loss_rows, backend.onpolicy_fn)
        print(f"  fixed-level loss {fixed['loss']:.4f} (nav {fixed['nav']:.4f}, fork {fixed['fork']:.4f}, fork error {fixed['fork_error']:.3f})")
        rows = evaluate_direction(delta, f"seed {seed}: |delta| {np.linalg.norm(delta):.4g} per step, {family.n_parameters():,} edit parameters")
        run = {"seed": seed, "norm": float(np.linalg.norm(delta)), "edit_parameters": family.n_parameters(),
               "history": result.history, "fixed_loss": fixed, "eval": rows}
        if args.family == "gain":
            run["gains_per_step"] = np.asarray(result.phi["g"]).tolist()
        if args.family == "rotation":
            from goalmisgen.edits.rotation import angles, length_drift

            run["turns_per_step"] = np.asarray(result.phi["t"]).tolist()
            run["turn_degrees"] = angles(family, result.phi)
            run["length_drift"] = length_drift(family, result.phi)
            print("  turn per step: " + ", ".join(f"colour {c}: {d:+.2f} deg" for c, d in run["turn_degrees"].items()))
        out["runs"].append(run)

    names = list(deltas)
    seed_cos = [cosine(deltas[a], deltas[b]) for i, a in enumerate(names) for b in names[i + 1 :]]
    out["seed_cosines"] = seed_cos
    if seed_cos:
        print(f"\nseed-to-seed cos(u_i, u_j): {', '.join(f'{c:+.3f}' for c in seed_cos)}")
    mean_delta = np.mean([deltas[n] for n in names], axis=0)

    if v_step is not None:
        weight = {n: cosine(deltas[n], v_step) for n in names}
        weight["mean"] = cosine(mean_delta, v_step)
        print("cos(u, v): " + ", ".join(f"{n} {c:+.3f}" for n, c in weight.items()))

        rows = np.random.default_rng(0).choice(len(backend.eval_targets), size=min(128, len(backend.eval_targets)), replace=False)
        batch = {k: np.asarray(v[rows]) for k, v in backend.eval_arrays.items()}
        tangents = compare.function_tangents(backend.logits_fn, flat, batch, {**deltas, "mean": mean_delta, "v": v_step})
        function = compare.cosine_matrix(tangents)
        from goalmisgen.edits.targets import FORK

        fork_tangents = compare.function_tangents(backend.logits_fn, flat, batch, {**deltas, "mean": mean_delta, "v": v_step}, roles=(FORK,))
        function_fork = compare.cosine_matrix(fork_tangents)
        print("function-space cos at forks only: " + ", ".join(f"{n} {function_fork[n]['v']:+.3f}" for n in [*names, "mean"]))
        print("function-space cos(J u, J v): " + ", ".join(f"{n} {function[n]['v']:+.3f}" for n in [*names, "mean"]))
        print("|J u| / |J v| per step: " + ", ".join(f"{n} {np.linalg.norm(tangents[n]) / np.linalg.norm(tangents['v']):.3f}" for n in [*names, "mean"]))
        modules = compare.per_module(mean_delta, v_step, groups, order)
        print(f"\n  {'module':<20}{'u share':>9}{'v share':>9}{'cos':>8}")
        for m in modules:
            print(f"  {m['module']:<20}{100 * m['u_share']:>8.1f}%{100 * m['v_share']:>8.1f}%{m['cos']:>+8.3f}")
        out["axis"].update(
            weight_cos=weight, function_cos=function, function_cos_fork=function_fork,
            function_norm_ratio={n: float(np.linalg.norm(tangents[n]) / np.linalg.norm(tangents["v"])) for n in [*names, "mean"]},
            modules=modules,
        )
        deltas["v_step"] = v_step

    out["seconds"] = time.time() - started
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    if args.save_deltas:
        args.save_deltas.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_deltas, **{k: v.astype(np.float32) for k, v in deltas.items()})
        print(f"wrote {args.save_deltas}")


if __name__ == "__main__":
    main()
