"""Does a colour gain survive the first normalisation, and is the axis's change a turn or a gain?

    uv run python scripts/direction_profile.py /workspace/data/runs/vit11nvg1.s9012 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --json figures/data/direction/vit11nvg1.s9012.json
    uv run python scripts/direction_profile.py /workspace/data/offline/runs/bcnv11.s1 --offline \
        --demos /workspace/data/offline/demos --json figures/data/direction/bcnv11.s1.json

**Survival** (E7 deliverable 1). A gain scales colour ``c``'s contribution to
the first layer without turning it. A pre-LayerNorm block divides a token by its
own standard deviation, so a change in length is discarded there and only a
change in direction reaches the next block. For each architecture this reports,
at every readable state, the size of the activation change a small gain makes
relative to the state itself, and the ratio between consecutive states. The
prediction is that the DRC and the ResNet -- ``IdentityNorm``, gates rather than
norms -- pass it on, and that the ViT and the route model do not.

For the two normalised networks it also splits the change at the embedding into
the part along the ones vector, the part along the token (both of which a
LayerNorm removes) and the rest, and applies the checkpoint's *own* first
LayerNorm to measure the survival exactly rather than to first order.

**Alignment** (E7 deliverable 3). The same profile for the rotation edit
(:mod:`goalmisgen.edits.rotation`), and then, at every state, the share of the
fine-tuned axis's activation change that the gain's and the turn's changes
explain -- the cross-architecture version of
:mod:`goalmisgen.analysis.gain_alignment`, with a matched-norm random write as
the control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.input_gain import input_kernel  # noqa: E402
from goalmisgen.analysis.norm_survival import decompose, layer_norm, relative_change  # noqa: E402
from goalmisgen.edits.rotation import RotationEdit, colour_planes, partner_map  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--offline", action="store_true", help="The base is a route model, not a cleanba agent.")
    parser.add_argument("--levels", type=str, help="RL: level dataset directory.")
    parser.add_argument("--demos", type=Path, help="Offline: directory holding valid.rho100 and test.rho100.")
    parser.add_argument("--n-levels", type=int, default=256)
    parser.add_argument("--gain", type=float, default=0.02, help="Brightness step, as a fraction: plane c times 1 + gain.")
    parser.add_argument("--turn", type=float, default=0.02, help="Rotation step, in units of the partner's unit slice.")
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--steps-tag", type=int, default=1000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--no-axis", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def profile(states_of, base_flat, deltas: dict[str, np.ndarray], names) -> dict:
    """Per state, each write's relative activation change and how much of the axis's it explains."""
    base_states = states_of(base_flat)
    changed = {label: states_of(base_flat + delta) for label, delta in deltas.items()}
    rows = []
    for index, name in enumerate(names):
        row = {"state": name}
        reference = np.asarray(base_states[index], dtype=np.float64)
        vectors = {}
        for label, states in changed.items():
            after = np.asarray(states[index], dtype=np.float64)
            row[f"rel_{label}"] = relative_change(reference, after)
            vectors[label] = (after - reference).ravel()
        if "axis" in vectors:
            target = vectors["axis"]
            for label in ("gain", "turn", "random"):
                if label in vectors:
                    row[f"r2_{label}"] = _r2(target, [vectors[label]])
            present = [l for l in ("gain", "turn") if l in vectors]
            if len(present) == 2:
                row["r2_both"] = _r2(target, [vectors[l] for l in present])
        rows.append(row)
    return {"states": rows}


def _r2(target: np.ndarray, predictors: list[np.ndarray]) -> float:
    """Share of ``target``'s variance a least-squares combination of ``predictors`` explains."""
    design = np.stack(predictors, axis=1)
    total = float(target @ target)
    if total <= 0:
        return float("nan")
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ solution
    return float(1.0 - (residual @ residual) / total)


def survival_table(rows: list[dict], key: str) -> list[dict]:
    """Consecutive ratios of a relative change: what each stage passed on."""
    out = []
    for before, after in zip(rows, rows[1:]):
        upstream = before.get(key, float("nan"))
        out.append({
            "from": before["state"], "to": after["state"],
            "survival": (after.get(key, float("nan")) / upstream) if upstream and upstream > 0 else float("nan"),
        })
    return out


def main() -> None:
    args = parse_args()
    print(header())
    print()

    import jax
    import jax.numpy as jnp

    if args.offline:
        states_of, base_flat, params, names, load_axis, norm_params = offline_setup(args)
    else:
        states_of, base_flat, params, names, load_axis, norm_params = rl_setup(args)

    kernel = input_kernel(params)
    print(f"input  {kernel.path} {kernel.shape}")
    print(f"states {', '.join(names)}")

    # The two writes, at a small matched-ish dose: a gain (length, no turn) and a rotation (turn, no length).
    gain_delta = np.zeros_like(base_flat)
    for channel in COLOUR:
        rows = kernel.plane(channel)
        gain_delta[rows] = args.gain * base_flat[rows] * (1 if channel == COLOUR[1] else -1)
    family = RotationEdit(size=base_flat.size, planes=colour_planes(kernel, COLOUR), partners=partner_map(COLOUR), base_flat=base_flat)
    turn_delta = np.asarray(family.delta({"t": jnp.array([-args.turn, args.turn])}), dtype=np.float64)

    deltas = {"gain": gain_delta, "turn": turn_delta}
    out = {"run": str(args.run), "offline": args.offline, "states": names, "gain": args.gain, "turn": args.turn,
           "input_kernel": {"path": kernel.path, "shape": list(kernel.shape)}}

    if not args.no_axis:
        axis = load_axis()
        # Match the axis's write to the gain's length, so the profile compares shapes and not sizes.
        scale = np.linalg.norm(gain_delta) / max(np.linalg.norm(axis), 1e-12)
        deltas["axis"] = axis * scale
        rng = np.random.default_rng(args.seed)
        random = rng.normal(size=base_flat.size)
        deltas["random"] = random / np.linalg.norm(random) * np.linalg.norm(gain_delta)
        out["axis_norm"] = float(np.linalg.norm(axis))

    result = profile(states_of, base_flat, deltas, names)
    out.update(result)

    print(f"\n=== relative activation change per state (|dh| / |h|) ===")
    labels = [l for l in ("gain", "turn", "axis", "random") if l in deltas]
    print(f"  {'state':<18}" + "".join(f"{l:>12}" for l in labels))
    for row in result["states"]:
        print(f"  {row['state']:<18}" + "".join(f"{row.get(f'rel_{l}', float('nan')):>12.3e}" for l in labels))

    print(f"\n=== survival: how much of the relative change each stage passed on ===")
    print(f"  {'stage':<40}" + "".join(f"{l:>12}" for l in labels))
    tables = {l: survival_table(result["states"], f"rel_{l}") for l in labels}
    out["survival"] = tables
    for index in range(len(result["states"]) - 1):
        stage = f"{tables[labels[0]][index]['from']} -> {tables[labels[0]][index]['to']}"
        print(f"  {stage:<40}" + "".join(f"{tables[l][index]['survival']:>12.3f}" for l in labels))

    if any(f"r2_gain" in row for row in result["states"]):
        print(f"\n=== share of the axis's change explained (R^2) ===")
        print(f"  {'state':<18}{'gain':>10}{'turn':>10}{'both':>10}{'random':>10}")
        for row in result["states"]:
            print(f"  {row['state']:<18}" + "".join(
                f"{row.get(key, float('nan')):>10.3f}" for key in ("r2_gain", "r2_turn", "r2_both", "r2_random")))

    if norm_params is not None:
        tokens, scale_p, bias_p = norm_params(base_flat)
        print(f"\n=== the first LayerNorm, on the embedding ===")
        for label in ("gain", "turn"):
            perturbed, _, _ = norm_params(base_flat + deltas[label])
            parts = decompose(tokens, perturbed - tokens)
            before = relative_change(tokens, perturbed)
            after = relative_change(layer_norm(tokens, scale_p, bias_p), layer_norm(perturbed, scale_p, bias_p))
            print(f"  {label:<6} ones {parts.ones:6.3f}  magnitude {parts.magnitude:6.3f}  direction {parts.direction:6.3f}"
                  f"   |dh|/|h| before {before:.3e} after {after:.3e}   survival {after / before if before > 0 else float('nan'):.3f}")
            out.setdefault("first_norm", {})[label] = {
                "ones": parts.ones, "magnitude": parts.magnitude, "direction": parts.direction,
                "before": before, "after": after, "survival": after / before if before > 0 else float("nan"),
            }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


def rl_setup(args):
    """States, parameters and axis for a cleanba agent, read through :class:`StateReader`."""
    import jax
    import jax.numpy as jnp
    from cleanba.cleanba_impala import load_train_state
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.edits.rl import draw_agent
    from goalmisgen.edits.targets import route_targets, usable
    from goalmisgen.envs.features import CorrelatedFeatures
    from goalmisgen.nets.readers import state_reader_for

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
    flat = np.asarray(flat, dtype=np.float64)
    print(f"base   {checkpoint}  ({flat.size:,} parameters), values {values}")

    sampler = config_for("test").sampler()
    scheme = CorrelatedFeatures(1.0)
    levels = []
    for index in sampler.indices:
        values_here = tuple(float(v) for v in sampler.dataset.values[index])
        level = sampler.dataset.level(int(index), scheme.assign(values_here, np.random.default_rng(int(index))))
        if usable(level):
            levels.append(level)
        if len(levels) == args.n_levels:
            break
    targets = route_targets(levels, size=11)
    # The agent has to be somewhere: read the state at each level's start cell.
    # cleanba's Policy takes NCHW, as every rollout in goalmisgen.edits.rl does.
    observations = draw_agent(jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0])).transpose(0, 3, 1, 2)

    reader = state_reader_for(policy)
    names = list(reader.layer_names)
    key = jax.random.PRNGKey(0)
    carry = policy.apply(state.params, jax.random.PRNGKey(1), observations.shape, method=policy.initialize_carry)
    starts = jnp.ones(observations.shape[0], dtype=bool)

    def states_of(flat_params):
        params = unravel(jnp.asarray(flat_params, dtype=jnp.float32))
        _, _, _, _, per_cell = reader.step(params, carry, observations, starts, key)
        return [np.asarray(grid) for grid in per_cell.features]

    def load_axis():
        from scripts.restricted_writes import load_sweep

        offsets, diffs = load_sweep(args.run, args.sweep, values, args, config, flat)
        axis, _ = fit_axis_and_drift(offsets, diffs)
        return axis

    # Only the ViT has a normalisation to pass through; the DRC and ResNet run IdentityNorm.
    first_norm = find_first_layer_norm(state.params)
    norm_params = None
    if first_norm is not None:
        def norm_params(flat_params):
            """The embedding grid, and the first block's LayerNorm scale and bias."""
            tree = unravel(jnp.asarray(flat_params, dtype=jnp.float32))
            embedding = states_of(flat_params)[0]
            norm = find_first_layer_norm(tree)
            return embedding, np.asarray(norm["scale"]), np.asarray(norm["bias"])

    return states_of, flat, state.params, names, load_axis, norm_params


def find_first_layer_norm(tree, wanted=("norm_attention", "ln_attention")):
    """The first block's pre-attention LayerNorm parameters, or ``None`` if the network has none.

    The DRC and the ResNet are configured with ``IdentityNorm`` and have no such
    thing; the ViT and the route model normalise every token before the first
    attention, which is the stage a gain has to survive.
    """
    if not hasattr(tree, "items"):
        return None
    # Sorted, so "block_0" is reached before "block_1" whatever order the checkpoint stored.
    for key in sorted(tree, key=str):
        value = tree[key]
        if str(key) in wanted and hasattr(value, "get") and "scale" in value:
            return value
        found = find_first_layer_norm(value, wanted)
        if found is not None:
            return found
    return None


def offline_setup(args):
    """States, parameters and axis for the route model, read off its residual stream."""
    import jax.numpy as jnp

    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.offline.axis import arm_dirs, load_base, load_diffs
    from goalmisgen.offline.demos import DemoSet

    from goalmisgen.edits.rl import draw_agent
    from goalmisgen.edits.targets import route_targets, usable

    base = load_base(args.run)
    config = base.model.config
    print(f"base   {args.run.name} @ step {base.step:,}  ({base.flat.size:,} parameters), values {base.values}")

    demos = DemoSet.load(args.demos / "test.rho100", hide_values=base.hide_values)
    levels = []
    for index in range(len(demos)):
        level = demos.level(index)
        if usable(level):
            levels.append(level)
        if len(levels) == args.n_levels:
            break
    targets = route_targets(levels, size=config.size, n_channels=config.n_channels)
    observations = np.asarray(draw_agent(jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0])))
    # The prefix alone: no actions taken yet, which is where the route is chosen.
    actions = jnp.full((len(levels), config.max_actions), config.pad, dtype=jnp.int32)
    names = ["embedding"] + [f"block {i}" for i in range(config.n_layers)]

    def residuals_of(flat_params):
        # unravel returns the full variables dict, {"params": ...} included, so
        # apply takes it as it is; wrapping it again is an extra params layer.
        tree = base.unravel(jnp.asarray(flat_params, dtype=jnp.float32))
        _, residuals = base.model.apply(tree, jnp.asarray(observations), actions)
        return tree, residuals

    def states_of(flat_params):
        _, residuals = residuals_of(flat_params)
        return [np.asarray(r[:, : config.n_cells], dtype=np.float32) for r in residuals]

    def norm_params(flat_params):
        """The embedding at the maze tokens, and the first block's LayerNorm scale and bias."""
        tree, residuals = residuals_of(flat_params)
        norm = tree["params"]["block_0"]["ln_attention"]
        return np.asarray(residuals[0][:, : config.n_cells]), np.asarray(norm["scale"]), np.asarray(norm["bias"])

    def load_axis():
        arms = arm_dirs(args.run, args.sweep, args.steps_tag)
        arms = {o: d for o, d in arms.items() if 1e-9 < abs(o) <= args.fit_window}
        diffs = load_diffs(base, arms, dtype=np.float64)
        offsets = np.array(sorted(diffs))
        axis, _ = fit_axis_and_drift(offsets, np.stack([diffs[o] for o in offsets]))
        return axis

    return states_of, base.flat, base.params, names, load_axis, norm_params


if __name__ == "__main__":
    main()
