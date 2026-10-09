"""One interface over a cleanba agent and a route model, for the edit search and its diagnostics.

The cheapest-edit search asks the same questions of both architectures -- logits
on a batch of route steps, a greedy closed-loop readout, the fine-tuned axis to
compare with -- but every one of them is computed differently. :class:`Backend`
is the answer to each, and :func:`rl_backend` / :func:`offline_backend` build it.

These lived in ``scripts/cheapest_edit.py`` and took its argparse namespace;
``edit_compare``, ``edit_overfit_check`` and ``fork_margin_diagnostic``
rewrote ``sys.argv`` and called cheapest_edit's parser to get one. The
arguments are now explicit, with cheapest_edit's defaults.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


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


ROUTE_ARRAYS = ("static", "positions", "toward_rich", "toward_poor", "role", "gap", "rich_above", "fields")


def rl_backend(
    run: Path,
    levels: str,
    *,
    n_edit: int = 4096,
    n_eval: int = 2048,
    sweep: str = "o1",
    arm_steps: int = 400_000,
    fit_window: float = 0.45,
    horizon: int = 40,
) -> Backend:
    """A cleanba agent ``run`` (its ``BASE.json``), edited on ``valid`` levels and read on ``test``."""
    import jax.numpy as jnp

    from goalmisgen import axis as rl_axis
    from goalmisgen.analysis import closed_loop
    from goalmisgen.analysis.route_targets import route_targets
    from goalmisgen.edits import rl as rl_edit

    # num_envs=1 and no seed: the environment is built only to load the weights;
    # every readout here rolls out inside JAX on stored levels.
    base = rl_axis.load_base(Path(run), levels, split="test", num_envs=1, seed=None)
    print(f"base   {base.checkpoint}  ({base.raveled.size:,} parameters), values {base.values}")

    size = base.config.max_size
    edit_targets = route_targets(rl_axis.route_levels(base.config, "valid", n_edit), size=size)
    eval_targets = route_targets(rl_axis.route_levels(base.config, "test", n_eval), size=size)
    rollout = closed_loop.make_rollout_fn(base.policy, base.unravel)

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
        out = {k: getattr(targets, k) for k in ROUTE_ARRAYS}
        out["start"] = targets.positions[:, 0, 0]
        return out

    def load_axis():
        return rl_axis.load_sweep(base, sweep, arm_steps=arm_steps, fit_window=fit_window)

    return Backend(
        base_flat=base.flat,
        unravel=base.unravel,
        params=base.params,
        logits_fn=rl_edit.make_logits_fn(base.policy, base.unravel),
        edit_arrays=arrays(edit_targets),
        eval_arrays=arrays(eval_targets),
        edit_targets=edit_targets,
        eval_targets=eval_targets,
        evaluate=evaluate,
        load_axis=load_axis,
        onpolicy_fn=rl_edit.make_onpolicy_fn(base.policy, base.unravel, horizon),
    )


def offline_backend(
    run: Path,
    demos: Path,
    *,
    n_edit: int = 4096,
    n_eval: int = 2048,
    sweep: str = "o1",
    steps_tag: int = 1000,
    fit_window: float = 0.45,
) -> Backend:
    """A route model ``run``, edited on ``demos/valid.rho100`` and read on ``demos/test.rho100``."""
    from goalmisgen.analysis.route_targets import route_targets, usable
    from goalmisgen.edits import bc as bc_edit
    from goalmisgen.offline.axis import load_base, load_sweep
    from goalmisgen.offline.decode import replay
    from goalmisgen.offline.demos import DemoSet
    from goalmisgen.offline.fast_decode import greedy_decode_cached

    run, demos = Path(run), Path(demos)
    base = load_base(run)
    print(f"base   {run.name} @ step {base.step:,}  ({base.flat.size:,} parameters), values {base.values}")
    max_actions = base.model.config.max_actions

    def levels_of(name, count):
        loaded = DemoSet.load(demos / name, hide_values=base.hide_values)
        out = []
        for index in range(len(loaded)):
            level = loaded.level(index)
            if usable(level):
                out.append(level)
            if len(out) == count:
                break
        return out, loaded

    edit_levels, _ = levels_of("valid.rho100", n_edit)
    eval_levels, eval_demos = levels_of("test.rho100", n_eval)
    channels = base.model.config.n_channels
    edit_targets = route_targets(edit_levels, size=base.model.config.size, n_channels=channels)
    eval_targets = route_targets(eval_levels, size=base.model.config.size, n_channels=channels)
    eval_level_list = {id(eval_targets): eval_levels, id(edit_targets): edit_levels}
    step_penalty = float(eval_demos.meta["step_penalty"])
    step_limit = int(eval_demos.meta["step_limit"])

    def evaluate(flat_params, targets, gains=None):
        import jax.numpy as jnp

        from goalmisgen.analysis.closed_loop import draw_agent
        from goalmisgen.analysis.input_gain import scale_planes

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
        return load_sweep(base, sweep, steps_tag, fit_window=fit_window)

    return Backend(
        base_flat=base.flat,
        unravel=base.unravel,
        params=base.params,
        logits_fn=bc_edit.make_logits_fn(base.model, base.unravel),
        edit_arrays=bc_edit.with_eos(edit_targets, max_actions),
        eval_arrays=bc_edit.with_eos(eval_targets, max_actions),
        edit_targets=edit_targets,
        eval_targets=eval_targets,
        evaluate=evaluate,
        load_axis=load_axis,
    )
