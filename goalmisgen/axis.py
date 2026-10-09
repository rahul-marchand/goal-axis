"""Reading a value-axis sweep of a cleanba agent off the volume: the RL twin of :mod:`goalmisgen.offline.axis`.

Every script that asks something of the value axis does the same five things
before it asks it: resolve the base agent from ``runs/<agent>/BASE.json``, build
an environment to evaluate it in, load its weights flat, load the arms of a sweep
as weight diffs from it, and decode thresholds greedily on held-out levels. Each
script used to do all five by hand, and the copies drifted -- in how many
environments they built, which split they read, how many bootstrap resamples
they drew -- so they live here, with every one of those differences an explicit
argument.

**The evaluation environment comes from the run's own record.**
:func:`eval_config` starts from the ``train_env`` saved in the checkpoint's
``cfg.json`` (:func:`goalmisgen.configs.saved.train_env`) and changes only what
evaluation must change: the dataset and split, the number of environments, the
seed, synchronous stepping. The scripts' literals said ``value_encoding="none"``,
size 11, 120 steps and correlation 1.0 because every base they were pointed at
was trained that way; a base trained otherwise -- with a value channel, or a
different observation -- is now evaluated the way it was trained instead of the
way the literal assumed. ``tests/test_axis.py`` checks that for the saved
configurations of the bases actually swept, the two are the same config.

**Arithmetic is left to the caller, deliberately.** Writing ``base + o * axis``
in float32 on the JAX array (``014``'s style, :attr:`Base.raveled`) and in
float64 before rounding (``scripts/restricted_writes.py``'s style,
:attr:`Base.flat` then :meth:`Base.params_of`) differ in the last place for a
fifth of the coordinates, which is enough to move a greedy rollout now and then.
Both styles exist in recorded results, so this module offers both operands and
picks neither.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state
from cleanba.environments import random_seed
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis.behaviour import (
    collect_episode_outcomes,
    indifference_point,
    summarise,
    value_distance_decisions,
)
from goalmisgen.analysis.metrics import bootstrap_episodes
from goalmisgen.configs import saved
from goalmisgen.configs.env import MazeConfig
from goalmisgen.volume import BaseMarker, discover_arms, read_base_marker, sweep_family, sweep_index

# --- the evaluation environment ---------------------------------------------------


def eval_config(
    train_env: MazeConfig,
    levels: str | None,
    *,
    split: str = "test",
    num_envs: int = 64,
    seed: int | None = 0,
    values: tuple[float, ...] | None = None,
    max_episode_steps: int | None = None,
    size: int | None = None,
    colour_keyed: bool | None = None,
) -> MazeConfig:
    """The agent's own training environment, pointed at held-out levels.

    ``values`` overrides what the objectives pay (and so their count); the
    default keeps the training values, which is what every readout at the base
    values wants. ``max_episode_steps`` and ``size`` likewise default to the
    training run's. ``seed=None`` draws one at random, as cleanba does for a
    config that sets none -- what the scripts whose literal never set one got.
    ``colour_keyed`` overrides how colours attach to objectives; the default is
    the run's own, which differs from value-rank colouring only for a ``k``
    arm's checkpoint read directly. ``levels=None`` samples live, which
    only matters to callers that build the environment to load weights and
    never step it.
    """
    overrides: dict[str, Any] = dict(
        num_envs=num_envs,
        level_dataset=levels,
        dataset_split=split,
        asynchronous=False,
    )
    # cleanba draws an unset seed at random, which is what a literal without one got.
    overrides["seed"] = seed if seed is not None else random_seed()
    if values is not None:
        overrides.update(objective_values=tuple(values), n_objectives=len(values))
    if max_episode_steps is not None:
        overrides["max_episode_steps"] = max_episode_steps
    if size is not None:
        overrides.update(min_size=size, max_size=size)
    if colour_keyed is not None:
        overrides["colour_keyed_features"] = colour_keyed
    return dataclasses.replace(train_env, **overrides)


def checkpoint_eval_config(checkpoint: Path, levels: str | None, **kwargs) -> MazeConfig:
    """:func:`eval_config` from the ``cfg.json`` beside ``checkpoint``; refuses a checkpoint with none."""
    train_env = saved.train_env(checkpoint)
    if train_env is None:
        raise FileNotFoundError(
            f"{checkpoint} has no readable cfg.json with a train_env, so there is no record of the environment "
            "it was trained in to evaluate it in"
        )
    return eval_config(train_env, levels, **kwargs)


# --- the base ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Base:
    """A base agent's weights, loaded against the environment it will be evaluated in."""

    checkpoint: Path
    values: tuple[float, ...]
    config: MazeConfig
    policy: Any
    params: Any
    raveled: Any
    """``ravel_pytree(params)[0]``: the float32 JAX vector, for writes done in float32."""
    unravel: Callable
    update: int
    run_dir: Path | None = None
    """The agent directory holding ``BASE.json`` and ``arms/``, when loaded through one."""
    marker: BaseMarker | None = None
    """That directory's ``BASE.json``, when loaded through one."""

    @property
    def flat(self) -> np.ndarray:
        """The weights as float64 numpy, for writes done in float64 and rounded once."""
        return np.asarray(self.raveled, dtype=np.float64)

    @property
    def arms(self) -> Path:
        if self.run_dir is None:
            raise ValueError(f"{self.checkpoint} was loaded without its agent directory, so it has no arms/")
        return self.run_dir / "arms"

    def params_of(self, flat: np.ndarray):
        """Parameters from a flat vector of any width, rounded to float32 once."""
        return self.unravel(np.asarray(flat, dtype=np.float32))

    def other_params(self, checkpoint: Path):
        """Another checkpoint's parameters (an arm's), built against this base's environment."""
        return load_params(checkpoint, self.config)


def load_checkpoint(
    checkpoint: Path, config: MazeConfig, values: tuple[float, ...] | None = None, marker: BaseMarker | None = None
) -> Base:
    """A checkpoint loaded against ``config``; ``values`` default to ``config``'s."""
    policy, _, _, state, update = load_train_state(Path(checkpoint), env_cfg=config)
    raveled, unravel = ravel_pytree(state.params)
    return Base(
        checkpoint=Path(checkpoint),
        values=tuple(values if values is not None else config.objective_values),
        config=config,
        policy=policy,
        params=state.params,
        raveled=raveled,
        unravel=unravel,
        update=int(update),
        run_dir=marker.run_dir if marker is not None else None,
        marker=marker,
    )


def load_base(run_dir: Path, levels: str | None = None, **eval_kwargs) -> Base:
    """The agent ``run_dir/BASE.json`` names, evaluated per :func:`eval_config`'s keywords.

    The objectives' values are ``BASE.json``'s, which is what every script that
    read the marker by hand passed to its literal.
    """
    marker = read_base_marker(run_dir)
    eval_kwargs.setdefault("values", marker.values)
    config = checkpoint_eval_config(marker.checkpoint, levels, **eval_kwargs)
    return load_checkpoint(marker.checkpoint, config, values=marker.values, marker=marker)


def load_params(checkpoint: Path, config: MazeConfig):
    """A checkpoint's parameters, built against ``config``."""
    _, _, _, state, _ = load_train_state(Path(checkpoint), env_cfg=config)
    return state.params


def load_flat(checkpoint: Path, config: MazeConfig):
    """A checkpoint's parameters as one float32 JAX vector."""
    flat, _ = ravel_pytree(load_params(checkpoint, config))
    return flat


def diff_from(base: Base, checkpoint: Path, *, subtract_in: str = "float32") -> np.ndarray:
    """``theta_arm - theta_base`` as float64.

    ``subtract_in`` is where the subtraction happens, and the two are not the
    same numbers: ``"float32"`` subtracts the JAX vectors and widens the result
    (014, restricted_writes and everything built on them); ``"float64"`` widens
    both first and subtracts exactly (far_axis_fit, 022, the arm-length ladder).
    They differ wherever two nearby weights are not within a factor of two of
    each other -- near zero, mostly -- so each caller keeps the one it was run with.
    """
    if subtract_in == "float32":
        return np.asarray(load_flat(checkpoint, base.config) - base.raveled, dtype=np.float64)
    if subtract_in == "float64":
        return np.asarray(load_flat(checkpoint, base.config), dtype=np.float64) - base.flat
    raise ValueError(f"subtract_in is 'float32' or 'float64', not {subtract_in!r}")


# --- arms ---------------------------------------------------------------------------


def arm_checkpoints(
    arms: Path,
    sweep: str,
    base_value: float,
    *,
    steps: int | None = None,
    at: int = -1,
    replicate: int = 1,
) -> dict[float, Path]:
    """One checkpoint of every arm of ``sweep``, keyed by the value it trained at.

    ``sweep`` is anything :func:`goalmisgen.volume.sweep_index` reads -- ``o1``,
    ``k0``, or the legacy ``v``/``c`` -- and its family keeps held-out arms out
    unless asked for by name.
    """
    return discover_arms(
        Path(arms), sweep_index(sweep), base_value, steps=steps, at=at, family=sweep_family(sweep), replicate=replicate
    )


class TooFewArms(ValueError):
    """A sweep has fewer than three arms inside the fit window, so no axis can be fitted from it."""


def load_sweep(base: Base, sweep: str, *, arm_steps: int | None, fit_window: float) -> tuple[np.ndarray, np.ndarray]:
    """Offsets and stacked float64 diffs of one sweep's fitted arms, ``0 < |offset| <= fit_window``.

    The fitted family only (``o``), the last checkpoint of each arm. Offsets are
    ``value - base_value`` as floats, not the names' hundredths, as the copies
    this replaces computed them.
    """
    objective = sweep_index(sweep)
    base_value = base.values[objective]
    found = discover_arms(base.arms, objective=objective, base_value=base_value, steps=arm_steps)
    offsets, diffs = [], []
    for value, checkpoint in sorted(found.items()):
        offset = value - base_value
        if abs(offset) < 1e-9 or abs(offset) > fit_window:
            continue
        offsets.append(offset)
        diffs.append(diff_from(base, checkpoint))
    if len(offsets) < 3:
        raise TooFewArms(f"sweep {sweep}: {len(offsets)} arms inside |offset| <= {fit_window}, need three")
    return np.array(offsets), np.stack(diffs)


def has_sweep(base: Base, sweep: str, *, arm_steps: int | None) -> bool:
    """Whether any fitted arm of ``sweep`` exists at that length."""
    objective = sweep_index(sweep)
    return bool(discover_arms(base.arms, objective, base.values[objective], steps=arm_steps))


def cached_axis(base: Base, sweep: str, arm_steps: int, cache: Path) -> np.ndarray:
    """The axis of every fitted arm of ``sweep`` (the null arm aside), cached as an npz.

    The basin figure's axis: no fit window, float32 diffs, and an ``.npz``
    holding ``axis``, ``drift``, ``arm_values``, ``base_value``, ``objective``
    and ``arm_steps`` that ``scripts/written_scurves.py`` reads back. A cache
    that exists is returned as is.
    """
    from goalmisgen.analysis.weights import fit_axis_and_drift

    cache = Path(cache)
    if cache.exists():
        payload = np.load(cache)
        print(f"axis: cached ({cache.name}, |axis| {np.linalg.norm(payload['axis']):.1f})")
        return payload["axis"]

    objective = sweep_index(sweep)
    base_value = base.values[objective]
    arms = discover_arms(base.arms, objective=objective, base_value=base_value, steps=arm_steps)
    print(f"axis: fitting from {len(arms)} {sweep}@{arm_steps} arms")
    diffs: dict[float, np.ndarray] = {}
    for value, checkpoint in sorted(arms.items()):
        diffs[value] = diff_from(base, checkpoint)
        print(f"  v={value:.2f}  |diff| {np.linalg.norm(diffs[value]):.1f}")
    fitted = {v: d for v, d in diffs.items() if abs(v - base_value) > 1e-9}  # the null arm is the drift measurement
    arm_values = sorted(fitted)
    offsets = np.array(arm_values) - base_value
    axis, drift = fit_axis_and_drift(offsets, np.stack([fitted[v] for v in arm_values]))
    print(f"axis: |axis| {np.linalg.norm(axis):.1f} per unit value, |drift| {np.linalg.norm(drift):.1f}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        axis=axis.astype(np.float32),
        drift=drift.astype(np.float32),
        arm_values=np.array(arm_values),
        base_value=base_value,
        objective=objective,
        arm_steps=arm_steps,
    )
    return axis.astype(np.float32)


# --- greedy rollouts and the threshold ----------------------------------------------


@dataclasses.dataclass(frozen=True)
class Threshold:
    """A decoded exchange rate: the indifference gap in steps, its interval, and whether episodes finish."""

    point: float
    low: float
    high: float
    reached: float
    outcomes: list = dataclasses.field(repr=False, compare=False)

    def as_row(self) -> dict[str, float]:
        return {"steps": float(self.point), "low": float(self.low), "high": float(self.high), "reached": float(self.reached)}

    def line(self, label: str, width: int = 30) -> str:
        """``014``'s table row: label, threshold, 95% interval, reach -- the format its parsers read."""
        return f"  {label:>{width}}{self.point:>8.1f}  [{self.low:5.1f}, {self.high:5.1f}]{self.reached:>11.1%}"


def read_threshold(outcomes: list[dict], *, resamples: int = 200, seed: int = 0) -> Threshold:
    """The logistic indifference point of ``outcomes``, with an episode bootstrap.

    ``resamples=0`` skips the bootstrap and reports a NaN interval.
    """
    gaps, took_richer, _ = value_distance_decisions(outcomes)
    point = indifference_point(gaps, took_richer)
    if resamples:
        low, high = bootstrap_episodes(
            lambda rows: indifference_point(gaps[rows], took_richer[rows]),
            np.arange(len(gaps)),
            resamples=resamples,
            seed=seed,
        )
    else:
        low, high = float("nan"), float("nan")
    return Threshold(point, low, high, summarise(outcomes).reached_objective, outcomes)


@dataclasses.dataclass
class Greedy:
    """Greedy rollouts of one policy in one vector environment, compiled once.

    Parameters are an argument of every call rather than a constant, so every
    arm and every written vector shares one compilation of ``get_action``.
    """

    policy: Any
    envs: Any
    get_action: Callable

    @classmethod
    def build(cls, policy, config: MazeConfig) -> "Greedy":
        return cls(
            policy, config.make(), jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
        )

    @classmethod
    def of(cls, base: Base) -> "Greedy":
        return cls.build(base.policy, base.config)

    def act(
        self,
        params,
        seed: int,
        gains: dict[int, float] | None = None,
        transform: Callable | None = None,
    ):
        """An ``act(observations, starts)`` for :func:`collect_episode_outcomes`, holding its own carry.

        ``transform`` rewrites each batch of observations (cleanba's NCHW) before
        the network sees it -- a repainting, say -- and ``gains`` then scale
        input planes. Both leave the environment, and so the levels and the
        outcomes' ground truth, untouched.
        """
        from goalmisgen.analysis.input_gain import scale_planes

        policy, get_action = self.policy, self.get_action
        carry = policy.apply(
            params, jax.random.PRNGKey(seed), self.envs.observation_space.shape, method=policy.initialize_carry
        )
        state = {"carry": carry, "key": jax.random.PRNGKey(seed)}

        def act(observations, starts):
            if transform is not None:
                observations = transform(observations)
            if gains:
                observations = scale_planes(observations, gains, channel_axis=1)
            state["carry"], action, _, state["key"] = get_action(
                params, state["carry"], observations, starts, state["key"], temperature=0.0
            )
            return np.asarray(action)

        return act

    def outcomes(
        self, params, episodes: int, seed: int, gains: dict[int, float] | None = None, transform: Callable | None = None
    ) -> list[dict]:
        """Episode infos from ``episodes`` greedy episodes, levels drawn from ``seed``."""
        return collect_episode_outcomes(self.envs, self.act(params, seed, gains, transform), episodes, seed=seed)

    def threshold(
        self,
        params,
        episodes: int,
        seed: int,
        *,
        gains: dict[int, float] | None = None,
        transform: Callable | None = None,
        resamples: int = 200,
        bootstrap_seed: int | None = None,
    ) -> Threshold:
        """:func:`read_threshold` of :meth:`outcomes`; the bootstrap is seeded with ``seed`` unless told otherwise."""
        return read_threshold(
            self.outcomes(params, episodes, seed, gains, transform),
            resamples=resamples,
            seed=seed if bootstrap_seed is None else bootstrap_seed,
        )


# --- route levels -------------------------------------------------------------------


def route_levels(config: MazeConfig, split: str, count: int) -> list:
    """The first ``count`` usable levels of ``split``, coloured by value rank at rho = 1.

    For the route-target analyses, which roll out on stored levels rather than
    through the vector environment. Colours are assigned with
    ``CorrelatedFeatures(1.0)`` seeded by level index, whatever ``config``'s
    own feature scheme, as every copy of this did.
    """
    from goalmisgen.analysis.route_targets import usable
    from goalmisgen.envs.features import CorrelatedFeatures

    sampler = dataclasses.replace(config, dataset_split=split).sampler()
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


__all__ = [
    "Base",
    "BaseMarker",
    "Greedy",
    "Threshold",
    "TooFewArms",
    "arm_checkpoints",
    "cached_axis",
    "checkpoint_eval_config",
    "diff_from",
    "eval_config",
    "has_sweep",
    "load_base",
    "load_checkpoint",
    "load_flat",
    "load_params",
    "load_sweep",
    "read_base_marker",
    "read_threshold",
    "route_levels",
]
