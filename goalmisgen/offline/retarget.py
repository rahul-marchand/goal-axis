"""The retargeting cost on mazes and route models: the maze goal family and the route model's scorer.

The metric itself is task-agnostic and lives in :mod:`goalmisgen.retarget`. This
module supplies its two task-specific parts:

- :class:`ExchangeRates`, the goal family: a goal is the expert's utility at an
  exchange rate ``tau = (v0 - v1) / step_penalty``, and :func:`retargeted`
  demonstrates the same levels under it;
- :class:`RouteScorer`: bits per route (the sum of ``-log2 p`` over a route's
  moves and its EOS, the same targets the trainer's cross-entropy reads), units
  are tokens.

**L* is zero here, and that is a property of the mazes, not of the code.** The
mazes are perfect, so every objective has one shortest path and the expert's
moves are a deterministic function of the level. Levels where the two
objectives tie exactly are the one exception, and :meth:`ExchangeRates.deterministic`
drops them for every arm alike (:func:`untied`), so the surplus description
length is the MDL itself.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.offline.demos import NO_ACTION, DemoSet, expert_route
from goalmisgen.offline.fast_decode import sample_cached
from goalmisgen.offline.model import RoutePrefixLM, cross_entropy, targets_from_routes
from goalmisgen.offline.train import initial_params, list_checkpoints, load_checkpoint, load_run_config

LN2 = float(np.log(2.0))

NULL_TAU = 10.0
"""The base's own exchange rate at values (1.0, 0.5)."""

PILOT_TAUS = (0.5, 2.5, 6.5, 8.5, 10.0, 12.5, 14.5, 18.5)
"""The pilot's goals. Half-integers never tie (distances are integers), so only the null's ties are dropped."""


def tau_values(tau: float, step_penalty: float = 0.05, richer: float = 1.0) -> tuple[float, float]:
    """The values (richer, poorer) whose exchange rate is ``tau`` steps."""
    return richer, richer - step_penalty * tau


def retargeted(demos: DemoSet, values: tuple[float, float]) -> DemoSet:
    """The same levels and colours, demonstrated by the expert at other values.

    The objective that was richer at the base values gets ``values[0]``, the
    other ``values[1]``. Colours are left alone, so this is only the right
    demonstration while ``values[0] > values[1]``: past parity the correlated
    scheme would have repainted colour 0 onto the other objective, and an arm
    there is a different task (see ``envs/colour_keyed.py``).
    """
    richer, poorer = values
    if not richer > poorer:
        raise ValueError(f"values {values} cross parity; colours would need repainting")
    old = np.asarray(demos.values)
    new = np.where(old == old.max(axis=1, keepdims=True), richer, poorer).astype(np.float64)
    shifted = demos.with_values(new)
    step_penalty = float(demos.meta["step_penalty"])
    step_limit = int(demos.meta["step_limit"])

    actions = np.full_like(np.asarray(demos.actions), -1)
    lengths = np.empty(len(demos), dtype=np.int16)
    target = np.empty(len(demos), dtype=np.int8)
    ambiguous = np.empty(len(demos), dtype=np.bool_)
    margin = np.empty(len(demos), dtype=np.float32)
    for index in range(len(demos)):
        goal, moves, solution = expert_route(shifted.level(index), step_penalty, step_limit)
        if len(moves) > actions.shape[1]:
            raise ValueError(f"level {index} needs {len(moves)} moves; the store holds {actions.shape[1]}")
        actions[index, : len(moves)] = moves
        lengths[index] = len(moves)
        target[index] = goal
        ambiguous[index] = solution.is_ambiguous
        margin[index] = solution.utility_margin if np.isfinite(solution.utility_margin) else np.inf
    return dataclasses.replace(
        shifted,
        actions=actions,
        lengths=lengths,
        target=target,
        ambiguous=ambiguous,
        utility_margin=margin,
        meta={**demos.meta, "values": [richer, poorer], "retargeted_from": demos.meta.get("values")},
    )


def untied(demos: DemoSet, taus: Sequence[float]) -> np.ndarray:
    """Indices of levels on which no exchange rate in ``taus`` leaves the expert indifferent.

    Distances are integers, so a tie needs ``d_richer - d_poorer == tau``
    exactly; half-integer rates never tie and the base's integer one can.
    """
    distances = np.asarray(demos.distances, dtype=np.float64)
    richer = np.argmax(np.asarray(demos.values), axis=1)
    rows = np.arange(len(demos))
    gap = distances[rows, richer] - distances[rows, 1 - richer]
    reachable = (distances >= 0).all(axis=1)
    tied = np.zeros(len(demos), dtype=bool)
    for tau in taus:
        tied |= np.isclose(gap, tau)
    return np.flatnonzero(reachable & ~tied)


@dataclasses.dataclass(frozen=True)
class ExchangeRates:
    """Two-objective mazes; a goal is an exchange rate ``tau`` in steps. The pool is demonstrated at ``null``."""

    null: float = NULL_TAU
    step_penalty: float = 0.05

    def check_pool(self, pool: DemoSet) -> None:
        if not np.isclose(float(pool.meta["step_penalty"]), self.step_penalty):
            raise ValueError(f"the pool's step penalty is {pool.meta['step_penalty']}, not {self.step_penalty}")
        if not np.allclose(pool.meta["values"], tau_values(self.null, self.step_penalty)):
            raise ValueError(f"the pool is demonstrated at {pool.meta['values']}, not the null goal's values")

    def deterministic(self, pool: DemoSet, goals: Sequence[float]) -> np.ndarray:
        return untied(pool, sorted(set(goals)))

    def demonstrations(self, pool: DemoSet, rows: np.ndarray, goal: float) -> DemoSet:
        return retargeted(pool.subset(rows), tau_values(goal, self.step_penalty))

    def changed(self, pool: DemoSet, rows: np.ndarray, data: DemoSet) -> np.ndarray:
        return np.asarray(data.target) != np.asarray(pool.target[rows])


@dataclasses.dataclass
class SampledRoutes:
    """Routes drawn from a model on given mazes: data a :class:`RouteScorer` reads by position."""

    _observations: np.ndarray
    actions: np.ndarray
    lengths: np.ndarray
    truncated: np.ndarray

    def __len__(self) -> int:
        return len(self.lengths)

    def observations(self, indices) -> np.ndarray:
        return self._observations[np.asarray(indices)]

    def routes(self, indices) -> np.ndarray:
        return self.actions[np.asarray(indices)]


class RouteScorer:
    """Bits per route of a :class:`RoutePrefixLM`, at its base goal; units are tokens (moves plus EOS).

    Also a :class:`~goalmisgen.retarget.scoring.Sampler`: routes are drawn token
    by token with the cached decoder, keyed by the maze's row in the data.
    """

    def __init__(self, model: RoutePrefixLM) -> None:
        self.model = model
        eos = model.config.eos

        def loss(params, observations, actions, lengths):
            logits, _ = model.apply(params, observations, actions)
            return cross_entropy(logits, targets_from_routes(actions, lengths, eos))

        @jax.jit
        def bits(params, observations, actions, lengths):
            targets = targets_from_routes(actions, lengths, eos)
            logits, _ = model.apply(params, observations, actions)
            log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            picked = jnp.take_along_axis(log_probs, jnp.maximum(targets, 0)[..., None], axis=-1)[..., 0]
            return -(picked * (targets >= 0)).sum(axis=-1) / LN2

        self.loss = loss
        self.bits = bits

    def batch(self, data: DemoSet, indices: np.ndarray) -> tuple:
        return (
            jnp.asarray(data.observations(indices)),
            jnp.asarray(data.routes(indices)),
            jnp.asarray(data.lengths[indices], dtype=jnp.int32),
        )

    def units(self, data: DemoSet, indices: np.ndarray) -> np.ndarray:
        return np.asarray(data.lengths[indices]).astype(np.int64) + 1

    def sample(self, params, data: DemoSet, indices: np.ndarray, seed: int) -> SampledRoutes:
        indices = np.asarray(indices)
        observations = np.asarray(data.observations(indices))
        drawn = sample_cached(self.model, params, observations, seed, rows=indices)
        actions = np.where(np.arange(drawn.actions.shape[1])[None, :] < drawn.lengths[:, None], drawn.actions, NO_ACTION)
        return SampledRoutes(observations, actions.astype(np.int32), drawn.lengths.astype(np.int16), ~drawn.emitted_eos)


@dataclasses.dataclass
class Agent:
    model: RoutePrefixLM
    params: dict
    hide_values: bool
    provenance: dict


def load_agent(spec: str | pathlib.Path) -> Agent:
    """A route model by run directory, at its final checkpoint; ``random:<run dir>`` is a fresh init of its architecture.

    The fresh init is the "learn everything" anchor, and uses key 0.
    """
    spec = str(spec)
    random = spec.startswith("random:")
    run_dir = pathlib.Path(spec.removeprefix("random:"))
    step, checkpoint = list_checkpoints(run_dir)[-1]
    model, params = load_checkpoint(checkpoint)
    if random:
        params = initial_params(model, jax.random.PRNGKey(0))
    hide_values = bool(load_run_config(run_dir)["demos"].get("hide_values", False))
    provenance = {"run": str(run_dir), "checkpoint_step": None if random else step, "init_key": 0 if random else None}
    return Agent(model, params, hide_values, provenance)
