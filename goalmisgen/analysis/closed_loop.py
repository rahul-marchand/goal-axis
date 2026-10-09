"""Greedy closed-loop rollouts of a cleanba policy, run entirely inside JAX, and what they read out.

A rollout plays a policy on a :class:`~goalmisgen.analysis.route_targets.RouteTargets`
batch without the gymnasium vector environment, so a network -- base, edited or
patched -- can be measured on thousands of held-out levels, every condition on
the *same* levels. It reproduces ``MazeEnv``'s rules -- a move into a wall costs
a step and goes nowhere, reaching either objective ends the episode, the step
limit truncates -- and ``tests/test_edits.py`` checks it against the
environment's own replay.

Observations are built as the encoder builds them (``static`` plus a one-hot
agent) and handed over NCHW, as cleanba's vector-env wrapper would.
:func:`read_outcomes` turns the outcomes into a threshold and its competence
checks.

This is shared machinery: the cheapest-edit search in :mod:`goalmisgen.edits`
reads its edits through it, and the analyses (gain alignment, leftover
patching, field decompositions) build their own rollouts from
:func:`draw_agent` and :data:`MOVE_ARRAY`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.behaviour import indifference_point
from goalmisgen.analysis.metrics import bootstrap_episodes
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL
from goalmisgen.envs.solver import MOVES

MOVE_ARRAY = jnp.asarray(MOVES, dtype=jnp.int32)


def draw_agent(static, positions):
    """``static`` ``(..., S, S, C)`` with the agent drawn at ``positions`` ``(..., 2)``."""
    size = static.shape[-2]
    agent = jax.nn.one_hot(positions[..., 0] * size + positions[..., 1], size * size, dtype=static.dtype)
    return static.at[..., AGENT_CHANNEL].set(agent.reshape(*positions.shape[:-1], size, size))


def make_rollout_fn(policy, unravel, step_limit: int = 120):
    """``(flat, static, start, gains) -> (reached_feature, steps)``, greedy, closed loop.

    ``gains`` ``(C,)`` multiplies the observation channel-wise before the network
    sees it -- ones for an untouched input, the brightness knob otherwise.
    ``reached_feature`` is -1 on timeout.
    """

    @jax.jit
    def rollout(flat, static, start, gains):
        params = unravel(flat)
        batch, size, _, channels = static.shape
        wall = static[..., WALL_CHANNEL] > 0.5
        colour = static[..., FIRST_FEATURE_CHANNEL : FIRST_FEATURE_CHANNEL + 2] > 0.5
        rows = jnp.arange(batch)
        carry = policy.apply(params, jax.random.PRNGKey(0), (batch, channels, size, size), method=policy.initialize_carry)

        def body(state, _):
            carry, position, done, steps, reached, first = state
            obs = (draw_agent(static, position) * gains).transpose(0, 3, 1, 2)
            carry, action, _, _ = policy.apply(
                params, carry, obs, first, jax.random.PRNGKey(0), temperature=0.0, method=policy.get_action
            )
            candidate = jnp.clip(position + MOVE_ARRAY[action], 0, size - 1)
            outside = jnp.any(position + MOVE_ARRAY[action] != candidate, axis=-1)
            blocked = outside | wall[rows, candidate[:, 0], candidate[:, 1]]
            position = jnp.where((done | blocked)[:, None], position, candidate)
            steps = steps + (~done).astype(jnp.int32)
            hit = colour[rows, position[:, 0], position[:, 1]]
            got = hit.any(-1) & ~done
            reached = jnp.where(got, jnp.argmax(hit, axis=-1), reached)
            done = done | got | (steps >= step_limit)
            return (carry, position, done, steps, reached, jnp.zeros_like(first)), None

        state = (
            carry,
            start,
            jnp.zeros(batch, dtype=bool),
            jnp.zeros(batch, dtype=jnp.int32),
            jnp.full(batch, -1, dtype=jnp.int32),
            jnp.ones(batch, dtype=bool),
        )
        (_, _, _, steps, reached, _), _ = jax.lax.scan(body, state, None, length=step_limit)
        return reached, steps

    return rollout


@dataclasses.dataclass(frozen=True)
class Readout:
    """What one parameter vector did on a fixed set of levels."""

    tau: float
    low: float
    high: float
    reached: float
    on_route: float
    took_rich: float
    tau_rich_above: float
    tau_rich_below: float

    def as_row(self) -> dict:
        return dataclasses.asdict(self)


def read_outcomes(targets, reached: np.ndarray, steps: np.ndarray, resamples: int = 200, seed: int = 0) -> Readout:
    """Threshold, competence and the two contingent thresholds from rollout outcomes.

    ``targets`` is the :class:`~goalmisgen.analysis.route_targets.RouteTargets` the
    rollouts started from. On-route is judged against the distance of whichever
    objective was reached; timeouts are excluded from the threshold, as in
    :func:`goalmisgen.analysis.behaviour.value_distance_decisions`.
    """
    reached, steps = np.asarray(reached), np.asarray(steps)
    finished = reached >= 0
    took_rich = (reached == targets.rich_feature).astype(float)
    chosen = np.where(took_rich > 0, targets.distances[:, 0], targets.distances[:, 1])
    gaps = targets.gap.astype(float)

    def tau_of(rows):
        rows = rows & finished
        return indifference_point(gaps[rows], took_rich[rows]) if rows.sum() > 10 else float("nan")

    tau = tau_of(np.ones_like(finished))
    index = np.flatnonzero(finished)
    low, high = bootstrap_episodes(
        lambda r: indifference_point(gaps[index[r]], took_rich[index[r]]), np.arange(len(index)), resamples=resamples, seed=seed
    )
    return Readout(
        tau=float(tau),
        low=float(low),
        high=float(high),
        reached=float(finished.mean()),
        on_route=float(np.mean(steps[finished] == chosen[finished])) if finished.any() else float("nan"),
        took_rich=float(took_rich[finished].mean()) if finished.any() else float("nan"),
        tau_rich_above=float(tau_of(targets.rich_above)),
        tau_rich_below=float(tau_of(~targets.rich_above)),
    )


def logistic_fit(gaps: np.ndarray, took_rich: np.ndarray, ridge: float = 1e-6, iterations: int = 50) -> tuple[float, float]:
    """Maximum-likelihood ``(threshold, scale)`` for ``p = sigmoid((threshold - gap) / scale)``, by Newton's method.

    Newton rather than the gradient descent in ``fit_logistic``: a sharp chooser
    has a steep slope, and gradient descent from zero stops well short of it,
    which would hand the search a softer target than the agent actually is.
    """
    x = np.stack([np.ones_like(gaps), gaps], axis=1)
    w = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + ridge * np.eye(2)
        w = w + np.linalg.solve(hessian, x.T @ (took_rich - p))
    bias, slope = w
    return float(-bias / slope), float(1.0 / abs(slope))
