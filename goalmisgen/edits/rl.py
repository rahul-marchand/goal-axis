"""Cleanba policies under the cheapest-edit search: teacher-forced logits and closed-loop rollouts.

Both run entirely inside JAX on a :class:`~goalmisgen.edits.targets.RouteTargets`
batch, so an edit can be evaluated on thousands of held-out levels without the
gymnasium vector environment, and every condition is measured on the *same*
levels. The rollout reproduces ``MazeEnv``'s rules -- a move into a wall costs a
step and goes nowhere, reaching either objective ends the episode, the step
limit truncates -- and ``tests/test_edits.py`` checks it against the
environment's own replay.

Observations are built as the encoder builds them (``static`` plus a one-hot
agent) and handed over NCHW, as cleanba's vector-env wrapper would.
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


def make_logits_fn(policy, unravel):
    """``(flat, batch) -> (N, 2, T, 4)`` logits along both branches of every level."""

    # One step at a time under jax.checkpoint rather than cleanba's own scan: the
    # backward pass then holds one step's activations instead of every step's,
    # which for the DRC is the difference between ~1 GB and a 16 GB gate tensor.
    def logits_fn(flat, batch):
        params = unravel(flat)
        static, positions = batch["static"], batch["positions"]
        n, branches, steps, _ = positions.shape
        size, channels = static.shape[1], static.shape[-1]
        batch = n * branches
        static = jnp.broadcast_to(static[:, None], (n, branches, size, size, channels)).reshape(batch, size, size, channels)
        positions = positions.reshape(batch, steps, 2).transpose(1, 0, 2)  # (T, B, 2)
        carry = policy.apply(params, jax.random.PRNGKey(0), (batch, channels, size, size), method=policy.initialize_carry)

        @jax.checkpoint
        def step(carry, inputs):
            position, first = inputs
            obs = draw_agent(static, position).transpose(0, 3, 1, 2)
            starts = jnp.full((batch,), first)
            carry, _, logits, _ = policy.apply(
                params, carry, obs, starts, jax.random.PRNGKey(0), temperature=0.0, method=policy.get_action
            )
            return carry, logits

        _, logits = jax.lax.scan(step, carry, (positions, jnp.arange(steps) == 0))
        return logits.transpose(1, 0, 2).reshape(n, branches, steps, -1)

    return logits_fn


def make_onpolicy_fn(policy, unravel, horizon: int = 40):
    """``(flat, batch) -> dict``: the greedy agent's own first ``horizon`` steps, with gradients through its logits.

    The positions are discrete, so no gradient flows through where the agent
    goes -- only through what it outputs at the states it actually visits, with
    the carry its own history produced. That is the difference from teacher
    forcing along the solver's routes, where an edit can change the decision at
    the solver's fork while the agent, in closed loop, commits somewhere else.

    Returns ``logits (B, L, 4)``, ``positions (B, L, 2)``, ``role (B, L)`` and
    the exclusive move sets ``toward_rich`` / ``toward_poor`` (B, L, 4), plus
    ``committed (B,)``. The commitment step is the first at which the agent
    takes a move that shortens exactly one objective's route while the other
    objective also had a move of its own available there; its role is FORK,
    earlier steps PREFIX, later live steps BRANCH.
    """
    from goalmisgen.edits.targets import BRANCH, FAR, FORK, PAD, PREFIX

    def fn(flat, batch):
        params = unravel(flat)
        static, fields = batch["static"], batch["fields"]
        size, channels = static.shape[1], static.shape[-1]
        n = static.shape[0]
        rows = jnp.arange(n)
        wall = static[..., WALL_CHANNEL] > 0.5
        colour = static[..., FIRST_FEATURE_CHANNEL : FIRST_FEATURE_CHANNEL + 2] > 0.5
        carry = policy.apply(params, jax.random.PRNGKey(0), (n, channels, size, size), method=policy.initialize_carry)

        @jax.checkpoint
        def step(state, _):
            carry, position, done, first = state
            obs = draw_agent(static, position).transpose(0, 3, 1, 2)
            carry, _, logits, _ = policy.apply(
                params, carry, obs, first, jax.random.PRNGKey(0), temperature=0.0, method=policy.get_action
            )
            action = jnp.argmax(jax.lax.stop_gradient(logits), axis=-1)

            raw = position[:, None, :] + MOVE_ARRAY[None]  # (B, 4, 2)
            neighbour = jnp.clip(raw, 0, size - 1)
            inside = jnp.all(raw == neighbour, axis=-1)
            here = fields[rows, :, position[:, 0], position[:, 1]]  # (B, 2)
            there = fields[rows[:, None], :, neighbour[..., 0], neighbour[..., 1]]  # (B, 4, 2)
            shorten = inside[..., None] & (here[:, None, :] < FAR) & (there == here[:, None, :] - 1)
            rich_moves, poor_moves = shorten[..., 0], shorten[..., 1]
            exclusive_rich, exclusive_poor = rich_moves & ~poor_moves, poor_moves & ~rich_moves

            candidate = neighbour[rows, action]
            blocked = ~inside[rows, action] | wall[rows, candidate[:, 0], candidate[:, 1]]
            moved = jnp.where((done | blocked)[:, None], position, candidate)
            hit = colour[rows, moved[:, 0], moved[:, 1]].any(-1)
            out = (logits, position, exclusive_rich, exclusive_poor, action, ~done)
            return (carry, moved, done | hit, jnp.zeros_like(first)), out

        state = (carry, batch["start"], jnp.zeros(n, dtype=bool), jnp.ones(n, dtype=bool))
        _, (logits, positions, ex_rich, ex_poor, actions, alive) = jax.lax.scan(step, state, None, length=horizon)
        logits, positions, ex_rich, ex_poor, actions, alive = (
            jnp.moveaxis(x, 0, 1) for x in (logits, positions, ex_rich, ex_poor, actions, alive)
        )
        took = jnp.arange(4)[None, None, :] == actions[..., None]
        commit = alive & ex_rich.any(-1) & ex_poor.any(-1) & ((took & ex_rich).any(-1) | (took & ex_poor).any(-1))
        committed = commit.any(-1)
        index = jnp.argmax(commit, axis=-1)
        t = jnp.arange(horizon)[None, :]
        role = jnp.where(
            ~alive, PAD,
            jnp.where(committed[:, None] & (t == index[:, None]), FORK,
                      jnp.where(committed[:, None] & (t > index[:, None]), BRANCH, PREFIX)),
        ).astype(jnp.int8)
        return {"logits": logits, "positions": positions, "toward_rich": ex_rich, "toward_poor": ex_poor,
                "role": role, "committed": committed}

    return fn


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

    ``targets`` is the :class:`~goalmisgen.edits.targets.RouteTargets` the
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


def logistic_scale(targets, reached: np.ndarray) -> float:
    """The base agent's logistic scale in steps: ``1 / |slope|`` of p(rich) against the gap."""
    finished = np.asarray(reached) >= 0
    took_rich = (np.asarray(reached) == targets.rich_feature).astype(float)[finished]
    gaps = targets.gap.astype(float)[finished]
    return logistic_fit(gaps, took_rich)[1]


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
