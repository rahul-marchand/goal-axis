"""Teacher-forced and on-policy logits of a cleanba policy, for the cheapest-edit search.

Both run entirely inside JAX on a :class:`~goalmisgen.analysis.route_targets.RouteTargets`
batch and carry gradients through the network's outputs. The greedy
closed-loop rollout that reads an edit's behaviour back out, and the
observation drawing the two share, live in :mod:`goalmisgen.analysis.closed_loop`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from goalmisgen.analysis.closed_loop import MOVE_ARRAY, draw_agent
from goalmisgen.analysis.route_targets import BRANCH, FAR, FORK, PAD, PREFIX
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL, WALL_CHANNEL


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
            ~alive,
            PAD,
            jnp.where(
                committed[:, None] & (t == index[:, None]),
                FORK,
                jnp.where(committed[:, None] & (t > index[:, None]), BRANCH, PREFIX),
            ),
        ).astype(jnp.int8)
        return {
            "logits": logits,
            "positions": positions,
            "toward_rich": ex_rich,
            "toward_poor": ex_poor,
            "role": role,
            "committed": committed,
        }

    return fn
