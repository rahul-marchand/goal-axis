"""What the axis does that the colour-1 objective is not there for, and where the brightening lever runs out.

Two questions left over from the gain story, both answered by running the
network on an observation with an objective **erased**.

**The leftover.** A gain on colour ``c``'s input plane changes nothing at all on
a level with no colour-``c`` object: the plane is zero, so scaling it is the
identity. The fitted axis is not so clean -- with the colour-1 objective removed
it still moves activations a lot -- and the open question is whether that
object-independent part does anything to the threshold. Write ``dh`` for the
axis's activation change on the ordinary observation and ``dh_null`` for its
change on the same observation with the objective erased, both read on the same
trajectory. Patching a site with

    h_base + w * dh + w_null * dh_null

at ``(1, 0)`` is the ordinary full patch, at ``(0, 1)`` the object-independent
part alone and at ``(1, -1)`` what is left when it is taken out. The share of
the write's own threshold shift each one carries says whether the leftover is a
second, object-free value bias or merely noise the fit dragged along.

**The lever's ceiling.** Brightening colour 1 stops moving the threshold past
about x3 while dimming colour 0 goes on to tau = -18, and the natural reading is
that the *representation* of a brightened objective saturates while a dimmed one
has no floor. Erasing an objective measures that directly:

    S_c(g) = rms of [h(obs) - h(obs without objective c)]   under input gain g

is how much of the activation is there because objective ``c`` is, with no
fitted direction in it. If the ceiling is real, ``S_1`` flattens as the colour-1
gain grows, at the gain where the threshold flattens, while ``S_0`` falls
without limit as colour 0 is dimmed.

Both reuse :func:`goalmisgen.analysis.gain_alignment.make_step`, so sites are
the same DRC sites in the same forward order.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.closed_loop import MOVE_ARRAY, draw_agent
from goalmisgen.analysis.gain_alignment import make_step
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL, WALL_CHANNEL

MODES = {"full": (1.0, 0.0), "object-independent": (0.0, 1.0), "object-dependent": (1.0, -1.0)}
"""Patch weights ``(w, w_null)`` on the write's change and on its object-free part."""


def erase(static: np.ndarray, channels) -> np.ndarray:
    """A copy of ``static`` (N, S, S, C) with every objective plane in ``channels`` zeroed."""
    out = np.array(static, dtype=np.float32, copy=True)
    for channel in channels:
        out[..., channel] = 0.0
    return out


def make_leftover_rollout(policy, unravel, step_limit: int = 120):
    """Closed-loop rollouts of a patched network, lockstepped with four references.

    ``(main_flat, base_flat, write_flat, static, static_null, start, mix (S,), w (S,), w_null (S,))``.
    The base and the write are stepped on both the ordinary observation and on
    ``static_null``, all four on the main network's own trajectory, so ``dh`` and
    ``dh_null`` are read on the states the patched agent actually visits. Sites
    with ``mix[s] = 1`` are *set* to ``h_base + w[s] * dh + w_null[s] * dh_null``;
    setting rather than adding keeps a patch on a recurrent cell from compounding
    with its own feedback on the next tick.

    Returns ``(reached_feature, steps)`` like :func:`goalmisgen.analysis.closed_loop.make_rollout_fn`.
    """
    step = make_step(policy)

    @jax.jit
    def rollout(main_flat, base_flat, write_flat, static, static_null, start, mix, w, w_null):
        main, base, write = unravel(main_flat), unravel(base_flat), unravel(write_flat)
        batch, size, _, channels = static.shape
        wall = static[..., WALL_CHANNEL] > 0.5
        colour = static[..., FIRST_FEATURE_CHANNEL : FIRST_FEATURE_CHANNEL + 2] > 0.5
        rows = jnp.arange(batch)
        carry = policy.apply(base, jax.random.PRNGKey(0), (batch, channels, size, size), method=policy.initialize_carry)

        def body(state, _):
            c_main, c_ref, c_write, c_ref_n, c_write_n, position, done, steps, reached, first = state
            obs = draw_agent(static, position)
            obs_null = draw_agent(static_null, position)
            c_ref, _, s_ref = step(base, c_ref, obs, first)
            c_write, _, s_write = step(write, c_write, obs, first)
            c_ref_n, _, s_ref_n = step(base, c_ref_n, obs_null, first)
            c_write_n, _, s_write_n = step(write, c_write_n, obs_null, first)
            patches = []
            for k in range(len(s_ref)):
                full = s_write[k] - s_ref[k]
                null = s_write_n[k] - s_ref_n[k]
                patches.append((mix[k], s_ref[k] + w[k] * full + w_null[k] * null))
            c_main, logits, _ = step(main, c_main, obs, first, patches)
            action = jnp.argmax(logits, axis=-1)
            candidate = jnp.clip(position + MOVE_ARRAY[action], 0, size - 1)
            outside = jnp.any(position + MOVE_ARRAY[action] != candidate, axis=-1)
            blocked = outside | wall[rows, candidate[:, 0], candidate[:, 1]]
            position = jnp.where((done | blocked)[:, None], position, candidate)
            steps = steps + (~done).astype(jnp.int32)
            hit = colour[rows, position[:, 0], position[:, 1]]
            got = hit.any(-1) & ~done
            reached = jnp.where(got, jnp.argmax(hit, axis=-1), reached)
            done = done | got | (steps >= step_limit)
            return (c_main, c_ref, c_write, c_ref_n, c_write_n, position, done, steps, reached, jnp.zeros_like(first)), None

        state = (
            carry, carry, carry, carry, carry, start,
            jnp.zeros(batch, dtype=bool), jnp.zeros(batch, dtype=jnp.int32), jnp.full(batch, -1, dtype=jnp.int32),
            jnp.ones(batch, dtype=bool),
        )
        (*_, steps, reached, _), _ = jax.lax.scan(body, state, None, length=step_limit)
        return reached, steps

    return rollout


def make_change_moments_fn(policy, unravel):
    """Teacher-forced sizes of the write's change and of its object-free part, per site.

    ``(base_flat, write_flats (V, P), static, static_null, positions (B, T, 2), mask (B, T))``
    returns per site ``{"ff", "nn", "fn", "hh", "n"}``: the summed squares of
    ``dh`` and ``dh_null``, their inner product, the base activation's squared
    norm and the rows counted. The object-independent *share* of the change is
    ``nn / ff``, and ``fn / sqrt(ff nn)`` says whether the two point the same way.
    """
    step = make_step(policy)

    @jax.jit
    def moments(base_flat, write_flats, static, static_null, positions, mask):
        base = unravel(base_flat)
        batch, steps = positions.shape[:2]
        channels, size = static.shape[-1], static.shape[1]
        carry = policy.apply(base, jax.random.PRNGKey(0), (batch, channels, size, size), method=policy.initialize_carry)
        stack = lambda tree, n: jax.tree.map(lambda z: jnp.broadcast_to(z, (n,) + z.shape), tree)  # noqa: E731
        n_var = write_flats.shape[0]

        def body(state, inputs):
            c_ref, c_ref_n, c_var, c_var_n, acc = state
            position, first, m = inputs
            obs = draw_agent(static, position)
            obs_null = draw_agent(static_null, position)
            starts = jnp.full((batch,), first)
            c_ref, _, s_ref = step(base, c_ref, obs, starts)
            c_ref_n, _, s_ref_n = step(base, c_ref_n, obs_null, starts)
            c_var, _, s_var = jax.vmap(lambda c, f: step(unravel(f), c, obs, starts))(c_var, write_flats)
            c_var_n, _, s_var_n = jax.vmap(lambda c, f: step(unravel(f), c, obs_null, starts))(c_var_n, write_flats)
            m = m.astype(static.dtype)
            new = []
            for k in range(len(s_ref)):
                full = s_var[k] - s_ref[k][None]
                null = s_var_n[k] - s_ref_n[k][None]
                weigh = lambda x: (x * m.reshape((1, -1) + (1,) * (x.ndim - 2))).reshape(n_var, -1)  # noqa: E731
                f, n = weigh(full), weigh(null)
                old = acc[k]
                new.append({
                    "ff": old["ff"] + (f * f).sum(-1), "nn": old["nn"] + (n * n).sum(-1), "fn": old["fn"] + (f * n).sum(-1),
                    "hh": old["hh"] + ((s_ref[k] * m.reshape((-1,) + (1,) * (s_ref[k].ndim - 1))) ** 2).sum(),
                    "n": old["n"] + m.sum(),
                })
            return (c_ref, c_ref_n, c_var, c_var_n, new), None

        _, _, probe = jax.eval_shape(lambda: step(base, carry, draw_agent(static, positions[:, 0]), jnp.ones((batch,), bool)))
        acc = [{"ff": jnp.zeros(n_var), "nn": jnp.zeros(n_var), "fn": jnp.zeros(n_var), "hh": jnp.zeros(()), "n": jnp.zeros(())}
               for _ in probe]
        state = (carry, carry, stack(carry, n_var), stack(carry, n_var), acc)
        inputs = (positions.transpose(1, 0, 2), jnp.arange(steps) == 0, mask.T)
        (*_, acc), _ = jax.lax.scan(body, state, inputs)
        return acc

    return moments


def make_signal_fn(policy, unravel):
    """Teacher-forced ``S_c(g)``: how much of the activation is there because objective ``c`` is.

    ``(base_flat, gains (G, C), static, static_no_0, static_no_1, positions, mask)``
    returns per site ``{"s0", "s1", "hh", "n"}`` of shape ``(G,)`` (``hh`` too):
    summed squares of ``h(obs) - h(obs without objective 0 | 1)`` and of ``h``
    itself, every observation scaled by that row of ``gains`` first.
    """
    step = make_step(policy)

    @jax.jit
    def signals(base_flat, gains, static, static_no_0, static_no_1, positions, mask):
        base = unravel(base_flat)
        batch, steps = positions.shape[:2]
        channels, size = static.shape[-1], static.shape[1]
        carry = policy.apply(base, jax.random.PRNGKey(0), (batch, channels, size, size), method=policy.initialize_carry)
        stack = lambda tree, n: jax.tree.map(lambda z: jnp.broadcast_to(z, (n,) + z.shape), tree)  # noqa: E731
        n_gain = gains.shape[0]

        def body(state, inputs):
            c_plain, c_no0, c_no1, acc = state
            position, first, m = inputs
            starts = jnp.full((batch,), first)
            obs = draw_agent(static, position)
            obs0 = draw_agent(static_no_0, position)
            obs1 = draw_agent(static_no_1, position)
            run = lambda c, o: jax.vmap(lambda cc, g: step(base, cc, o * g, starts))(c, gains)  # noqa: E731
            c_plain, _, s_plain = run(c_plain, obs)
            c_no0, _, s_no0 = run(c_no0, obs0)
            c_no1, _, s_no1 = run(c_no1, obs1)
            m = m.astype(static.dtype)
            new = []
            for k in range(len(s_plain)):
                weigh = lambda x: (x * m.reshape((1, -1) + (1,) * (x.ndim - 2))).reshape(n_gain, -1)  # noqa: E731
                h = weigh(s_plain[k])
                d0 = h - weigh(s_no0[k])
                d1 = h - weigh(s_no1[k])
                old = acc[k]
                new.append({"s0": old["s0"] + (d0 * d0).sum(-1), "s1": old["s1"] + (d1 * d1).sum(-1),
                            "hh": old["hh"] + (h * h).sum(-1), "n": old["n"] + m.sum()})
            return (c_plain, c_no0, c_no1, new), None

        _, _, probe = jax.eval_shape(lambda: step(base, carry, draw_agent(static, positions[:, 0]), jnp.ones((batch,), bool)))
        acc = [{"s0": jnp.zeros(n_gain), "s1": jnp.zeros(n_gain), "hh": jnp.zeros(n_gain), "n": jnp.zeros(())} for _ in probe]
        state = (stack(carry, n_gain), stack(carry, n_gain), stack(carry, n_gain), acc)
        inputs = (positions.transpose(1, 0, 2), jnp.arange(steps) == 0, mask.T)
        (*_, acc), _ = jax.lax.scan(body, state, inputs)
        return acc

    return signals
