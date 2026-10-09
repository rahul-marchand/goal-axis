"""Is the value axis a gain on the input all the way down, or does it become something else?

The restricted writes put a large share of the DRC axis's effect in the input
convolution, and inside it partly a gain on each colour plane. But most of the
effect survives *without* the input convolution, so the question is what the
downstream part does. Two stories predict different activation changes:

* **gain, entangled with planning** -- the write changes how loudly each colour
  is carried through the network. Wherever you look, the activation change
  ``dh_l`` is what turning up a colour's brightness would have done there.
* **a factorised circuit** -- somewhere the write's activation change stops
  looking like a brightness change, and that residual is what moves the
  threshold.

Brightness gives the gain story a reference at every site with no fitted
direction in it. At layer ``l``, colour ``c``'s contribution is

    C_l^c = dh_l / dlog(brightness_c)   (central difference in observation space)

computed on the base network running the same observations. The write's change
``dh_l`` (central difference over +-offset) is regressed on ``[C_l^0, C_l^1]``
with two coefficients per channel (``per-channel``) or two overall
(``global``, the strict "this write *is* a brightness change"). R^2 is
uncentred: the fraction of the change's energy the gain explains.

Then mediation. During closed-loop rollouts a *main* network with the base's
weights has one site group *set* to the unpatched base's value there plus the
write's change -- in full, only its gain-explained component, or only the
residual (an interchange intervention, so recurrent feedback does not count a
change twice) -- with the
write, the unpatched base and four brightness copies stepped in lockstep on the
main network's own trajectory so the change and the gain reference are both
read on the states the main agent actually visits. The threshold shift each
patch produces, as a fraction of the write's own, is the share of the effect
carried by that component at that site.

Every site is read *after* its nonlinearity, in forward order, the recurrent
cells once per tick. The forward pass here re-assembles the DRC from cleanba's
own flax modules rather than calling ``Policy.get_action`` so that sites can be
read and patched; ``tests/test_gain_alignment.py`` checks it reproduces the
policy's logits step for step.
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from cleanba.convlstm import ConvLSTMCell, ConvLSTMConfig
from cleanba.network import IdentityNorm

from goalmisgen.analysis.closed_loop import MOVE_ARRAY, draw_agent
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL, WALL_CHANNEL


@dataclasses.dataclass(frozen=True)
class Site:
    name: str
    group: str
    """Sites patched together in mediation: a cell's every tick is one group."""


def sites_of(cfg: ConvLSTMConfig) -> list[Site]:
    """Every readable site, in the order :func:`make_step` visits them."""
    out = [
        Site("input conv" if i == 0 else f"conv {i}", "input conv" if i == 0 else f"conv {i}") for i in range(len(cfg.embed))
    ]
    for tick in range(cfg.repeats_per_step):
        for depth in range(cfg.n_recurrent):
            out.append(Site(f"cell {depth} tick {tick}", f"cell {depth}"))
    out += [Site("dense", "dense"), Site("logits", "logits")]
    return out


def groups_of(sites: list[Site]) -> list[str]:
    seen: list[str] = []
    for site in sites:
        if site.group not in seen:
            seen.append(site.group)
    return seen


def make_step(policy):
    """``(params, carry, obs_nhwc, starts, patches) -> (carry, logits, sites)``: one DRC step, every site exposed.

    ``patches`` is ``None`` or a list, one ``(mix, target)`` pair per site in
    :func:`sites_of` order: the site's value becomes ``value + mix * (target -
    value)`` before anything downstream reads it (for a cell, before the tick's
    next layer, the next tick's recurrence and the head). ``mix = 1`` is an
    interchange intervention -- the site is *set*, so a patch on a recurrent cell
    does not compound with its own feedback on the next tick, which an additive
    patch does. ``obs_nhwc`` is the observation as the encoder builds it, before
    cleanba's /255.
    """
    cfg = policy.cfg
    if not isinstance(cfg, ConvLSTMConfig):
        raise TypeError(f"gain alignment reassembles the DRC; got {type(cfg).__name__}")
    if cfg.residual or not isinstance(cfg.norm, IdentityNorm) or cfg.normalize_input:
        raise NotImplementedError("residual cells, normalisation layers and normalize_input are not reassembled")
    convs = [c.make_conv() for c in cfg.embed]
    cell = ConvLSTMCell(cfg.recurrent)
    denses = [nn.Dense(h) for h in cfg.mlp_hiddens]
    actor = nn.Dense(policy.n_actions)

    def step(params, carry, obs, starts, patches=None):
        net = params["params"]["network_params"]
        sites = []

        def hook(value):
            if patches is not None:
                mix, target = patches[len(sites)]
                value = value + mix * (target - value)
            sites.append(value)
            return value

        x = obs / 255.0
        for i, conv in enumerate(convs):
            x = conv.apply({"params": net[f"conv_list_{i}"]}, x)
            if cfg.use_relu and i < len(convs) - 1:
                x = nn.relu(x)
            x = hook(x)

        keep = (~starts).astype(x.dtype)
        carry = jax.tree.map(lambda z: z * keep.reshape((-1,) + (1,) * (z.ndim - 1)), carry)
        for _ in range(cfg.repeats_per_step):
            carry = list(carry)
            prev = carry[-1].h
            for depth in range(cfg.n_recurrent):
                state, h = cell.apply({"params": net[f"cell_list_{depth}"]}, carry[depth], x, prev)
                h = hook(h)
                carry[depth] = state.replace(h=h)
                prev = h

        hidden = carry[-1].h + x if cfg.skip_final else carry[-1].h
        hidden = hidden.reshape(hidden.shape[0], -1)
        for i, dense in enumerate(denses):
            hidden = nn.relu(dense.apply({"params": net[f"dense_list_{i}"]}, hidden))
        hidden = hook(hidden)
        logits = hook(actor.apply({"params": params["params"]["actor_params"]["Output"]}, hidden))
        return carry, logits, sites

    return step


def brightness_gains(n_inputs: int, eps: float, colours=(FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)) -> np.ndarray:
    """``(4, n_inputs)``: colour 0 at 1+eps, 1-eps, then colour 1 at 1+eps, 1-eps."""
    out = np.ones((4, n_inputs), dtype=np.float32)
    for k, channel in enumerate(colours):
        out[2 * k, channel] = 1 + eps
        out[2 * k + 1, channel] = 1 - eps
    return out


def _sum_channels(x, mask):
    """Sum over every axis but the last, rows weighted by ``mask`` (batch,)."""
    x = x * mask.reshape((-1,) + (1,) * (x.ndim - 1))
    return x.reshape(-1, x.shape[-1]).sum(0)


def make_moments_fn(policy, unravel, eps: float = 0.05):
    """Teacher-forced sufficient statistics for the gain regression, per variant and site.

    ``(base_flat, flats (2V, P), gains (2V, I), static (B, S, S, I), positions (B, T, 2), mask (B, T))``
    where rows ``0..V-1`` of ``flats``/``gains`` are each variant at +offset and
    rows ``V..2V-1`` the same variants at -offset. Returns per site a dict of
    per-channel sums: ``cc`` (3, K) of C0^2, C0 C1, C1^2; ``dc`` (V, 2, K);
    ``dd`` (V, K); ``hh`` (K,) of the base activation; ``n`` rows counted.
    """
    step = make_step(policy)

    @jax.jit
    def moments(base_flat, flats, gains, static, positions, mask):
        base = unravel(base_flat)
        n_inputs = static.shape[-1]
        bright = jnp.asarray(brightness_gains(n_inputs, eps))
        batch, steps = positions.shape[:2]
        carry = policy.apply(
            base, jax.random.PRNGKey(0), (batch, n_inputs, static.shape[1], static.shape[2]), method=policy.initialize_carry
        )
        stack = lambda tree, n: jax.tree.map(lambda z: jnp.broadcast_to(z, (n,) + z.shape), tree)  # noqa: E731
        n_var = flats.shape[0]

        def one(params, carry, obs, g, starts):
            return step(params, carry, obs * g, starts)

        def body(state, inputs):
            c_base, c_bright, c_var, acc = state
            position, first, m = inputs
            obs = draw_agent(static, position)
            starts = jnp.full((batch,), first)
            c_base, _, s_base = step(base, c_base, obs, starts)
            c_bright, _, s_bright = jax.vmap(lambda c, g: one(base, c, obs, g, starts))(c_bright, bright)
            c_var, _, s_var = jax.vmap(lambda c, f, g: one(unravel(f), c, obs, g, starts))(c_var, flats, gains)
            m = m.astype(static.dtype)
            new = []
            for k in range(len(s_base)):
                c0 = (s_bright[k][0] - s_bright[k][1]) / (2 * eps)
                c1 = (s_bright[k][2] - s_bright[k][3]) / (2 * eps)
                d = (s_var[k][: n_var // 2] - s_var[k][n_var // 2 :]) / 2
                sum_ = lambda x: _sum_channels(x, m)  # noqa: E731
                vsum = jax.vmap(sum_)
                cc = jnp.stack([sum_(c0 * c0), sum_(c0 * c1), sum_(c1 * c1)])
                dc = jnp.stack([vsum(d * c0[None]), vsum(d * c1[None])], axis=1)
                dd = vsum(d * d)
                hh = sum_(s_base[k] ** 2)
                old = acc[k]
                new.append(
                    {
                        "cc": old["cc"] + cc,
                        "dc": old["dc"] + dc,
                        "dd": old["dd"] + dd,
                        "hh": old["hh"] + hh,
                        "n": old["n"] + m.sum(),
                    }
                )
            return (c_base, c_bright, c_var, new), None

        # Shapes of every site, for the zero accumulators.
        _, _, probe = jax.eval_shape(lambda: step(base, carry, draw_agent(static, positions[:, 0]), jnp.ones((batch,), bool)))
        acc = []
        for site in probe:
            k = site.shape[-1]
            acc.append(
                {
                    "cc": jnp.zeros((3, k)),
                    "dc": jnp.zeros((n_var // 2, 2, k)),
                    "dd": jnp.zeros((n_var // 2, k)),
                    "hh": jnp.zeros((k,)),
                    "n": jnp.zeros(()),
                }
            )
        state = (carry, stack(carry, 4), stack(carry, n_var), acc)
        inputs = (positions.transpose(1, 0, 2), jnp.arange(steps) == 0, mask.T)
        (_, _, _, acc), _ = jax.lax.scan(body, state, inputs)
        return acc

    return moments


def add_moments(a, b):
    """Sum two accumulator lists elementwise; ``None`` starts one.

    Shared with :mod:`goalmisgen.analysis.leftover`, whose moment and signal
    functions return the same list-of-dicts shape.
    """
    if a is None:
        return [{key: np.asarray(value) for key, value in site.items()} for site in b]
    return [{key: np.asarray(x[key]) + np.asarray(y[key]) for key in x} for x, y in zip(a, b)]


@dataclasses.dataclass(frozen=True)
class Fit:
    r2_per_channel: float
    r2_global: float
    coef_per_channel: np.ndarray  # (2, K)
    coef_global: np.ndarray  # (2,)
    change: float
    """RMS of the write's change relative to RMS of the base activation."""


def fit_site(moments: dict, variant: int, ridge: float = 1e-9) -> Fit:
    """Least squares of ``d`` on ``[C0, C1]`` from summed moments; uncentred R^2."""
    cc = np.asarray(moments["cc"], dtype=np.float64)  # (3, K)
    dc = np.asarray(moments["dc"][variant], dtype=np.float64)  # (2, K)
    dd = np.asarray(moments["dd"][variant], dtype=np.float64)  # (K,)
    hh = np.asarray(moments["hh"], dtype=np.float64)
    total = float(dd.sum())
    gram = np.stack([np.stack([cc[0], cc[1]], -1), np.stack([cc[1], cc[2]], -1)], -2)  # (K, 2, 2)
    scale = np.maximum(np.trace(gram, axis1=1, axis2=2), 1e-30)
    gram_r = gram + ridge * scale[:, None, None] * np.eye(2)
    coef = np.linalg.solve(gram_r, dc.T[..., None])[..., 0]  # (K, 2)
    explained = np.einsum("ki,ki->k", coef, dc.T)
    g_gram = gram.sum(0) + ridge * float(scale.sum()) * np.eye(2)
    g_coef = np.linalg.solve(g_gram, dc.sum(1))
    g_explained = float(g_coef @ dc.sum(1))
    if total <= 1e-20 * max(float(hh.sum()), 1e-30):
        nan = float("nan")
        return Fit(nan, nan, coef.T, g_coef, 0.0)
    return Fit(
        r2_per_channel=float(1 - (total - explained.sum()) / total),
        r2_global=float(1 - (total - g_explained) / total),
        coef_per_channel=coef.T,
        coef_global=g_coef,
        change=float(np.sqrt(total / max(float(hh.sum()), 1e-30))),
    )


def make_mediation_rollout(policy, unravel, step_limit: int = 120, eps: float = 0.05):
    """Closed-loop greedy rollouts of a patched main network, lockstepped with its references.

    ``(main_flat, base_flat, write_flat, write_gains, static, start, mix (S,), full_w (S,), gain_w (S,), coef_a, coef_b)``
    with ``coef_a``/``coef_b`` lists of per-site ``(K,)`` arrays. Sites with
    ``mix[s] = 1`` are *set* to ``h_base + full_w[s] * dh + gain_w[s] * (a * C0 + b * C1)``,
    ``dh`` being the write minus the unpatched base on the same carry history.
    Full patch = (1, 0), gain = (0, 1), residual = (1, -1).
    Returns ``(reached_feature, steps)`` like :func:`goalmisgen.analysis.closed_loop.make_rollout_fn`.
    """
    step = make_step(policy)

    @jax.jit
    def rollout(main_flat, base_flat, write_flat, write_gains, static, start, mix, full_w, gain_w, coef_a, coef_b):
        main, base, write = unravel(main_flat), unravel(base_flat), unravel(write_flat)
        batch, size, _, channels = static.shape
        bright = jnp.asarray(brightness_gains(channels, eps))
        wall = static[..., WALL_CHANNEL] > 0.5
        colour = static[..., FIRST_FEATURE_CHANNEL : FIRST_FEATURE_CHANNEL + 2] > 0.5
        rows = jnp.arange(batch)
        carry = policy.apply(base, jax.random.PRNGKey(0), (batch, channels, size, size), method=policy.initialize_carry)
        stack = lambda tree, n: jax.tree.map(lambda z: jnp.broadcast_to(z, (n,) + z.shape), tree)  # noqa: E731

        def body(state, _):
            c_main, c_ref, c_write, c_bright, position, done, steps, reached, first = state
            obs = draw_agent(static, position)
            c_ref, _, s_ref = step(base, c_ref, obs, first)
            c_write, _, s_write = step(write, c_write, obs * write_gains, first)
            c_bright, _, s_bright = jax.vmap(lambda c, g: step(base, c, obs * g, first))(c_bright, bright)
            patches = []
            for k in range(len(s_ref)):
                c0 = (s_bright[k][0] - s_bright[k][1]) / (2 * eps)
                c1 = (s_bright[k][2] - s_bright[k][3]) / (2 * eps)
                gain = coef_a[k] * c0 + coef_b[k] * c1
                patches.append((mix[k], s_ref[k] + full_w[k] * (s_write[k] - s_ref[k]) + gain_w[k] * gain))
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
            return (c_main, c_ref, c_write, c_bright, position, done, steps, reached, jnp.zeros_like(first)), None

        state = (
            carry,
            carry,
            carry,
            stack(carry, 4),
            start,
            jnp.zeros(batch, dtype=bool),
            jnp.zeros(batch, dtype=jnp.int32),
            jnp.full(batch, -1, dtype=jnp.int32),
            jnp.ones(batch, dtype=bool),
        )
        (_, _, _, _, _, _, steps, reached, _), _ = jax.lax.scan(body, state, None, length=step_limit)
        return reached, steps

    return rollout


def fraction_carried(tau_patch_plus: float, tau_patch_minus: float, tau_write_plus: float, tau_write_minus: float) -> float:
    """Antisymmetric threshold shift of a patch over that of the write itself."""
    denominator = tau_write_plus - tau_write_minus
    return float((tau_patch_plus - tau_patch_minus) / denominator) if abs(denominator) > 1e-9 else float("nan")
