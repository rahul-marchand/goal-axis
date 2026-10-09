"""What the planner propagates, and how two objectives combine in it.

Every account of the threshold except a reactive policy has to propagate
distance information through the maze. They differ in *what* is propagated and
in *how* the two objectives combine:

* **utility field** -- value is baked in at the source and the objectives
  compete, each cell taking whichever field is stronger there;
* **salience-weighted diffusion** -- value is at the source and the fields
  superpose;
* **factorised** -- what spreads is value-free distance, and value is applied
  once at a late comparison;
* **reactive** -- nothing spreads at all.

Running the same states with an objective erased separates these without
fitting anything. With ``h(x)`` the activation at a site on observation variant
``x``,

    A_c = h(only c) - h(neither)          what is there because objective c is
    D   = h(both)  - h(neither)           what is there because either is
    I   = D - A_0 - A_1                   what the two do only together

``A_c`` is a spatial field, one vector per maze cell, so its size against the
BFS distance from objective ``c`` is the field's shape, and how it grows over
ticks is the propagation. ``I`` is the interaction: small means the objectives
superpose, large means they compete. Applying a value manipulation and
decomposing the change,

    dh(both) = dh(neither) + dA_0 + dA_1 + dI,

says where value enters: in ``dA_c`` it rides the field, in ``dI`` it acts at
the comparison, in ``dh(neither)`` it is a bias that does not need an objective
at all.

The four variants are run on the **same teacher-forced positions**, so every
variant sees identical states and the later ticks stay comparable; erasing an
objective changes where a free-running agent would go, which would confound the
comparison from the second step onwards.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.gain_alignment import make_step
from goalmisgen.edits.rl import draw_agent
from goalmisgen.edits.targets import FAR
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL, WALL_CHANNEL

BOTH, RICH_ONLY, POOR_ONLY, NEITHER = 0, 1, 2, 3
"""Rows of the variant axis, in the order :func:`erasure_statics` builds them."""


def erasure_statics(targets) -> np.ndarray:
    """``(4, N, S, S, C)``: both objectives, only the richer, only the poorer, neither.

    An objective is erased by zeroing the one cell of its colour plane, which is
    all the plane holds. Which plane that is differs by level whenever colour
    does not track value, so ``targets.rich_feature`` picks it per level.
    """
    static = np.asarray(targets.static, dtype=np.float32)
    out = np.stack([static.copy() for _ in range(4)])
    for row, rich_feature in enumerate(np.asarray(targets.rich_feature, dtype=np.int64)):
        rich = FIRST_FEATURE_CHANNEL + int(rich_feature)
        poor = FIRST_FEATURE_CHANNEL + 1 - int(rich_feature)
        out[RICH_ONLY, row, :, :, poor] = 0.0
        out[POOR_ONLY, row, :, :, rich] = 0.0
        out[NEITHER, row, :, :, rich] = 0.0
        out[NEITHER, row, :, :, poor] = 0.0
    return out


def free_cells(targets) -> np.ndarray:
    """``(N, S, S)`` bool: cells an agent can stand in (padding counts as wall)."""
    return np.asarray(targets.static)[..., WALL_CHANNEL] < 0.5


def make_grid_fn(policy, unravel, n_steps: int, site_index: tuple[int, ...]):
    """``(flat, gain, statics (4,B,S,S,C), positions (B,T,2)) -> [(T,4,B,S,S,K)]``.

    One teacher-forced rollout per variant, stepped in lockstep on the same
    positions, returning the chosen sites' grids at every step. ``gain`` scales
    the observation, so a brightness manipulation and a weight manipulation go
    through the same path.
    """
    step = make_step(policy)

    @jax.jit
    def grids(flat, gain, statics, positions):
        params = unravel(flat)
        variants, batch, size, _, channels = statics.shape
        carry = policy.apply(params, jax.random.PRNGKey(0), (batch, channels, size, size), method=policy.initialize_carry)
        carries = jax.tree.map(lambda z: jnp.broadcast_to(z, (variants,) + z.shape), carry)

        def body(state, inputs):
            index, first = inputs
            position = positions[:, index]
            starts = jnp.full((batch,), first)

            def one(carry_v, static_v):
                return step(params, carry_v, draw_agent(static_v, position) * gain, starts)

            carries_next, _, sites = jax.vmap(one)(state, statics)
            return carries_next, [sites[k] for k in site_index]

        _, out = jax.lax.scan(body, carries, (jnp.arange(n_steps), jnp.arange(n_steps) == 0))
        return out

    return grids


def decompose(grid: np.ndarray):
    """``(4, ...) -> (A_rich, A_poor, D, I)`` for one site at one step."""
    both, rich, poor, none = grid
    a_rich = rich - none
    a_poor = poor - none
    total = both - none
    return a_rich, a_poor, total, total - a_rich - a_poor


def cell_norms(field: np.ndarray) -> np.ndarray:
    """``(B, S, S, K) -> (B, S, S)``: the field's size at each cell."""
    return np.sqrt((np.asarray(field, dtype=np.float64) ** 2).sum(-1))


def energies(a_rich, a_poor, total, inter, mask) -> dict:
    """Summed squares over masked cells, for the shares the report quotes."""
    weigh = lambda x: float(((np.asarray(x, dtype=np.float64) ** 2).sum(-1) * mask).sum())  # noqa: E731
    return {
        "rich": weigh(a_rich), "poor": weigh(a_poor), "total": weigh(total), "interaction": weigh(inter),
        "sum_of_parts": weigh(np.asarray(a_rich) + np.asarray(a_poor)), "cells": float(mask.sum()),
    }


def combination_fit(a_near, a_far, total, mask) -> dict:
    """Least squares ``D ~ alpha * A_near + gamma * A_far`` over masked cells.

    Superposition predicts ``(1, 1)``; a winner-take-all rule predicts
    ``(1, 0)``, the further objective contributing nothing where the nearer one
    is present. ``gamma`` is therefore how much of the losing objective survives
    the combination, and it is identifiable only where both fields are actually
    present -- hence the mask and the cell count returned with it.
    """
    m = mask[..., None].astype(np.float64)
    near = np.asarray(a_near, dtype=np.float64) * m
    far = np.asarray(a_far, dtype=np.float64) * m
    target = np.asarray(total, dtype=np.float64) * m
    gram = np.array([[(near * near).sum(), (near * far).sum()], [(near * far).sum(), (far * far).sum()]])
    rhs = np.array([(near * target).sum(), (far * target).sum()])
    if np.linalg.cond(gram) > 1e10:
        return {"alpha": float("nan"), "gamma": float("nan"), "r2": float("nan"),
                "cells": float(mask.sum()), "identifiable": False}
    alpha, gamma = np.linalg.solve(gram, rhs)
    residual = target - alpha * near - gamma * far
    denominator = (target * target).sum()
    return {"alpha": float(alpha), "gamma": float(gamma),
            "r2": float(1 - (residual * residual).sum() / denominator) if denominator > 0 else float("nan"),
            "cells": float(mask.sum()), "identifiable": True}


def field_profile(field, distance, mask, max_distance: int = 12) -> dict:
    """Mean field size per BFS distance from its own objective, and the decay fit.

    Returns the per-distance means, the slope of ``ln size`` on distance (a
    straight line means exponential decay), the slope of size on distance (a
    straight line means a cost-subtracted utility) and the half-distance.
    """
    size = cell_norms(field)
    distance = np.asarray(distance)
    usable = mask & (distance < FAR) & (distance >= 0)
    means, counts = [], []
    for d in range(max_distance + 1):
        rows = usable & (distance == d)
        counts.append(int(rows.sum()))
        means.append(float(size[rows].mean()) if rows.any() else float("nan"))
    means = np.asarray(means)
    good = np.isfinite(means) & (np.asarray(counts) >= 20) & (means > 0)
    out = {"means": means.tolist(), "counts": counts, "log_slope": float("nan"),
           "log_r2": float("nan"), "linear_slope": float("nan"), "half_distance": float("nan")}
    if good.sum() >= 3:
        d = np.arange(max_distance + 1)[good]
        log_fit = np.polyfit(d, np.log(means[good]), 1)
        predicted = np.polyval(log_fit, d)
        residual = np.log(means[good]) - predicted
        out["log_slope"] = float(log_fit[0])
        out["log_r2"] = float(1 - (residual ** 2).sum() / ((np.log(means[good]) - np.log(means[good]).mean()) ** 2).sum())
        out["linear_slope"] = float(np.polyfit(d, means[good], 1)[0])
        half = means[good][0] / 2
        below = np.where(means[good] <= half)[0]
        out["half_distance"] = float(d[below[0]]) if len(below) else float("nan")
    return out


def chebyshev_distance(targets) -> np.ndarray:
    """``(N, 2, S, S)``: grid distance to each objective, ignoring walls.

    The companion to ``targets.fields``, which is the BFS distance *through* the
    maze. A convolution spreads in grid distance whatever the walls do, so a
    field that tracks BFS distance is doing something a stack of convolutions
    does not do by itself: it is routing around walls. Comparing how well each
    explains the measured field is the test for propagation against blur.
    """
    bfs = np.asarray(targets.fields)
    n, objectives, height, width = bfs.shape
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    out = np.zeros_like(bfs)
    for level in range(n):
        for objective in range(objectives):
            flat = int(np.argmin(bfs[level, objective]))
            source = (flat // width, flat % width)
            out[level, objective] = np.maximum(np.abs(rows - source[0]), np.abs(columns - source[1]))
    return out
