"""The field decomposition for networks whose depth replaces the DRC's ticks.

:mod:`goalmisgen.analysis.field_decomp` asks what a planner propagates by
running the same states with an objective erased. It is written against the
DRC, whose working state is a carry stepped over ticks, so propagation is
something you watch happen *in time*.

A ResNet and a ViT have no carry. Whatever spreading they do happens once, down
the stack, so the analogue of a tick is a **layer**: a ResNet stage, or a
transformer block. Everything else is the same measurement -- the same four
observation variants, the same ``A_c``/``D``/``I``, and the same fits, imported
from ``field_decomp`` rather than reimplemented, so the numbers line up with
the DRC's and the route model's tables.

Two consequences worth stating plainly rather than burying in the arithmetic:

* **There is no teacher forcing to do.** A feedforward network's activations
  depend on the observation alone, so one forward pass per variant with the
  agent at its start cell is the whole measurement. The DRC's worry -- that
  erasing an objective moves a free-running agent and desynchronises later
  ticks -- cannot arise.
* **Depth is not time.** A field that reaches further at stage 5 than stage 1
  has been carried there by five layers of convolution, which is a different
  claim from the DRC's recurrence carrying it further on the next tick. The
  comparison that separates spreading from blurring is BFS distance against
  grid distance, and it applies unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.closed_loop import draw_agent


def observations_for(statics: np.ndarray, positions: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """``(4, B, S, S, C)`` statics and ``(B, 2)`` positions -> ``(4, B, C, S, S)``.

    The agent is drawn at the same cell in every variant, and ``gain`` scales
    the observation's channels, so a brightness manipulation and a weight
    manipulation reach the network by the same path -- as in
    :func:`~goalmisgen.analysis.field_decomp.make_grid_fn`. cleanba's
    observations are NCHW.
    """
    out = []
    for variant in range(statics.shape[0]):
        drawn = np.asarray(draw_agent(jnp.asarray(statics[variant]), jnp.asarray(positions)), dtype=np.float32)
        out.append((drawn * gain).transpose(0, 3, 1, 2))
    return np.stack(out)


def depth_grids(reader, params, observations: np.ndarray, seed: int = 0) -> list[np.ndarray]:
    """One forward pass per variant; ``[(4, B, S, S, K)]``, one entry per layer.

    The reader returns the per-cell state of the pass that chose the action, so
    these are the grids the network actually computed on the way to its
    decision, not a separate probe of it.
    """
    policy = reader.policy
    key = jax.random.PRNGKey(seed)
    per_variant: list[tuple[np.ndarray, ...]] = []
    for variant in range(observations.shape[0]):
        batch = observations[variant]
        carry = policy.apply(params, key, batch.shape, method=policy.initialize_carry)
        starts = jnp.ones((batch.shape[0],), dtype=bool)
        _, _, _, _, state = reader.step(params, carry, jnp.asarray(batch), starts, key)
        per_variant.append(tuple(np.asarray(g, dtype=np.float32) for g in state.features))
    n_layers = len(per_variant[0])
    return [np.stack([per_variant[v][layer] for v in range(len(per_variant))]) for layer in range(n_layers)]


def differing_cells(bfs: np.ndarray, grid: np.ndarray, minimum: int = 3) -> np.ndarray:
    """Cells where BFS and grid distance disagree by at least ``minimum``.

    Where a wall forces a detour the two distances come apart, and only there
    does "the field follows the maze" say something different from "the field
    spreads outward". Restricting the comparison to those cells is what E12
    could not do, which is why its BFS and grid fits were too close to call.
    """
    return np.abs(np.asarray(bfs, dtype=np.float64) - np.asarray(grid, dtype=np.float64)) >= minimum


def linear_fit_r2(profile: dict) -> float:
    """R^2 of the straight-line fit whose slope ``field_profile`` reports.

    ``field_profile`` returns the log fit's R^2 but only the linear fit's
    slope, which leaves "exponential decay or cost-subtracted utility"
    undecidable from its output alone. Its per-distance means and counts are
    enough to finish the job, so this reads them rather than refitting the
    field.
    """
    means = np.asarray(profile["means"], dtype=np.float64)
    counts = np.asarray(profile["counts"])
    good = np.isfinite(means) & (counts >= 20) & (means > 0)
    if good.sum() < 3:
        return float("nan")
    distance = np.arange(len(means))[good]
    size = means[good]
    predicted = np.polyval(np.polyfit(distance, size, 1), distance)
    denominator = ((size - size.mean()) ** 2).sum()
    return float(1 - ((size - predicted) ** 2).sum() / denominator) if denominator > 0 else float("nan")
