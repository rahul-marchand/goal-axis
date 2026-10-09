"""E12's field decomposition, for the route model.

The DRC version (:mod:`goalmisgen.analysis.field_decomp`) reads a spatial grid
out of a ConvLSTM's carry. A route model needs no such reassembly: its maze
cells *are* tokens, so the prefix residual stream already is a field, one vector
per cell, and the blocks play the part the DRC's ticks play.

Everything else is deliberately identical, so the tables can be read next to
E12's: the same four observation variants, the same ``A_c = h(only c) -
h(neither)``, the same combination fit, the same profile against BFS distance.

Two things differ, and both are stated in the report rather than hidden here:

* **Erasure is exact.** A colour plane of a route model's observation holds a
  single 1.0 at its objective's cell (plus the value channel, when values are
  not hidden), so deleting an objective is deleting one cell's content, not
  suppressing a spread-out pattern.
* **The source-change control is objective scaling.** E12 calibrated "where does
  a value manipulation land" against brightness, which is a source change by
  construction, and asked whether the axis lands the same way. A route model's
  threshold answers brightness non-monotonically, so brightness cannot be
  *dose-matched* here -- but scaling an objective's own cell still changes only
  that objective's token content, so it remains a valid calibration for *where*
  the energy of a source change ends up. It calibrates the shape of the answer,
  not its size.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.analysis.field_decomp import BOTH, NEITHER, POOR_ONLY, RICH_ONLY, cell_norms
from goalmisgen.edits.targets import FAR
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL, WALL_CHANNEL


def objective_channels(demos, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rich, poor)`` each ``(B, 3)``: colour plane, row, column of that objective."""
    indices = np.asarray(indices)
    values = np.asarray(demos.values)[indices]
    positions = np.asarray(demos.positions)[indices].astype(np.int64)
    features = np.asarray(demos.feature_ids)[indices].astype(np.int64)
    richer = np.argmax(values, axis=1)
    rows = np.arange(len(indices))
    pick = lambda k: np.stack(  # noqa: E731
        [FIRST_FEATURE_CHANNEL + features[rows, k], positions[rows, k, 0], positions[rows, k, 1]], axis=1
    )
    return pick(richer), pick(1 - richer)


def erasure_observations(demos, indices: np.ndarray) -> np.ndarray:
    """``(4, B, S, S, C)``: both objectives, only the richer, only the poorer, neither.

    An objective is erased by zeroing its cell in its own colour plane, and in
    the value channel when the model is shown values at all.
    """
    indices = np.asarray(indices)
    base = np.asarray(demos.observations(indices), dtype=np.float32)
    out = np.stack([base.copy() for _ in range(4)])
    rich, poor = objective_channels(demos, indices)
    rows = np.arange(len(indices))
    value_channel = FIRST_FEATURE_CHANNEL + demos.n_features if not demos.hide_values else None

    def erase(variant: int, who: np.ndarray) -> None:
        out[variant, rows, who[:, 1], who[:, 2], who[:, 0]] = 0.0
        if value_channel is not None:
            out[variant, rows, who[:, 1], who[:, 2], value_channel] = 0.0

    erase(RICH_ONLY, poor)
    erase(POOR_ONLY, rich)
    erase(NEITHER, rich)
    erase(NEITHER, poor)
    return out


def scale_objective(observations: np.ndarray, who: np.ndarray, factor: float) -> np.ndarray:
    """Copy of ``observations`` (any leading axes) with one objective's cell scaled.

    The route model's analogue of E12's brightness: it touches that objective's
    own token content and nothing else, which is what makes it a source change
    by construction.
    """
    out = np.array(observations, dtype=np.float32, copy=True)
    rows = np.arange(who.shape[0])
    if out.ndim == 5:
        out[:, rows, who[:, 1], who[:, 2], who[:, 0]] *= factor
    else:
        out[rows, who[:, 1], who[:, 2], who[:, 0]] *= factor
    return out


def cell_grids(streams: np.ndarray, size: int) -> np.ndarray:
    """``(depths, B, prefix, width) -> (depths, B, S, S, width)``, dropping SEP."""
    streams = np.asarray(streams)
    depths, batch, prefix, width = streams.shape
    cells = size * size
    if prefix < cells:
        raise ValueError(f"prefix {prefix} shorter than {cells} cell tokens")
    return streams[:, :, :cells].reshape(depths, batch, size, size, width)


def free_cells(demos, indices: np.ndarray) -> np.ndarray:
    """``(B, S, S)`` bool: cells an agent could stand in."""
    observations = np.asarray(demos.observations(np.asarray(indices)))
    return observations[..., WALL_CHANNEL] < 0.5


def grid_distance_from(positions: np.ndarray, size: int) -> np.ndarray:
    """``(B, S, S)`` Chebyshev distance from each level's given cell, walls ignored."""
    rows = np.arange(size)[None, :, None]
    columns = np.arange(size)[None, None, :]
    source_row = positions[:, 0][:, None, None]
    source_column = positions[:, 1][:, None, None]
    return np.maximum(np.abs(rows - source_row), np.abs(columns - source_column))


def profile(field: np.ndarray, distance: np.ndarray, mask: np.ndarray, max_distance: int = 12) -> dict:
    """Mean field size per distance, with BOTH fits' quality.

    E12 reported only the exponential fit's R^2, which left "decays like a
    discount" and "falls like a step cost" indistinguishable. Both are returned
    here so the two can be compared directly.
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
    out = {"means": means.tolist(), "counts": counts, "log_slope": float("nan"), "log_r2": float("nan"),
           "linear_slope": float("nan"), "linear_r2": float("nan"), "half_distance": float("nan"),
           "cells": int(usable.sum())}
    if good.sum() < 3:
        return out
    d = np.arange(max_distance + 1)[good]
    observed = means[good]
    r2 = lambda y, fit: float(  # noqa: E731
        1 - ((y - np.polyval(fit, d)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    )
    log_fit = np.polyfit(d, np.log(observed), 1)
    linear_fit = np.polyfit(d, observed, 1)
    out["log_slope"] = float(log_fit[0])
    out["log_r2"] = r2(np.log(observed), log_fit)
    out["linear_slope"] = float(linear_fit[0])
    out["linear_r2"] = r2(observed, linear_fit)
    below = np.where(observed <= observed[0] / 2)[0]
    out["half_distance"] = float(d[below[0]]) if len(below) else float("nan")
    return out


__all__ = [
    "BOTH", "NEITHER", "POOR_ONLY", "RICH_ONLY",
    "cell_grids", "erasure_observations", "free_cells", "grid_distance_from",
    "objective_channels", "profile", "scale_objective",
]
