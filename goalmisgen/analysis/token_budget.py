"""How much of a route model's token is the maze content, and how much is position.

A gain or a rotation on ``cell_in`` can only move the part of a token that came
through ``cell_in``. The route model builds a cell token as

    x = cell_in(cell) + row_embedding[row] + col_embedding[col]

so the content term is at most rank ``n_channels`` -- four -- inside a
``d_model``-dimensional token, and the two position terms are learned and free to
be as large as training wants them. If they dominate, an input-layer edit turns
the token hardly at all, the first LayerNorm rescales what little is left, and
the value knob has to live downstream. That is the *dilution* account of why both
two-parameter knobs are weak on the route model, and it is a claim about a
quantity, so it can be measured rather than argued.

:func:`budget` splits the tokens three ways -- the colour channels' contribution,
the rest of the content (walls, the agent, and ``cell_in``'s bias), and position
-- and reports each as a share of the token, of the token's variation across
cells, and of what survives the model's own first normalisation. The last of
those is the one to read: it is the fraction of the post-norm token an edit at
``cell_in`` can reach, and the dilution account predicts it falls as ``d_model``
grows while the axis stays in the blocks.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from goalmisgen.analysis.norm_survival import layer_norm


@dataclasses.dataclass(frozen=True)
class Budget:
    """Shares of a token attributable to one of its additive parts."""

    norm: float
    """Share of the pooled ``||x||^2``."""

    variation: float
    """Share of the pooled ``||x - mean_x||^2``, the part that distinguishes cells."""

    after_norm: float
    """Relative change in the post-LayerNorm token when this part is removed."""


def _share(part: np.ndarray, whole: np.ndarray) -> float:
    """``<part, whole> / ||whole||^2``: the part's signed share, so the parts sum to one.

    A plain ``||part||^2 / ||whole||^2`` double-counts whenever the parts are
    correlated, and the three parts of a token are not orthogonal.
    """
    denominator = float(np.square(whole).sum())
    return float((part * whole).sum()) / denominator if denominator > 0 else float("nan")


def budget(parts: dict[str, np.ndarray], scale: np.ndarray | None = None, bias: np.ndarray | None = None) -> dict[str, Budget]:
    """Split tokens into the shares carried by each named additive part.

    ``parts`` maps a name to an array shaped like the tokens; they must sum to
    the tokens. ``scale`` and ``bias`` are the first LayerNorm's parameters, used
    to measure each part's reach after normalisation.
    """
    if not parts:
        raise ValueError("no parts to account for")
    names = list(parts)
    stacked = [np.asarray(parts[name], dtype=np.float64) for name in names]
    shapes = {array.shape for array in stacked}
    if len(shapes) != 1:
        raise ValueError(f"parts have different shapes: {sorted(shapes)}")
    tokens = sum(stacked)
    centred = tokens - tokens.mean(axis=tuple(range(tokens.ndim - 1)), keepdims=True)
    normed = layer_norm(tokens, scale, bias)
    denominator = float(np.linalg.norm(normed))

    out = {}
    for name, part in zip(names, stacked, strict=True):
        without = layer_norm(tokens - part, scale, bias)
        reach = float(np.linalg.norm(without - normed)) / denominator if denominator > 0 else float("nan")
        part_centred = part - part.mean(axis=tuple(range(part.ndim - 1)), keepdims=True)
        out[name] = Budget(
            norm=_share(part, tokens),
            variation=_share(part_centred, centred),
            after_norm=reach,
        )
    return out


def cell_token_parts(params, cells: np.ndarray, size: int, colour: tuple[int, ...]) -> dict[str, np.ndarray]:
    """The additive parts of every cell token of a route model.

    ``cells`` is ``(batch, n_cells, n_channels)`` as the model reshapes its
    observation, and ``params`` the unravelled ``{"params": ...}`` tree.
    """
    tree = params["params"] if "params" in params else params
    kernel = np.asarray(tree["cell_in"]["kernel"], dtype=np.float64)
    cells = np.asarray(cells, dtype=np.float64)
    n_cells = cells.shape[-2]
    rows = np.arange(n_cells) // size
    cols = np.arange(n_cells) % size

    mask = np.zeros(cells.shape[-1], dtype=bool)
    mask[list(colour)] = True
    position = np.asarray(tree["row_embedding"], dtype=np.float64)[rows] + np.asarray(tree["col_embedding"], dtype=np.float64)[cols]
    parts = {
        "colour": cells[..., mask] @ kernel[mask],
        "other content": cells[..., ~mask] @ kernel[~mask] + np.asarray(tree["cell_in"].get("bias", 0.0), dtype=np.float64),
        "position": np.broadcast_to(position, cells.shape[:-1] + position.shape[-1:]),
    }
    return {name: np.ascontiguousarray(value) for name, value in parts.items()}
