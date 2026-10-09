"""What a normalisation does to a colour gain.

A gain on colour ``c`` scales that colour's contribution to the first layer and
leaves its direction alone (:mod:`goalmisgen.analysis.input_gain`). Whether the
rest of the network can see that depends on what the first thing downstream
preserves.

``nn.LayerNorm`` centres a token and divides by its own standard deviation, so
it is invariant to adding any multiple of the ones vector and to scaling the
token. Write the centred token ``h~`` and a perturbation ``d``; to first order

    LN(h + d) - LN(h) = (d~ - h~ <h~, d~> / ||h~||^2) / sigma,

so the part of ``d`` along ``1`` and the part along ``h~`` are both discarded
and only the rest -- the part that turns the token -- reaches the next block.
:func:`decompose` splits a perturbation into those three pieces, and
:func:`survival` measures the whole thing end to end by running the network
twice, without assuming the linearisation.

The DRC and the ResNet are configured with ``IdentityNorm`` and their first
nonlinearity is a gate, not a normalisation, so nothing there removes a length
change: the prediction is a survival near one for those two and near zero for
the ViT and the route model, whose blocks are pre-LayerNorm.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Decomposition:
    """A perturbation split by what a LayerNorm keeps, as shares of ``||d||^2``."""

    ones: float
    """Along the all-ones vector: removed by centring."""

    magnitude: float
    """Along the centred token: removed by dividing by the standard deviation."""

    direction: float
    """The rest: what reaches the next block."""

    @property
    def killed(self) -> float:
        return self.ones + self.magnitude


def decompose(tokens: np.ndarray, delta: np.ndarray) -> Decomposition:
    """Split ``delta`` at each token into ones / magnitude / direction shares.

    ``tokens`` and ``delta`` are ``(..., width)``; the shares are pooled over
    every token, weighted by ``||delta||^2``, so tokens the perturbation does
    not touch contribute nothing.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if tokens.shape != delta.shape:
        raise ValueError(f"tokens {tokens.shape} and delta {delta.shape} differ")
    width = tokens.shape[-1]

    ones = delta.mean(axis=-1, keepdims=True)
    along_ones = np.square(ones).sum(axis=-1) * width

    centred_tokens = tokens - tokens.mean(axis=-1, keepdims=True)
    centred_delta = delta - ones
    squared = np.square(centred_tokens).sum(axis=-1)
    projection = np.where(squared > 0, (centred_tokens * centred_delta).sum(axis=-1), 0.0)
    along_token = np.where(squared > 0, np.square(projection) / np.maximum(squared, 1e-30), 0.0)

    total = np.square(delta).sum(axis=-1)
    kept = np.maximum(total - along_ones - along_token, 0.0)
    scale = max(float(total.sum()), 1e-30)
    return Decomposition(
        ones=float(along_ones.sum()) / scale,
        magnitude=float(along_token.sum()) / scale,
        direction=float(kept.sum()) / scale,
    )


def layer_norm(
    x: np.ndarray, scale: np.ndarray | None = None, bias: np.ndarray | None = None, epsilon: float = 1e-6
) -> np.ndarray:
    """``nn.LayerNorm`` in numpy, for applying a checkpoint's own first norm by hand."""
    x = np.asarray(x, dtype=np.float64)
    centred = x - x.mean(axis=-1, keepdims=True)
    normed = centred / np.sqrt(np.square(centred).mean(axis=-1, keepdims=True) + epsilon)
    if scale is not None:
        normed = normed * np.asarray(scale, dtype=np.float64)
    if bias is not None:
        normed = normed + np.asarray(bias, dtype=np.float64)
    return normed


def relative_change(base: np.ndarray, perturbed: np.ndarray) -> float:
    """``||perturbed - base|| / ||base||``, pooled over everything."""
    base = np.asarray(base, dtype=np.float64)
    delta = np.asarray(perturbed, dtype=np.float64) - base
    denominator = float(np.linalg.norm(base))
    return float(np.linalg.norm(delta)) / denominator if denominator > 0 else float("nan")


def survival(before: tuple[np.ndarray, np.ndarray], after: tuple[np.ndarray, np.ndarray]) -> float:
    """How much of a relative change is left after a stage.

    Each argument is ``(base, perturbed)`` read at one point in the network.
    One means the stage passed the perturbation on at the same relative size;
    zero means it removed it.
    """
    downstream = relative_change(*after)
    upstream = relative_change(*before)
    return downstream / upstream if upstream > 0 else float("nan")
