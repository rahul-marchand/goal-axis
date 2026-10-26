"""How a model scores data: the one place the metric touches a model's forward pass.

A :class:`Scorer` turns rows of a demonstration set into model inputs and gives
two things of them: the training loss (what the learner descends) and the code
length of each example in bits (what the prequential code sums). For the code to
be the trainer's own objective, ``loss`` should be the mean per-unit
cross-entropy whose sum, in bits, ``bits`` reports per example.

``units`` is what an example's bits are spread over (tokens of a route, say),
for references that compare goals whose examples differ in length.

A :class:`Sampler` is a scorer that can also draw examples from a model, which
a coding mechanism that trains on a teacher's samples (requential coding)
needs. Inputs (mazes, prompts) come from the data; only outputs are drawn.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Scorer(Protocol):
    def batch(self, data: Any, indices: np.ndarray) -> tuple:
        """Model inputs for rows ``indices`` of ``data``, as device arrays."""
        ...

    def loss(self, params, *batch) -> Any:
        """The training loss on one batch. Pure, so a learner can differentiate and jit it."""
        ...

    def bits(self, params, *batch) -> Any:
        """``-log2 p`` of each example in the batch, shape ``(B,)``. Called often, so jit it."""
        ...

    def units(self, data: Any, indices: np.ndarray) -> np.ndarray:
        """How many coded units (tokens) each example has."""
        ...


class Sampler(Scorer, Protocol):
    def sample(self, params, data: Any, indices: np.ndarray, seed: int) -> Any:
        """One output per row ``indices`` of ``data``, drawn from ``params``, as data this scorer reads by position.

        Reproducible from ``seed`` and the row ids alone (the decoder of a
        requential code replays it). The result has ``truncated``, per example,
        marking draws the sampler had to cut short, which are not exact draws.
        """
        ...


def example_bits(scorer: Scorer, params, data, indices, chunk: int = 512) -> np.ndarray:
    """``-log2 p`` under ``params`` of each row in ``indices``, in float64."""
    indices = np.asarray(indices)
    out = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), chunk):
        rows = indices[start : start + chunk]
        out[start : start + len(rows)] = np.asarray(scorer.bits(params, *scorer.batch(data, rows)))
    return out
