"""One split of a level pool into the stream D, validation V and test T, for a whole study.

Arms are compared with each other and with their null arm level for level, so
the split must not depend on the arm. It is a function of the study alone: the
rows eligible for every goal of the study (:meth:`GoalFamily.deterministic` over
the whole goal grid, not the arm's own goal) and one fixed shuffle. Only D's
order changes with an ordering replicate.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Split:
    rows: np.ndarray
    """Pool rows used, in the order T, D, V. ``test``, ``train`` and ``valid`` index into these."""

    test: np.ndarray
    train: np.ndarray
    """In the order the code reads it."""

    valid: np.ndarray


def study_split(eligible: np.ndarray, n: int, n_valid: int, n_test: int, seed: int = 0, ordering: int = 0) -> Split:
    """The first ``n_test + n + n_valid`` of ``eligible`` under a fixed shuffle, as T, then D, then V.

    T first and V last, so changing V's size for a noise replicate leaves T and
    D alone. ``ordering`` permutes D only (0 keeps the shuffle's order).
    """
    need = n + n_valid + n_test
    if len(eligible) < need:
        raise ValueError(f"the pool has {len(eligible):,} eligible levels, fewer than the {need:,} asked for")
    rows = np.asarray(eligible)[np.random.default_rng(seed).permutation(len(eligible))[:need]]
    test = np.arange(n_test)
    train = np.arange(n_test, n_test + n)
    valid = np.arange(n_test + n, need)
    if ordering:
        train = train[np.random.default_rng(ordering).permutation(len(train))]
    return Split(rows=rows, test=test, train=train, valid=valid)
