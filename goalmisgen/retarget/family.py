"""What a goal is: the one interface between the metric and an environment.

A goal family turns a pool of levels, demonstrated at the agent's own goal (the
*null* goal), into demonstrations of the expert at another goal on the same
levels. It also says which levels the code may use: the metric reads ``L* = 0``
off a deterministic expert, so levels on which some goal of the study leaves the
expert indifferent are dropped for every arm alike. Where the expert is not
deterministic the code would have to coarsen to what the goal determines (a
class code) instead; no family here needs that yet.

Goals are JSON scalars or lists (an exchange rate, a set of values), so a run's
record can name its goal.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

Goal = Any


class GoalFamily(Protocol):
    null: Goal
    """The agent's own goal: the pool is demonstrated at it, and its arm is the null arm."""

    def check_pool(self, pool) -> None:
        """Raise unless ``pool`` is demonstrated at the null goal."""
        ...

    def deterministic(self, pool, goals: Sequence[Goal]) -> np.ndarray:
        """Pool rows on which the expert's demonstration is unique at every goal in ``goals``."""
        ...

    def demonstrations(self, pool, rows: np.ndarray, goal: Goal) -> Any:
        """The expert at ``goal`` on pool rows ``rows``, in that order, as data a :class:`Scorer` reads."""
        ...

    def changed(self, pool, rows: np.ndarray, data) -> np.ndarray:
        """Per row, whether the goal's expert decides differently from the null goal's (``data`` from :meth:`demonstrations`)."""
        ...
