"""Demonstrations of several goals at once, each tagged with its goal.

Training data for a route model with goal vectors (``ModelConfig.n_goals``):
one demonstration set per goal, here per pair of objective values and so per
exchange rate ``tau``, drawn from as one set. A demonstration's goal id is the
index of the set it came from, and ids follow the order the sets are given in.
The sets either hold the same levels at different values or split one pool
between them (``GoalMixture.load``'s ``levels``).

With the values hidden, nothing in an observation says which goal applies, so
a model fitting this mixture must carry all of it in the goal vector.
"""

from __future__ import annotations

import pathlib
from typing import Literal, Sequence

import numpy as np

from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.mixture import MixedDemoSet


def exchange_rate(values: Sequence[float], step_penalty: float) -> float:
    """Steps the richer objective is worth: ``(v_rich - v_poor) / step_penalty``."""
    richer, poorer = sorted(values, reverse=True)
    return (richer - poorer) / step_penalty


class GoalMixture(MixedDemoSet):
    """One part per goal; ``goal_ids`` names the part each demonstration came from."""

    def goal_ids(self, indices) -> np.ndarray:
        indices = np.asarray(indices)
        ids = np.empty(len(indices), dtype=np.int32)
        for k, rows, _ in self._split(indices):
            ids[rows] = k
        return ids

    @property
    def taus(self) -> tuple[float, ...]:
        return tuple(exchange_rate(p.meta["values"], float(p.meta["step_penalty"])) for p in self.parts)

    @property
    def meta(self) -> dict:
        return {
            **self.parts[0].meta,
            "goals": [
                {"path": None if p.path is None else str(p.path), "values": list(p.meta["values"]), "tau": tau}
                for p, tau in zip(self.parts, self.taus)
            ],
        }

    @classmethod
    def load(
        cls,
        paths: Sequence[str | pathlib.Path],
        hide_values: bool = True,
        levels: Literal["same", "disjoint"] = "same",
    ) -> "GoalMixture":
        """The sets at ``paths``, in that order, one goal each.

        ``levels`` says how the sets' levels relate, and is checked rather than
        trusted, because a mixture built the wrong way trains without complaint:

        - ``"same"``: every set holds the same levels in the same order, so goals
          differ only in values (``bcgv11``, the o1 sweep's 25 sets). Checked on
          the levels themselves, not the source fingerprint: that covers the
          sampler's objective values, so each values tag of one pool has its own
          fingerprint even though fixed values consume no randomness and the
          layouts are identical (``make_demo_subset.py`` relies on this).
        - ``"disjoint"``: the sets partition part of one pool, each level shown
          at one goal only (:func:`split_by_goal`). This keeps the pool's full
          variety of mazes, which ``"same"`` divides by the number of goals. The
          sets must share a source fingerprint, since only then do equal level
          indices mean equal levels.
        """
        if levels not in ("same", "disjoint"):
            raise ValueError(f"levels must be 'same' or 'disjoint', got {levels!r}")
        parts = tuple(DemoSet.load(path, hide_values=hide_values) for path in paths)
        if levels == "same":
            _check_same_levels(paths, parts)
        else:
            _check_disjoint_levels(paths, parts)
        taus = [exchange_rate(p.meta["values"], float(p.meta["step_penalty"])) for p in parts]
        if len(set(np.round(taus, 9))) != len(taus):
            raise ValueError(f"two goal sets share an exchange rate: {taus}")
        return cls(parts)


def _check_same_levels(paths: Sequence, parts: Sequence[DemoSet]) -> None:
    first = parts[0]
    for path, part in zip(paths, parts):
        same = len(part) == len(first) and all(
            np.array_equal(np.asarray(getattr(part, name)), np.asarray(getattr(first, name)))
            for name in ("level_index", "walls_packed", "agent", "positions")
        )
        if not same:
            raise ValueError(f"{path} holds different levels from {paths[0]}")


def _check_disjoint_levels(paths: Sequence, parts: Sequence[DemoSet]) -> None:
    fingerprints = {p.meta.get("source_fingerprint") for p in parts}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError(f"disjoint goal sets must come from one level pool; source fingerprints {fingerprints}")
    seen: set[int] = set()
    for path, part in zip(paths, parts):
        mine = np.asarray(part.level_index)
        if len(np.unique(mine)) != len(mine):
            raise ValueError(f"{path} repeats a level")
        shared = seen.intersection(mine.tolist())
        if shared:
            raise ValueError(f"{path} shares {len(shared)} levels with an earlier goal set")
        seen.update(mine.tolist())


def with_base_goal_vector(params, model, vector) -> dict:
    """``params`` with the base goal's vector replaced by ``vector``.

    Every decoder and scorer uses the base slot, so this runs the agent at
    ``vector``: another goal's vector, a point between two, an oracle setting.
    """
    table = np.array(params["params"]["goal_embedding"])
    table[model.config.base_goal] = vector
    return {**params, "params": {**params["params"], "goal_embedding": table}}


def split_by_goal(n: int, n_goals: int, seed: int) -> list[np.ndarray]:
    """Each of ``n`` demonstrations assigned one goal uniformly at random: the rows of each goal, in order.

    The partition behind a ``levels="disjoint"`` mixture. Drawn independently
    per demonstration rather than dealt out evenly, so a goal's share is
    binomial; with hundreds of thousands of levels that is within a percent.
    """
    goal = np.random.default_rng(seed).integers(n_goals, size=n)
    return [np.flatnonzero(goal == k) for k in range(n_goals)]
