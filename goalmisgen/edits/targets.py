"""What a threshold-``tau`` agent would do on a batch of levels, as teacher-forcing targets.

The cheapest-edit search asks for the smallest change to a network that makes
it behave like an agent with a different exchange rate. It needs a
differentiable stand-in for "behaves like threshold ``tau``", and the solver
supplies one: on every level, the two shortest routes (to the richer and the
poorer objective), and at every cell on them the moves that shorten each.

Four choices here are deliberate, and each one closes a way for the search to
spend its budget on something other than the threshold.

**Set-valued targets.** Where several moves are equally short, any of them is
right. The loss is on ``q(A) = sum over a in A of pi(a)``, never on one
solver-chosen move, so the edit is not asked to learn the solver's tie-breaking.

**Decide once, at the fork.** The routes share a corridor up to the last common
cell, and the distance gap is the same at every cell of it (both distances fall
by one per step). The target there is the soft mixture
``-p log q(A_rich) - (1 - p) log q(A_poor)``, which on a shared corridor, where
both sets hold the corridor move, is plain navigation. Past the fork each branch
just heads for its goal, weighted by the probability the target agent would be
on it. A state-feedback rule that re-decided at every cell would ask a committed
agent to turn round, which is a different behaviour from a different threshold.

**Soft labels.** ``p = sigmoid((tau - gap) / scale)`` with the *base* agent's
own logistic scale, so that hitting the target moves the threshold without also
sharpening it. With hard labels the cheapest edit for a noisy chooser would
include making it less noisy.

**The fork is weighted apart.** A route has one fork and a dozen navigation
steps the base already gets right; a plain mean would give the choice almost no
gradient. ``role`` marks the fork so the loss can average the two separately.

This module is JAX-free, like the dataset and demo modules, so it can be built
in a spawned worker without importing a runtime.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from goalmisgen.envs.level import Level
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL, WALL_CHANNEL
from goalmisgen.envs.solver import MOVES, UNREACHABLE, distance_field, path_to_objective, walls_blocking_other_objectives

PAD, PREFIX, FORK, BRANCH = 0, 1, 2, 3
"""``role`` of a step: padding, shared corridor before the fork, the fork, one branch after it."""

FAR = 10_000
"""Distance recorded for walls, padding and cells cut off from an objective."""

RICH, POOR = 0, 1
"""Branch order along axis 1 of every per-branch array."""


@dataclasses.dataclass(frozen=True)
class RouteTargets:
    """Both branches of every level, padded to a common length ``T``.

    Shapes: ``N`` levels, 2 branches (rich, poor), ``T`` steps, ``S`` maze side.
    """

    static: np.ndarray  # (N, S, S, C) float32 - the observation with no agent drawn
    positions: np.ndarray  # (N, 2, T, 2) int32 - agent cell at each step
    moves: np.ndarray  # (N, 2, T) int32 - the move taken from that cell along the branch, -1 on padding
    lengths: np.ndarray  # (N, 2) int32 - moves on each branch
    toward_rich: np.ndarray  # (N, 2, T, 4) bool - moves that shorten the route to the richer objective
    toward_poor: np.ndarray  # (N, 2, T, 4) bool
    role: np.ndarray  # (N, 2, T) int8 - PAD / PREFIX / FORK / BRANCH, prefix and fork on the rich branch only
    gap: np.ndarray  # (N,) float32 - d_rich - d_poor from the start
    rich_above: np.ndarray  # (N,) bool - the richer objective is on a higher row than the poorer
    distances: np.ndarray  # (N, 2) int32 - d_rich, d_poor
    rich_feature: np.ndarray  # (N,) int8 - colour of the richer objective
    fields: np.ndarray  # (N, 2, S, S) int16 - distance to the richer / poorer objective, FAR where unreachable or padding

    def __len__(self) -> int:
        return len(self.gap)

    @property
    def steps(self) -> int:
        return self.positions.shape[2]

    def subset(self, rows: np.ndarray) -> "RouteTargets":
        return RouteTargets(**{f.name: getattr(self, f.name)[rows] for f in dataclasses.fields(self)})


def shortening_moves(field: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    """(4,) bool: moves from ``cell`` onto a neighbour one step closer in ``field``."""
    height, width = field.shape
    out = np.zeros(len(MOVES), dtype=bool)
    here = field[cell]
    if here == UNREACHABLE:
        return out
    for action, (dr, dc) in enumerate(MOVES):
        r, c = cell[0] + dr, cell[1] + dc
        if 0 <= r < height and 0 <= c < width and field[r, c] != UNREACHABLE and field[r, c] == here - 1:
            out[action] = True
    return out


def static_observation(level: Level, size: int, n_channels: int) -> np.ndarray:
    """The encoder's observation with the agent channel left empty; padding counts as wall."""
    height, width = level.shape
    obs = np.zeros((size, size, n_channels), dtype=np.float32)
    obs[:, :, WALL_CHANNEL] = 1.0
    obs[:height, :width, WALL_CHANNEL] = level.walls.astype(np.float32)
    for objective in level.objectives:
        obs[objective.position[0], objective.position[1], FIRST_FEATURE_CHANNEL + objective.feature_id] = 1.0
    return obs


def usable(level: Level) -> bool:
    """Two objectives of different value, both reachable, neither under the agent."""
    if len(level.objectives) != 2 or level.objectives[0].value == level.objectives[1].value:
        return False
    return all(path_to_objective(level, k) is not None and len(path_to_objective(level, k)) > 1 for k in range(2))


def route_targets(levels: Sequence[Level], size: int, n_channels: int = 4) -> RouteTargets:
    """Teacher-forcing geometry for ``levels``. Every level must pass :func:`usable`."""
    rows = []
    for level in levels:
        if not usable(level):
            raise ValueError("route_targets needs usable levels; filter with usable() first")
        rich = int(np.argmax([o.value for o in level.objectives]))
        order = (rich, 1 - rich)
        fields = [distance_field(walls_blocking_other_objectives(level, k), level.objectives[k].position) for k in order]
        paths = [path_to_objective(level, k) for k in order]
        fork = 0
        for t, (a, b) in enumerate(zip(paths[0], paths[1])):
            if a != b:
                break
            fork = t
        rows.append((level, order, fields, paths, fork))

    steps = max(len(path) - 1 for _, _, _, paths, _ in rows for path in paths)
    n = len(rows)
    out = dict(
        static=np.zeros((n, size, size, n_channels), dtype=np.float32),
        positions=np.zeros((n, 2, steps, 2), dtype=np.int32),
        moves=np.full((n, 2, steps), -1, dtype=np.int32),
        lengths=np.zeros((n, 2), dtype=np.int32),
        toward_rich=np.zeros((n, 2, steps, 4), dtype=bool),
        toward_poor=np.zeros((n, 2, steps, 4), dtype=bool),
        role=np.zeros((n, 2, steps), dtype=np.int8),
        gap=np.zeros(n, dtype=np.float32),
        rich_above=np.zeros(n, dtype=bool),
        distances=np.zeros((n, 2), dtype=np.int32),
        rich_feature=np.zeros(n, dtype=np.int8),
        fields=np.full((n, 2, size, size), FAR, dtype=np.int16),
    )
    move_index = {move: index for index, move in enumerate(MOVES)}
    for i, (level, order, fields, paths, fork) in enumerate(rows):
        out["static"][i] = static_observation(level, size, n_channels)
        d = [len(path) - 1 for path in paths]
        out["distances"][i] = d
        out["rich_feature"][i] = level.objectives[order[0]].feature_id
        height, width = level.shape
        for branch, field in enumerate(fields):
            out["fields"][i, branch, :height, :width] = np.where(field == UNREACHABLE, FAR, field)
        out["gap"][i] = d[0] - d[1]
        out["rich_above"][i] = level.objectives[order[0]].position[0] < level.objectives[order[1]].position[0]
        for branch, path in enumerate(paths):
            length = len(path) - 1
            out["lengths"][i, branch] = length
            for t in range(length):
                cell = path[t]
                out["positions"][i, branch, t] = cell
                out["moves"][i, branch, t] = move_index[(path[t + 1][0] - cell[0], path[t + 1][1] - cell[1])]
                out["toward_rich"][i, branch, t] = shortening_moves(fields[RICH], cell)
                out["toward_poor"][i, branch, t] = shortening_moves(fields[POOR], cell)
                if t < fork:
                    out["role"][i, branch, t] = PREFIX if branch == RICH else PAD
                elif t == fork:
                    out["role"][i, branch, t] = FORK if branch == RICH else PAD
                else:
                    out["role"][i, branch, t] = BRANCH
            # Positions past the end stay at the last cell so an observation can still be drawn.
            out["positions"][i, branch, length:] = path[length - 1]
    return RouteTargets(**out)

