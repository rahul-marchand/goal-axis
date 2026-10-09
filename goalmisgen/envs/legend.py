"""An observation where colour says nothing and a legend in the border says which objective pays.

Every hidden-value agent in the project reads what an objective is worth off its
colour, and colour sits *on the objective's cell*. The value axis those agents
grow is concentrated in the input convolution and is shaped like a gain on each
colour plane. That leaves open whether a gain is simply the cheapest thing an
input-level value cue permits, or what these networks do regardless. This
encoding is the positive control for the second reading: the cue to value is
moved somewhere a per-plane gain cannot reach the objective.

Four channels, the same count as the hidden-value encoding, one row and one
column wider::

    0  wall        padding counts as wall, as ever
    1  agent
    2  objectives  every objective, on one shared plane: no colour
    3  legend      two marks in the padding border: (r, side-1) and (side-1, c),
                   where (r, c) is the objective that carries feature 0

Feature 0 is what :class:`~goalmisgen.envs.features.CorrelatedFeatures` paints
on the richest objective, so at rho=1 the legend points at the richer objective
exactly where colour 0 used to sit on it. Nothing in the level changes -- only
where the cue is drawn -- and the fixed values stay hidden, so a fine-tuning
sweep manufactures a value axis the same way it does for ``novalue11g1``.

Why a border rather than a mark near the objective: a mark next to the cell is
still inside one 3x3 kernel of it, so a first-layer gain on the mark would be a
gain on the objective in all but name. Row and column marks in the border meet
only at the objective and are up to ``side - 1`` cells away from it. Binding
them to a location is a conjunction across the maze, which no linear input layer
computes, so value has to join the objective's representation downstream. A
per-plane gain on the objectives plane scales both objectives alike and cannot
favour one of them.

The legend is task input, not ground truth: it is the same bit colour carried,
and the network still has to find distances for itself.

**Activation is a config field**, ``MazeConfig(observation="legend")``, so it is
written into every run's ``cfg.json`` and every reader rebuilds the observation
the weights were trained on. (It used to be a ``sitecustomize`` import hook that
patched ``MazeConfig.encoder`` process-wide, which no run recorded; the runs it
made are read through their ``OBSERVATION`` file instead, see
:func:`goalmisgen.configs.saved.train_env`.) A checkpoint trained one way cannot
be silently read the other: the observation is 12x12 here against 11x11, so the
parameter shapes disagree and loading fails.

This module is deliberately outside ``dataset.CONTENT_MODULES``: it changes how
a stored level is drawn, never what it contains, so existing level datasets
serve it unchanged.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np

from goalmisgen.envs.level import Level, Position
from goalmisgen.envs.observation import AGENT_CHANNEL, WALL_CHANNEL

OBJECTIVE_CHANNEL = 2
LEGEND_CHANNEL = 3
LEGEND_FEATURE = 0
"""The feature whose holder the legend points at: the richest objective at rho=1."""


@dataclasses.dataclass(frozen=True)
class LegendEncoder:
    """Objectives on one uncoloured plane; the feature-0 holder named by row and column marks in the border."""

    max_size: int
    n_features: int = 2

    @property
    def side(self) -> int:
        return self.max_size + 1

    @property
    def n_channels(self) -> int:
        return 4

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.side, self.side, self.n_channels)

    def __post_init__(self) -> None:
        if self.max_size < 3:
            raise ValueError(f"max_size must be at least 3, got {self.max_size}")
        if self.n_features < 2:
            raise ValueError("a legend distinguishes objectives, so it needs at least two features")

    def reset(self, level: Level) -> None:
        """Stateless."""

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def encode(self, level: Level, agent_position: Position) -> np.ndarray:
        height, width = level.shape
        if height > self.max_size or width > self.max_size:
            raise ValueError(f"level of shape {level.shape} does not fit inside a {self.side}x{self.side} legend observation")
        observation = np.zeros(self.shape, dtype=np.float32)
        observation[:, :, WALL_CHANNEL] = 1.0
        observation[:height, :width, WALL_CHANNEL] = level.walls.astype(np.float32)

        row, col = agent_position
        if not (0 <= row < height and 0 <= col < width) or level.is_wall(agent_position):
            raise ValueError(f"agent_position {agent_position} is not a free cell of the level")
        observation[row, col, AGENT_CHANNEL] = 1.0

        marked = [o for o in level.objectives if o.feature_id == LEGEND_FEATURE]
        if len(marked) != 1:
            raise ValueError(f"expected exactly one objective with feature {LEGEND_FEATURE}, found {len(marked)}")
        for objective in level.objectives:
            if objective.feature_id >= self.n_features:
                raise ValueError(f"objective has feature_id={objective.feature_id} but n_features={self.n_features}")
            observation[objective.position[0], objective.position[1], OBJECTIVE_CHANNEL] = 1.0

        # The border row and column lie outside every maze (side = max_size + 1),
        # so a mark there can never coincide with an objective.
        r, c = marked[0].position
        observation[r, self.side - 1, LEGEND_CHANNEL] = 1.0
        observation[self.side - 1, c, LEGEND_CHANNEL] = 1.0
        return observation


def legend_encoder(config) -> LegendEncoder:
    """The legend twin of ``MazeConfig.encoder`` for a given configuration."""
    if config.value_encoding != "none":
        raise ValueError(
            "the legend encoding is the hidden-value control; a value channel would hand the "
            f"network the very cue the legend exists to relocate (value_encoding={config.value_encoding!r})"
        )
    if config.pad_size is not None and config.pad_size != config.max_size + 1:
        raise ValueError(f"the legend fixes the observation at max_size + 1; pad_size={config.pad_size} was asked for")
    return LegendEncoder(max_size=config.max_size, n_features=config.n_features or config.n_objectives)
