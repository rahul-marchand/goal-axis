"""What an edit breaks besides the threshold: navigation alone, and choices away from the threshold.

The cheapest-edit search is scored on moving the threshold. An edit confined to
the readout could move it by damaging something else -- steering toward one
colour by making the agent worse at reaching the other, or by turning a sharp
chooser into a noisy one whose logistic crossing happens to land on the target.
Two readouts guard against that, and neither involves a choice at the threshold:

* **One objective at a time.** Each two-objective level is re-posed with only
  the richer or only the poorer objective left in it. There is nothing to
  choose, so a harmless value edit leaves success and route length where the
  base had them. Distances are recomputed with the other objective gone, since
  it no longer blocks a corridor.
* **Choices far from the threshold.** Levels whose gap sits at least ``margin``
  steps from the threshold have a clear right answer for a threshold agent;
  accuracy there is what a sharp chooser keeps and a noisy one loses.

JAX-free, like :mod:`goalmisgen.analysis.route_targets`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from goalmisgen.analysis.route_targets import static_observation
from goalmisgen.envs.level import Level
from goalmisgen.envs.solver import UNREACHABLE, distance_field


@dataclasses.dataclass(frozen=True)
class SingleObjective:
    """Levels holding one objective each. Shapes: ``N`` levels, ``S`` maze side, ``C`` channels."""

    static: np.ndarray  # (N, S, S, C) float32
    start: np.ndarray  # (N, 2) int32
    distance: np.ndarray  # (N,) int32 - shortest route from the start with only this objective present
    feature: np.ndarray  # (N,) int8 - colour of the objective that was kept

    def __len__(self) -> int:
        return len(self.distance)


def single_objective(levels: Sequence[Level], keep: str, size: int, n_channels: int = 4) -> SingleObjective:
    """Each level with only its ``keep`` (``"rich"`` or ``"poor"``) objective left in it."""
    if keep not in ("rich", "poor"):
        raise ValueError(f"keep must be 'rich' or 'poor', got {keep!r}")
    static, start, distance, feature = [], [], [], []
    for level in levels:
        values = [o.value for o in level.objectives]
        index = int(np.argmax(values) if keep == "rich" else np.argmin(values))
        objective = level.objectives[index]
        alone = Level(walls=level.walls, agent_start=level.agent_start, objectives=(objective,))
        d = int(distance_field(level.walls, objective.position)[level.agent_start])
        if d == UNREACHABLE:
            raise ValueError("an unreachable objective has no route to score")
        static.append(static_observation(alone, size, n_channels))
        start.append(level.agent_start)
        distance.append(d)
        feature.append(objective.feature_id)
    return SingleObjective(
        static=np.stack(static).astype(np.float32),
        start=np.asarray(start, dtype=np.int32),
        distance=np.asarray(distance, dtype=np.int32),
        feature=np.asarray(feature, dtype=np.int8),
    )


def navigation(single: SingleObjective, reached: np.ndarray, steps: np.ndarray) -> dict:
    """Success, shortest-route rate and mean excess steps on single-objective levels.

    Excess is ``steps - distance`` over successful episodes; a timeout counts
    against success only.
    """
    reached, steps = np.asarray(reached), np.asarray(steps)
    success = reached == single.feature
    excess = (steps - single.distance)[success]
    return {
        "success": float(success.mean()),
        "shortest": float(np.mean(excess == 0)) if success.any() else float("nan"),
        "excess_steps": float(excess.mean()) if success.any() else float("nan"),
    }


def decisions_away_from(gap: np.ndarray, rich_feature: np.ndarray, reached: np.ndarray, tau: float, margin: float) -> dict:
    """Accuracy of a threshold-``tau`` agent on finished levels at least ``margin`` steps from ``tau``.

    The right answer there is the richer objective when ``gap < tau`` and the
    poorer when ``gap > tau``.
    """
    gap, reached = np.asarray(gap, dtype=float), np.asarray(reached)
    rows = (reached >= 0) & (np.abs(gap - tau) >= margin)
    if not rows.any() or not np.isfinite(tau):
        return {"accuracy": float("nan"), "n": int(rows.sum())}
    took_rich = reached[rows] == np.asarray(rich_feature)[rows]
    return {"accuracy": float(np.mean(took_rich == (gap[rows] < tau))), "n": int(rows.sum())}


def relative_change(delta: np.ndarray, base_flat: np.ndarray, groups: dict[str, np.ndarray], order: Sequence[str]) -> dict:
    """``|delta|`` inside each module relative to the base's own weights there, and over the edit's support.

    Raw L2 is not comparable across modules of different scale, so "cheapest"
    between a readout edit and an input edit needs a unit that is.
    """
    out = {}
    support = np.zeros(len(delta), dtype=bool)
    for name in order:
        rows = groups[name]
        d = float(np.linalg.norm(delta[rows]))
        if d > 0:
            support[rows] = True
        base = float(np.linalg.norm(base_flat[rows]))
        out[name] = d / base if base > 0 else float("nan")
    base_support = float(np.linalg.norm(base_flat[support]))
    out["support"] = float(np.linalg.norm(delta[support])) / base_support if base_support > 0 else float("nan")
    return out
