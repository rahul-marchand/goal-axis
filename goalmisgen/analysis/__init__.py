"""Measuring agent behaviour against ground truth, and probing internals.

The re-exports below resolve lazily. ``activations`` reaches cleanba and so
JAX, and an eager import here made every submodule -- ``metrics``'s pure array
statistics, ``behaviour``, ``geometry`` -- pay for that runtime too, including
in spawned workers and plotting scripts that never touch a network. Importing
the package now costs nothing; a name loads its module on first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "Rollout": "activations",
    "collect_rollouts": "activations",
    "stack_layers": "activations",
    "BehaviourSummary": "behaviour",
    "bin_by_margin": "behaviour",
    "collect_episode_outcomes": "behaviour",
    "summarise": "behaviour",
    "DistanceBand": "probes",
    "ProbeResult": "probes",
    "auc_interval": "probes",
    "cell_dataset": "probes",
    "probe": "probes",
    "probe_by_distance": "probes",
}

__all__ = [
    "BehaviourSummary",
    "DistanceBand",
    "ProbeResult",
    "Rollout",
    "auc_interval",
    "bin_by_margin",
    "cell_dataset",
    "collect_episode_outcomes",
    "collect_rollouts",
    "probe",
    "probe_by_distance",
    "stack_layers",
    "summarise",
]

if TYPE_CHECKING:
    from goalmisgen.analysis.activations import Rollout, collect_rollouts, stack_layers
    from goalmisgen.analysis.behaviour import BehaviourSummary, bin_by_margin, collect_episode_outcomes, summarise
    from goalmisgen.analysis.probes import (
        DistanceBand,
        ProbeResult,
        auc_interval,
        cell_dataset,
        probe,
        probe_by_distance,
    )


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
