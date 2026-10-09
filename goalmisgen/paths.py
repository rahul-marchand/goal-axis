"""Where things live: the data volume's root, and the few places under it every script reaches for.

Scripts used to spell ``/workspace/data/...`` out in their argument defaults
and module constants, so the volume's mount point was decided thirty times
over and moving it meant editing every one. It is decided here instead:
:func:`data_root` reads ``GOALMISGEN_DATA`` and falls back to
``/workspace/data``, where the RunPod network volume is mounted, so nothing
that ran before behaves differently.

*What* a thing under the root is called -- an arm, a checkpoint, a level
dataset's key -- belongs to :mod:`goalmisgen.volume`; this module only says
where the root is and joins names onto it. JAX-free, so launchers and
plotting scripts can use it without loading a network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from goalmisgen.volume import dataset_dirname

DATA_ENV = "GOALMISGEN_DATA"
"""Environment variable naming the data root, for a volume mounted somewhere else."""

DEFAULT_DATA_ROOT = Path("/workspace/data")
"""Where the pods mount the network volume's data, and so where every recorded run lives."""

BASE_VALUES = (1.0, 0.5)
"""What the two objectives pay at every base agent swept so far."""


def data_root() -> Path:
    """The data volume's root: ``$GOALMISGEN_DATA``, or ``/workspace/data``."""
    return Path(os.environ.get(DATA_ENV) or DEFAULT_DATA_ROOT)


def repo_root() -> Path:
    """The checkout this package was imported from (for writing into results/ or figures/)."""
    return Path(__file__).resolve().parent.parent


def runs_dir(root: Path | None = None) -> Path:
    """``runs/``: one directory per RL agent, its arms beneath it."""
    return (root or data_root()) / "runs"


def run_dir(agent: str, root: Path | None = None) -> Path:
    """``runs/<agent>``, e.g. ``runs/novalue11g1.s1234``."""
    return runs_dir(root) / agent


def level_dataset(values: Iterable[float] = BASE_VALUES, n_levels: int = 500_000, root: Path | None = None) -> Path:
    """``levels/values/<values>@<count>``, the shared dataset at those objective values.

    The default, ``1.00-0.50@500k``, is the measurement dataset every value-axis
    readout decodes its held-out split from.
    """
    return (root or data_root()) / "levels" / "values" / dataset_dirname(values, n_levels)


def offline_dir(root: Path | None = None) -> Path:
    """``offline/``: the route model's demonstrations and runs."""
    return (root or data_root()) / "offline"


def offline_runs_dir(root: Path | None = None) -> Path:
    """``offline/runs``: one directory per route model, e.g. ``bcnv11.s1``."""
    return offline_dir(root) / "runs"


def offline_demos_dir(root: Path | None = None) -> Path:
    """``offline/demos``: demonstration sets, one directory per split and correlation."""
    return offline_dir(root) / "demos"


def offline_demos(split: str = "test", rho: int = 100, root: Path | None = None) -> Path:
    """``offline/demos/<split>.rho<rho>``, e.g. the held-out ``test.rho100``."""
    return offline_demos_dir(root) / f"{split}.rho{rho:03d}"


def figures_data_dir(root: Path | None = None) -> Path:
    """``figures-data/``: the volume's large figure inputs (npz decodes, cached axes)."""
    return (root or data_root()) / "figures-data"
