"""The ore-field level distribution, spelled out once for the scripts that generate and read it.

``scripts/generate_ore_fields.py`` draws a dataset from :func:`ore_field_sampler`
and records its arguments in ``sampler.json``; ``scripts/generate_craftax_demos.py``
rebuilds the same sampler from that file (:func:`load_sampler`) to check the
dataset's fingerprint. Kept out of :mod:`goalmisgen.craftax.levels`, whose own
source is part of that fingerprint. Numpy only.
"""

from __future__ import annotations

import json
from pathlib import Path

from goalmisgen.craftax.levels import OreFieldGenerator
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.values import FixedValues


def ore_field_sampler(size: int, density: float, values: tuple[float, ...]) -> MazeLevelSampler:
    """The one place the ore-field distribution is spelled out; scripts and tests share it."""
    return MazeLevelSampler(
        generator=OreFieldGenerator(density),
        size_range=(size, size),
        n_objectives=len(values),
        values=FixedValues(tuple(values)),
    )


def load_sampler(levels: Path) -> MazeLevelSampler:
    """Rebuild the sampler an ore-field dataset was drawn from, from its sampler.json."""
    spec = json.loads((Path(levels) / "sampler.json").read_text())
    return ore_field_sampler(int(spec["size"]), float(spec["density"]), tuple(float(v) for v in spec["values"]))
