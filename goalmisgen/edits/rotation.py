"""The direction analogue of :class:`~goalmisgen.edits.core.GainEdit`.

A gain edit scales the input kernel's slice for colour ``c``, which is exactly
scaling that colour's plane in the observation. It changes how *loud* a colour
is and nothing else: the contribution a colour makes to the first layer keeps
its direction and changes its length.

That is the wrong currency for a normalised network. The ViT and the route
model are pre-LayerNorm transformers, so the first thing either does with a
token is divide it by its own standard deviation. A change in length is
discarded there; only a change in direction survives. If value is salience
written in whatever coordinate the architecture preserves, then the edit those
two networks can use is a rotation, not a gain -- and the brightness sweeps say
as much already, since brightening a colour moves the ViT's and the route
model's thresholds the *wrong way* while it moves the DRC's the right way.

The rotation this module writes turns colour ``c``'s slice towards the other
colour's, holding its length fixed:

    K_c -> ||K_c|| * (u_c + t_c u_p) / ||u_c + t_c u_p||,     u = K / ||K||

One parameter per colour plane, the same two numbers a gain edit has, and
``t = 0`` is no edit. Large ``t_c`` makes colour ``c``'s tokens point where
colour ``p``'s point, i.e. the network stops being able to tell them apart --
which on its own drives the threshold to zero (prefer the nearer objective)
rather than past it. Past it is the *swap* regime, ``t_c > t_p > 0``, where
colour ``c``'s direction is closer to colour ``p``'s original heading than
colour ``p``'s own now is; the family reaches it, and the search is free to.

The edit is exactly length-preserving by construction, which is what makes it
the clean complement to the gain: between them they span "same direction, new
length" and "same length, new direction" on the same slice of the same tensor.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.input_gain import InputKernel
from goalmisgen.edits.core import EditFamily


def colour_planes(kernel: InputKernel, channels) -> dict[int, np.ndarray]:
    """``channel -> flat indices`` of the input kernel's slice reading that channel."""
    return {int(c): kernel.plane(int(c)) for c in channels}


def partner_map(channels) -> dict[int, int]:
    """Each colour's partner: with two colours, the other one.

    Raises for any other count, because "the direction of the other colour" is
    the whole content of the edit and there is no sensible default for three.
    """
    channels = [int(c) for c in channels]
    if len(channels) != 2:
        raise ValueError(f"a rotation needs exactly two colour planes, got {channels}")
    return {channels[0]: channels[1], channels[1]: channels[0]}


@dataclasses.dataclass
class RotationEdit(EditFamily):
    """One scalar per colour plane: turn its input slice towards the other colour's.

    ``planes`` maps a channel to the flat indices of its kernel slice and
    ``partners`` to the channel it turns towards; both slices must be the same
    size, which they are when they come from one input kernel.
    """

    size: int
    planes: dict[int, np.ndarray]
    partners: dict[int, int]
    base_flat: np.ndarray
    name: str = "rotation"

    def __post_init__(self) -> None:
        for channel, rows in self.planes.items():
            partner = self.partners[channel]
            if partner not in self.planes:
                raise ValueError(f"channel {channel} turns towards {partner}, which is not one of the planes")
            if len(rows) != len(self.planes[partner]):
                raise ValueError(f"slices for {channel} and {partner} differ in size")

    def _unit(self, channel: int) -> jnp.ndarray:
        k = jnp.asarray(self.base_flat[self.planes[channel]], dtype=jnp.float32)
        return k / jnp.maximum(jnp.linalg.norm(k), 1e-12)

    def init(self, key) -> dict:
        return {"t": jnp.zeros(len(self.planes))}

    def delta(self, phi) -> jnp.ndarray:
        flat = jnp.zeros(self.size)
        for i, channel in enumerate(sorted(self.planes)):
            rows = jnp.asarray(self.planes[channel])
            k = jnp.asarray(self.base_flat[self.planes[channel]], dtype=jnp.float32)
            mixed = self._unit(channel) + phi["t"][i] * self._unit(self.partners[channel])
            turned = jnp.linalg.norm(k) * mixed / jnp.maximum(jnp.linalg.norm(mixed), 1e-12)
            flat = flat.at[rows].set(turned - k)
        return flat

    def penalty(self, phi) -> jnp.ndarray:
        """Charge for the weight change itself, as the gain family does."""
        return jnp.sum(jnp.square(self.delta(phi)))

    def rescale(self, phi, factor):
        """Not available: ``delta`` is not linear in ``t``, so no ``t`` scales it by ``factor``.

        Only ``SearchConfig.max_norm`` calls this. A rotation is already bounded
        -- the longest it can be is the chord across the slice's own sphere --
        so a norm ball on top of it has nothing to do.
        """
        raise NotImplementedError("RotationEdit is not linear in its parameters; run without --max-norm")


def angles(family: RotationEdit, phi) -> dict[int, float]:
    """Per plane, the angle in degrees between the base slice and the turned one."""
    out = {}
    delta = np.asarray(family.delta(phi), dtype=np.float64)
    for channel, rows in family.planes.items():
        k = np.asarray(family.base_flat[rows], dtype=np.float64)
        turned = k + delta[rows]
        denominator = np.linalg.norm(k) * np.linalg.norm(turned)
        cosine = float(k @ turned / denominator) if denominator > 0 else float("nan")
        out[int(channel)] = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return out


def length_drift(family: RotationEdit, phi) -> dict[int, float]:
    """Per plane, ``||K_c + delta_c|| / ||K_c|| - 1``: zero if the rotation is exact."""
    out = {}
    delta = np.asarray(family.delta(phi), dtype=np.float64)
    for channel, rows in family.planes.items():
        k = np.asarray(family.base_flat[rows], dtype=np.float64)
        norm = np.linalg.norm(k)
        out[int(channel)] = float(np.linalg.norm(k + delta[rows]) / norm - 1.0) if norm > 0 else float("nan")
    return out
