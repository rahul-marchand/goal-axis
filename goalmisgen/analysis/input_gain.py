"""The value axis's input-layer slice as a gain on each colour plane, and its twin in the observation.

The restricted writes found that most of the DRC's axis that does anything sits
in the input convolution, and that inside it the axis is almost entirely on the
two colour planes: raising a colour's value amplifies that plane's kernel and
attenuates the other's. A gain on a plane's kernel has an exact twin outside the
network. Every architecture in the project reads the observation through a
linear first layer -- the DRC's and the ResNet's input convolution, the ViT's
patch embedding, the route model's ``cell_in`` -- and a bias is shared across
input channels, so

    K[..., c, :] -> (1 + g) K[..., c, :]    is identical to    obs[..., c] -> (1 + g) obs[..., c].

Scaling what comes in on plane ``c`` is scaling what the network sees of colour
``c``. That makes two things measurable without training anything:

* **the brightness curve** -- threshold against each plane's input gain, a
  property of the network with no fitted direction in it at all;
* **the gain part of the axis** -- the axis's input-layer slice, projected on
  plane ``c`` onto the base kernel's own direction there. Its coefficient is the
  brightness that the write amounts to, per unit of value offset, and the
  remainder is whatever the axis does in the input layer that is *not* a gain.

The identity is arithmetic, and ``tests/test_input_gain.py`` checks it on a
real convolution rather than trusting it; the scripts measure it anyway, as a
check that the plumbing is right.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
from jax import tree_util

from goalmisgen.analysis.layers import path_string

INPUT_KERNEL = re.compile(r"(conv_list_0|xXx_Input_xXx|network_params/embed|^params/cell_in)/kernel$")
"""Paths of the first linear layer's kernel, one per architecture.

DRC ``conv_list_0``, ResNet ``xXx_Input_xXx`` (inside ``ScaledInputSpec``), RL
transformer ``embed``, route model ``cell_in``. Convolution kernels are
``(kh, kw, in, out)`` and dense kernels ``(in, out)``, so the input channel is
the second-to-last axis in both.
"""


@dataclasses.dataclass(frozen=True)
class InputKernel:
    """Where the first layer's kernel sits in the flattened parameter vector."""

    path: str
    start: int
    shape: tuple[int, ...]

    @property
    def n_inputs(self) -> int:
        return self.shape[-2]

    def plane(self, channel: int) -> np.ndarray:
        """Flat indices of the kernel weights that read input channel ``channel``."""
        if not 0 <= channel < self.n_inputs:
            raise ValueError(f"channel {channel} is not an input of a kernel shaped {self.shape}")
        index = np.arange(int(np.prod(self.shape))).reshape(self.shape)
        return self.start + index[..., channel, :].ravel()


def input_kernel(params) -> InputKernel:
    """Locate the first linear layer's kernel. Raises unless exactly one matches."""
    found = []
    start = 0
    for path, leaf in tree_util.tree_flatten_with_path(params)[0]:
        name = path_string(path)
        size = int(np.prod(np.shape(leaf)))
        if INPUT_KERNEL.search(name):
            found.append(InputKernel(name, start, tuple(int(n) for n in np.shape(leaf))))
        start += size
    if len(found) != 1:
        raise ValueError(f"expected one input kernel, found {[k.path for k in found]}")
    return found[0]


def gain_coefficients(axis: np.ndarray, base_flat: np.ndarray, kernel: InputKernel, channels) -> dict[int, float]:
    """Per plane, the axis's projection onto the base kernel there, as a gain per unit offset.

    ``beta_c = <axis_c, K_c> / ||K_c||^2``, so ``base + offset * gain part`` equals
    the base with plane ``c`` scaled by ``1 + offset * beta_c``.
    """
    out = {}
    for channel in channels:
        rows = kernel.plane(channel)
        k = np.asarray(base_flat[rows], dtype=np.float64)
        out[int(channel)] = float(np.asarray(axis[rows], dtype=np.float64) @ k / (k @ k))
    return out


def gain_part(axis: np.ndarray, base_flat: np.ndarray, kernel: InputKernel, channels) -> np.ndarray:
    """The full-length vector holding only the gain component of the axis on ``channels``."""
    out = np.zeros(len(axis), dtype=np.float64)
    for channel, beta in gain_coefficients(axis, base_flat, kernel, channels).items():
        rows = kernel.plane(channel)
        out[rows] = beta * np.asarray(base_flat[rows], dtype=np.float64)
    return out


def kernel_cosines(axis: np.ndarray, base_flat: np.ndarray, kernel: InputKernel) -> dict[int, float]:
    """Per input channel, cos(axis slice, base kernel slice) -- how much of the slice is a gain."""
    out = {}
    for channel in range(kernel.n_inputs):
        rows = kernel.plane(channel)
        a = np.asarray(axis[rows], dtype=np.float64)
        k = np.asarray(base_flat[rows], dtype=np.float64)
        denominator = np.linalg.norm(a) * np.linalg.norm(k)
        out[channel] = float(a @ k / denominator) if denominator > 0 else float("nan")
    return out


def scale_planes(observations: np.ndarray, gains: dict[int, float], channel_axis: int) -> np.ndarray:
    """A copy of ``observations`` with channel ``c`` multiplied by ``gains[c]``."""
    out = np.array(observations, dtype=np.float32, copy=True)
    moved = np.moveaxis(out, channel_axis, -1)
    for channel, gain in gains.items():
        moved[..., channel] *= gain
    return out
