"""The gain part of an input-layer write is a brightness change, exactly.

Checked on real layers rather than asserted from linearity: a convolution named
the way cleanba names the DRC's, and the route model's ``cell_in``. If a future
architecture put a normalisation or a per-channel bias in front of the first
layer, this is the test that would say the identity no longer holds.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import numpy as np
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis.input_gain import (
    gain_coefficients,
    gain_part,
    input_kernel,
    kernel_cosines,
    scale_planes,
)
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM


class TinyConvNet(nn.Module):
    def setup(self):
        self.conv_list = [nn.Conv(6, (3, 3)), nn.Conv(4, (3, 3))]

    def __call__(self, x):
        return self.conv_list[1](nn.relu(self.conv_list[0](x)))


def test_gain_part_of_a_conv_write_equals_scaling_the_input_planes():
    net = TinyConvNet()
    rng = np.random.default_rng(0)
    x = rng.random((3, 5, 5, 4)).astype(np.float32)
    params = net.init(jax.random.PRNGKey(0), x)
    flat, unravel = ravel_pytree(params)
    flat = np.asarray(flat, dtype=np.float64)
    kernel = input_kernel(params)
    assert kernel.shape == (3, 3, 4, 6)

    axis = rng.normal(size=flat.size)
    offset = 0.3
    betas = gain_coefficients(axis, flat, kernel, channels=(2, 3))
    written = net.apply(unravel(np.asarray(flat + offset * gain_part(axis, flat, kernel, (2, 3)), np.float32)), x)
    brightened = net.apply(params, scale_planes(x, {c: 1 + offset * b for c, b in betas.items()}, channel_axis=-1))
    np.testing.assert_allclose(np.asarray(written), np.asarray(brightened), rtol=1e-4, atol=1e-5)


def test_the_same_identity_holds_for_the_route_models_cell_embedding():
    config = ModelConfig(size=5, d_model=16, n_layers=1, n_heads=2, max_actions=6, n_channels=4)
    model = RoutePrefixLM(config, remat=False)
    rng = np.random.default_rng(1)
    obs = rng.random((2, 5, 5, 4)).astype(np.float32)
    actions = np.zeros((2, 6), dtype=np.int32)
    params = model.init(jax.random.PRNGKey(0), obs, actions)
    flat, unravel = ravel_pytree(params)
    flat = np.asarray(flat, dtype=np.float64)
    kernel = input_kernel(params)
    assert kernel.path == "params/cell_in/kernel" and kernel.shape == (4, 16)

    axis = rng.normal(size=flat.size)
    betas = gain_coefficients(axis, flat, kernel, channels=(2,))
    written, _ = model.apply(unravel(np.asarray(flat - 0.5 * gain_part(axis, flat, kernel, (2,)), np.float32)), obs, actions)
    brightened, _ = model.apply(params, scale_planes(obs, {2: 1 - 0.5 * betas[2]}, channel_axis=-1), actions)
    np.testing.assert_allclose(np.asarray(written), np.asarray(brightened), rtol=1e-4, atol=1e-5)


def test_a_write_along_the_kernel_itself_is_all_gain():
    params = TinyConvNet().init(jax.random.PRNGKey(0), np.zeros((1, 5, 5, 4), np.float32))
    flat = np.asarray(ravel_pytree(params)[0], dtype=np.float64)
    kernel = input_kernel(params)
    axis = np.zeros_like(flat)
    axis[kernel.plane(3)] = -2.0 * flat[kernel.plane(3)]
    assert np.isclose(gain_coefficients(axis, flat, kernel, (3,))[3], -2.0)
    assert np.isclose(kernel_cosines(axis, flat, kernel)[3], -1.0)
    np.testing.assert_allclose(gain_part(axis, flat, kernel, (3,)), axis)


def test_scale_planes_touches_only_the_named_channels_and_copies():
    obs = np.ones((2, 4, 3, 3), dtype=np.float32)  # NCHW
    out = scale_planes(obs, {2: 1.5}, channel_axis=1)
    assert np.all(out[:, 2] == 1.5) and np.all(out[:, [0, 1, 3]] == 1.0)
    assert np.all(obs == 1.0)
