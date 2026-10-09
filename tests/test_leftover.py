"""The erased-objective rollout, its change moments and the objective-signal statistic.

Three identities hold by construction and are worth pinning, because every
number in the leftover analysis is a difference of two of these runs: erasing
nothing makes the object-free part of a change *be* the change (so the
object-dependent patch is a no-op and the object-independent one is the write);
a full patch at the logits is the write; and at the input convolution, which is
linear, the signal of an objective is exactly proportional to the gain on its
plane.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis import closed_loop
from goalmisgen.analysis.gain_alignment import add_moments, sites_of
from goalmisgen.analysis.input_gain import input_kernel
from goalmisgen.analysis.leftover import (
    erase,
    make_change_moments_fn,
    make_leftover_rollout,
    make_signal_fn,
)
from goalmisgen.analysis.route_targets import PAD, route_targets, usable
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL

SIZE = 7
COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


@pytest.fixture(scope="module")
def drc():
    config = MazeConfig(
        max_episode_steps=40, num_envs=2, min_size=SIZE, max_size=SIZE, asynchronous=False, value_encoding="none",
        colour_is_the_only_value_cue=True, objective_values=(1.0, 0.5), feature_value_correlation=1.0,
    )
    args = maze_drc33(min_size=SIZE, max_size=SIZE)
    net = dataclasses.replace(args.net, n_recurrent=2, repeats_per_step=2)
    policy, _, params = net.init_params(config.make(), jax.random.PRNGKey(0))
    sampler = config.sampler()
    rng = np.random.default_rng(0)
    levels = []
    while len(levels) < 5:
        level = sampler.sample(rng)
        if usable(level):
            levels.append(level)
    return policy, params, route_targets(levels, size=SIZE)


def test_erase_zeroes_only_the_named_planes(drc):
    _, _, targets = drc
    out = erase(targets.static, [COLOUR[1]])
    assert out[..., COLOUR[1]].sum() == 0
    assert out[..., COLOUR[0]].sum() == targets.static[..., COLOUR[0]].sum() > 0
    np.testing.assert_array_equal(out[..., 0], targets.static[..., 0])


def test_no_patch_is_the_base_and_a_full_patch_at_the_logits_is_the_write(drc):
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    flat = jnp.asarray(flat)
    rng = np.random.default_rng(1)
    write = flat + jnp.asarray(rng.normal(size=flat.size).astype(np.float32) * 0.2)
    rollout = make_leftover_rollout(policy, unravel, step_limit=30)
    plain = closed_loop.make_rollout_fn(policy, unravel, step_limit=30)
    static = jnp.asarray(targets.static)
    null = jnp.asarray(erase(targets.static, [COLOUR[1]]))
    start = jnp.asarray(targets.positions[:, 0, 0])
    zeros = jnp.zeros(len(sites_of(policy.cfg)))

    base_r, base_s = plain(flat, static, start, jnp.ones(4))
    r, s = rollout(flat, flat, write, static, null, start, zeros, zeros, zeros)
    np.testing.assert_array_equal(np.asarray(r), np.asarray(base_r))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(base_s))

    write_r, write_s = plain(write, static, start, jnp.ones(4))
    at_logits = zeros.at[-1].set(1.0)
    r, s = rollout(flat, flat, write, static, null, start, at_logits, at_logits, zeros)
    np.testing.assert_array_equal(np.asarray(r), np.asarray(write_r))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(write_s))


def test_with_nothing_erased_the_object_dependent_patch_is_the_base_and_the_free_part_is_the_write(drc):
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    flat = jnp.asarray(flat)
    rng = np.random.default_rng(2)
    write = flat + jnp.asarray(rng.normal(size=flat.size).astype(np.float32) * 0.2)
    rollout = make_leftover_rollout(policy, unravel, step_limit=30)
    plain = closed_loop.make_rollout_fn(policy, unravel, step_limit=30)
    static = jnp.asarray(targets.static)
    start = jnp.asarray(targets.positions[:, 0, 0])
    ones = jnp.ones(len(sites_of(policy.cfg)))
    zeros = jnp.zeros_like(ones)

    # static_null = static, so dh_null = dh everywhere.
    base_r, _ = plain(flat, static, start, jnp.ones(4))
    r, _ = rollout(flat, flat, write, static, static, start, ones, ones, -ones)
    np.testing.assert_array_equal(np.asarray(r), np.asarray(base_r))

    at_logits_mix = zeros.at[-1].set(1.0)
    write_r, _ = plain(write, static, start, jnp.ones(4))
    r, _ = rollout(flat, flat, write, static, static, start, at_logits_mix, zeros, at_logits_mix)
    np.testing.assert_array_equal(np.asarray(r), np.asarray(write_r))


def test_change_moments_collapse_when_nothing_is_erased(drc):
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    rng = np.random.default_rng(3)
    write = np.asarray(flat) + rng.normal(size=flat.size).astype(np.float32) * 0.1
    moments_fn = make_change_moments_fn(policy, unravel)
    n = len(targets)
    positions = targets.positions.reshape(n * 2, targets.steps, 2)
    static = np.repeat(targets.static, 2, axis=0)
    mask = targets.role.reshape(n * 2, targets.steps) != PAD
    acc = add_moments(None, moments_fn(jnp.asarray(flat), jnp.asarray(write[None]), jnp.asarray(static), jnp.asarray(static),
                                       jnp.asarray(positions), jnp.asarray(mask)))
    for site, m in zip(sites_of(policy.cfg), acc):
        assert m["ff"][0] > 0, site.name
        np.testing.assert_allclose(m["nn"][0], m["ff"][0], rtol=1e-4)
        np.testing.assert_allclose(m["fn"][0], m["ff"][0], rtol=1e-4)


def test_a_gain_on_colour_1_has_no_change_left_once_colour_1_is_erased(drc):
    """The identity the leftover analysis rests on: a gain is nothing without its object."""
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    flat = np.asarray(flat, dtype=np.float64)
    kernel = input_kernel(params)
    gain_write = flat.copy()
    gain_write[kernel.plane(COLOUR[1])] *= 1.3
    moments_fn = make_change_moments_fn(policy, unravel)
    n = len(targets)
    positions = targets.positions.reshape(n * 2, targets.steps, 2)
    static = np.repeat(targets.static, 2, axis=0)
    mask = targets.role.reshape(n * 2, targets.steps) != PAD
    writes = np.stack([gain_write.astype(np.float32)])
    acc = add_moments(None, moments_fn(jnp.asarray(flat, dtype=jnp.float32), jnp.asarray(writes), jnp.asarray(static),
                                       jnp.asarray(erase(static, [COLOUR[1]])), jnp.asarray(positions), jnp.asarray(mask)))
    for site, m in zip(sites_of(policy.cfg), acc):
        assert m["ff"][0] > 0, site.name
        assert m["nn"][0] / m["ff"][0] < 1e-6, (site.name, m["nn"][0] / m["ff"][0])


def test_objective_signal_is_zero_without_erasure_and_linear_in_the_gain_at_the_input_conv(drc):
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    signals_fn = make_signal_fn(policy, unravel)
    n = len(targets)
    positions = targets.positions.reshape(n * 2, targets.steps, 2)
    static = np.repeat(targets.static, 2, axis=0)
    mask = targets.role.reshape(n * 2, targets.steps) != PAD
    gains = np.ones((3, static.shape[-1]), np.float32)
    gains[1, COLOUR[1]] = 2.0
    gains[2, COLOUR[1]] = 4.0
    no_0 = erase(static, [COLOUR[0]])
    no_1 = erase(static, [COLOUR[1]])

    same = add_moments(None, signals_fn(jnp.asarray(flat), jnp.asarray(gains), jnp.asarray(static), jnp.asarray(static),
                                        jnp.asarray(static), jnp.asarray(positions), jnp.asarray(mask)))
    assert np.allclose(same[0]["s0"], 0) and np.allclose(same[0]["s1"], 0)

    acc = add_moments(None, signals_fn(jnp.asarray(flat), jnp.asarray(gains), jnp.asarray(static), jnp.asarray(no_0),
                                       jnp.asarray(no_1), jnp.asarray(positions), jnp.asarray(mask)))
    # The input convolution is linear, so erasing colour 1 removes exactly g * (its kernel
    # contribution): the signal's norm is proportional to the gain, and colour 0's is untouched.
    s1 = np.sqrt(acc[0]["s1"])
    np.testing.assert_allclose(s1[1] / s1[0], 2.0, rtol=1e-3)
    np.testing.assert_allclose(s1[2] / s1[0], 4.0, rtol=1e-3)
    np.testing.assert_allclose(acc[0]["s0"][1] / acc[0]["s0"][0], 1.0, rtol=1e-3)
