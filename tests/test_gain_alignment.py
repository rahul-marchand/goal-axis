"""The reassembled DRC step, the gain regression and the patched rollout.

The equivalence test is the one that matters: every number in the gain-alignment
tables is read off :func:`make_step`, so it has to be the policy's own forward
pass, tick for tick and step for step. The rest pin down the two identities the
analysis leans on -- brightness *is* a gain, so it fits with R^2 = 1 at every
site; a zero patch is the base and a full patch at the logits is the write.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis.gain_alignment import (
    add_moments,
    fit_site,
    fraction_carried,
    make_mediation_rollout,
    make_moments_fn,
    make_step,
    sites_of,
)
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33
from goalmisgen.edits import rl as rl_edit
from goalmisgen.edits.targets import PAD, route_targets, usable
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL

SIZE = 7


def tiny_drc():
    config = MazeConfig(
        max_episode_steps=40, num_envs=2, min_size=SIZE, max_size=SIZE, asynchronous=False, value_encoding="none",
        colour_is_the_only_value_cue=True, objective_values=(1.0, 0.5), feature_value_correlation=1.0,
    )
    args = maze_drc33(min_size=SIZE, max_size=SIZE)
    net = dataclasses.replace(args.net, n_recurrent=2, repeats_per_step=2)
    policy, _, params = net.init_params(config.make(), jax.random.PRNGKey(0))
    return config, policy, params


@pytest.fixture(scope="module")
def drc():
    config, policy, params = tiny_drc()
    sampler = config.sampler()
    rng = np.random.default_rng(0)
    levels = []
    while len(levels) < 5:
        level = sampler.sample(rng)
        if usable(level):
            levels.append(level)
    return config, policy, params, route_targets(levels, size=SIZE)


def test_step_reproduces_the_policys_logits_over_several_steps(drc):
    _, policy, params, targets = drc
    step = make_step(policy)
    n = len(targets)
    carry = policy.apply(params, jax.random.PRNGKey(0), (n, 4, SIZE, SIZE), method=policy.initialize_carry)
    ours = carry
    for t in range(4):
        obs = rl_edit.draw_agent(jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, t]))
        starts = jnp.full((n,), t == 0)
        carry, _, logits, _ = policy.apply(
            params, carry, obs.transpose(0, 3, 1, 2), starts, jax.random.PRNGKey(0), temperature=0.0, method=policy.get_action
        )
        ours, our_logits, sites = step(params, ours, obs, starts)
        np.testing.assert_allclose(np.asarray(our_logits), np.asarray(logits), rtol=1e-4, atol=1e-5)
        assert len(sites) == len(sites_of(policy.cfg))
        np.testing.assert_allclose(np.asarray(sites[-1]), np.asarray(logits), rtol=1e-4, atol=1e-5)
    for mine, theirs in zip(ours, carry):
        np.testing.assert_allclose(np.asarray(mine.c), np.asarray(theirs.c), rtol=1e-4, atol=1e-5)


def test_patching_a_site_moves_what_is_downstream_of_it_only(drc):
    _, policy, params, targets = drc
    step = make_step(policy)
    n = len(targets)
    carry = policy.apply(params, jax.random.PRNGKey(0), (n, 4, SIZE, SIZE), method=policy.initialize_carry)
    obs = rl_edit.draw_agent(jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0]))
    starts = jnp.ones((n,), bool)
    _, logits, sites = step(params, carry, obs, starts)
    k = 3  # a cell at the first tick
    patches = [(0.0, jnp.zeros_like(s)) for s in sites]
    patches[k] = (1.0, sites[k] + 1)
    _, patched_logits, patched = step(params, carry, obs, starts, patches)
    for i in range(k):
        np.testing.assert_allclose(np.asarray(patched[i]), np.asarray(sites[i]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(patched[k]), np.asarray(sites[k]) + 1, atol=1e-5)
    assert not np.allclose(np.asarray(patched_logits), np.asarray(logits))


def test_brightness_fits_the_gain_reference_exactly_and_a_random_write_does_not(drc):
    _, policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    moments_fn = make_moments_fn(policy, unravel, eps=0.05)
    n = len(targets)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=flat.size).astype(np.float32) * 0.02
    ones = np.ones(4, np.float32)
    plus_bright, minus_bright = ones.copy(), ones.copy()
    plus_bright[FIRST_FEATURE_CHANNEL] = 1.05
    minus_bright[FIRST_FEATURE_CHANNEL] = 0.95
    flats = jnp.stack([flat, flat + noise, flat, flat - noise])
    gains = jnp.asarray(np.stack([plus_bright, ones, minus_bright, ones]))
    positions = targets.positions.reshape(n * 2, targets.steps, 2)
    static = np.repeat(targets.static, 2, axis=0)
    mask = targets.role.reshape(n * 2, targets.steps) != PAD
    acc = add_moments(None, moments_fn(jnp.asarray(flat), flats, gains, jnp.asarray(static), jnp.asarray(positions), jnp.asarray(mask)))
    for site, moments in zip(sites_of(policy.cfg), acc):
        bright = fit_site(moments, 0)
        assert bright.r2_global > 0.999, (site.name, bright)
        # 5% brighter on colour 0 is a coefficient of 0.05 in log-brightness units.
        assert abs(bright.coef_global[0] - 0.05) < 1e-3 and abs(bright.coef_global[1]) < 1e-3, (site.name, bright)
        assert fit_site(moments, 1).r2_per_channel < 0.9, site.name


def test_patched_rollout_is_the_base_unpatched_and_the_write_when_patched_at_the_logits(drc):
    _, policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    flat = jnp.asarray(flat)
    rng = np.random.default_rng(1)
    write = flat + jnp.asarray(rng.normal(size=flat.size).astype(np.float32) * 0.2)
    sites = sites_of(policy.cfg)
    mediate = make_mediation_rollout(policy, unravel, step_limit=30)
    plain = rl_edit.make_rollout_fn(policy, unravel, step_limit=30)
    static, start, ones = jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0]), jnp.ones(4)
    zeros = jnp.zeros(len(sites))
    _, _, probe = make_step(policy)(params, policy.apply(params, jax.random.PRNGKey(0), (1, 4, SIZE, SIZE), method=policy.initialize_carry),
                                    static[:1], jnp.ones((1,), bool))
    coef = [jnp.zeros(s.shape[-1]) for s in probe]

    base_r, base_s = plain(flat, static, start, ones)
    r, s = mediate(flat, flat, write, ones, static, start, zeros, zeros, zeros, coef, coef)
    np.testing.assert_array_equal(np.asarray(r), np.asarray(base_r))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(base_s))

    write_r, write_s = plain(write, static, start, ones)
    at_logits = zeros.at[len(sites) - 1].set(1.0)
    r, s = mediate(flat, flat, write, ones, static, start, at_logits, at_logits, zeros, coef, coef)
    np.testing.assert_array_equal(np.asarray(r), np.asarray(write_r))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(write_s))


def test_fraction_carried_is_antisymmetric_and_nan_without_an_effect():
    assert fraction_carried(2.0, -2.0, 4.0, -4.0) == pytest.approx(0.5)
    assert np.isnan(fraction_carried(1.0, 0.0, 3.0, 3.0))
