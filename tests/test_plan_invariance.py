"""The measurement pieces behind ``scripts/plan_invariance.py``."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from goalmisgen.analysis import plan_invariance as pi
from goalmisgen.analysis.weights import cosine
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.nets.readers import state_reader_for

SIZE = 7


@pytest.fixture(scope="module")
def drc():
    config = MazeConfig(
        max_episode_steps=40,
        num_envs=3,
        min_size=SIZE,
        max_size=SIZE,
        asynchronous=False,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        objective_values=(1.0, 0.5),
        feature_value_correlation=1.0,
    )
    net = dataclasses.replace(maze_drc33(min_size=SIZE, max_size=SIZE).net, n_recurrent=2, repeats_per_step=3)
    envs = config.make()
    policy, _, params = net.init_params(envs, jax.random.PRNGKey(0))
    observations, _ = envs.reset(seed=0)
    return policy, params, np.asarray(observations, dtype=np.float32)


def test_ticks_walk_the_stock_step_one_repeat_at_a_time(drc):
    policy, params, observations = drc
    reader = state_reader_for(policy)
    carry = policy.apply(params, jax.random.PRNGKey(0), observations.shape, method=policy.initialize_carry)
    starts = np.zeros(observations.shape[0], dtype=bool)
    hs, logits = pi.make_tick_runner(policy, 6)(params, observations)
    for step in range(2):
        carry, _, stock_logits, _, state = reader.step(params, carry, observations, starts, jax.random.PRNGKey(0))
        if step == 0:
            first_step = carry
        tick = 3 * (step + 1) - 1
        for layer, h in enumerate(state.features):
            np.testing.assert_allclose(np.asarray(hs[tick, layer]), h, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(logits[tick]), np.asarray(stock_logits), rtol=1e-5, atol=1e-6)
    cells, _ = pi.make_tick_runner(policy, 3, state="c")(params, observations)
    for layer, c in enumerate(reader.state_of_carry(first_step).cell_state):
        np.testing.assert_allclose(np.asarray(cells[2, layer]), c, rtol=1e-5, atol=1e-6)


def test_an_input_gain_on_an_empty_plane_changes_nothing(drc):
    from jax.flatten_util import ravel_pytree

    from goalmisgen.analysis.input_gain import gain_part, input_kernel

    policy, params, observations = drc
    flat, unravel = ravel_pytree(params)
    flat = np.asarray(flat, np.float64)
    kernel = input_kernel(params)
    axis = np.random.default_rng(0).normal(size=flat.size)
    channel = FIRST_FEATURE_CHANNEL + 1
    # Large, because the policy divides its input by 255 and a small gain on a fresh network is lost in float noise.
    written = unravel(np.asarray(flat + 300.0 * gain_part(axis, flat, kernel, (channel,)), np.float32))
    run = pi.make_tick_runner(policy, 3)
    emptied = pi.remove_objectives(observations, [1])
    base, _ = run(params, emptied)
    edited, _ = run(written, emptied)
    full_base, _ = run(params, observations)
    full_edit, _ = run(written, observations)
    emptied_change = np.abs(np.asarray(base) - np.asarray(edited)).max()
    full_change = np.abs(np.asarray(full_base) - np.asarray(full_edit)).max()
    assert full_change > 1e-3
    assert emptied_change < 1e-3 * full_change


def plus_observation():
    """A 5x5 corridor cross: agent in the centre, objective 0 up, objective 1 left, walls on the corners."""
    obs = np.zeros((5, 5, FIRST_FEATURE_CHANNEL + 2), dtype=np.float32)
    obs[:, :, 0] = 1
    obs[2, :, 0] = 0
    obs[:, 2, 0] = 0
    obs[2, 2, 1] = 1
    obs[0, 2, FIRST_FEATURE_CHANNEL] = 1
    obs[2, 0, FIRST_FEATURE_CHANNEL + 1] = 1
    return obs


def test_level_geometry_splits_routes_and_blocks_the_other_objective():
    geo = pi.level_geometry(plus_observation())
    assert geo.masks["agent"][2, 2]
    assert geo.masks["route 0 only"][1, 2] and not geo.masks["route 0 only"][2, 1]
    assert geo.masks["route 1 only"][2, 1]
    assert not geo.masks["shared route"].any()
    assert geo.masks["off route"][3, 2] and geo.masks["off route"][2, 3]
    assert geo.distance[0][2, 2] == 2 and geo.distance[1][2, 2] == 2
    assert np.isnan(geo.distance[0][2, 0])  # objective 1 blocks, and the cell is it
    union = sum(geo.masks[name].astype(int) for name in pi.LOCI)
    assert np.array_equal(union > 0, geo.free) and union.max() == 1


def test_locus_enrichment_finds_a_change_confined_to_one_route():
    obs = plus_observation()
    geo = pi.level_geometry(obs)
    change = np.where(geo.masks["route 0 only"], 1.0, 0.0)[None]
    enriched = pi.locus_enrichment(change, [geo])
    assert enriched["route 0 only"] == pytest.approx(geo.free.sum() / geo.masks["route 0 only"].sum())
    assert enriched["route 1 only"] == 0
    assert pi.map_correlations(change, [geo])["objective 0"] > 0


def test_field_shift_reads_offset_and_stretch():
    rng = np.random.default_rng(0)
    label = rng.uniform(1, 15, size=2000)
    base = label + rng.normal(size=label.size)
    shift = pi.field_shift(base, base + 2.0, label)
    assert shift["offset"] == pytest.approx(2.0 / label.std())
    assert shift["slope"] == pytest.approx(1.0)
    stretched = pi.field_shift(base, 1.5 * base, label)
    assert stretched["slope"] == pytest.approx(1.5)
    assert stretched["by_distance"][-1] > stretched["by_distance"][0]


def test_subspace_share_and_stable_levels_and_random_matching():
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(2, 32))
    inside = rng.normal(size=(50, 2)) @ directions
    assert pi.subspace_share(inside, directions) == pytest.approx(1.0)
    isotropic = rng.normal(size=(20000, 32))
    assert pi.subspace_share(isotropic, directions) == pytest.approx(2 / 32, abs=0.01)

    np.testing.assert_allclose(pi.readout_directions(np.ones(3), np.array([1.0, 2.0, 1e-8])), [1.0, 0.5, 10.0])

    choices = {"a": np.array([0, 1, 1, -1]), "b": np.array([0, 0, 1, -1])}
    assert pi.stable_levels(choices).tolist() == [True, False, True, False]

    axis = rng.normal(size=100)
    groups = {"x": np.arange(10), "y": np.arange(10, 100)}
    random = pi.module_matched_random(axis, groups, seed=3)
    for index in groups.values():
        assert np.linalg.norm(random[index]) == pytest.approx(np.linalg.norm(axis[index]))
    assert abs(cosine(random, axis)) < 0.5


def test_interpolate_gain_matches_in_log_gain():
    gains, taus = [0.5, 1.0, 2.0], [12.0, 8.0, 4.0]
    assert pi.interpolate_gain(gains, taus, 8.0) == pytest.approx(1.0)
    assert pi.interpolate_gain(gains, taus, 6.0) == pytest.approx(np.sqrt(2))
    assert pi.interpolate_gain(gains, taus, 1.0) == pytest.approx(2.0)
