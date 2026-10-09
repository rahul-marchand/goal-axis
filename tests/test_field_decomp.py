"""The erasure decomposition: the identity it rests on, and the two fits.

The decomposition is exact by construction -- ``h(both) - h(neither)`` is the
sum of the two contributions plus the interaction, whatever the network does --
so the test that matters is that the grids come from the policy's own forward
pass and that erasing really removes one objective and nothing else. The fits
are checked against data built to have a known answer.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis.field_decomp import (
    BOTH,
    NEITHER,
    POOR_ONLY,
    RICH_ONLY,
    chebyshev_distance,
    combination_fit,
    decompose,
    erasure_statics,
    field_profile,
    free_cells,
    make_grid_fn,
)
from goalmisgen.analysis.gain_alignment import sites_of
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33
from goalmisgen.edits.targets import route_targets, usable
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL

SIZE = 7


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
    while len(levels) < 4:
        level = sampler.sample(rng)
        if usable(level):
            levels.append(level)
    return policy, params, route_targets(levels, size=SIZE)


def test_erasure_removes_one_objective_and_leaves_the_rest_alone(drc):
    _, _, targets = drc
    statics = erasure_statics(targets)
    colours = slice(FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 2)
    assert statics[BOTH] == pytest.approx(np.asarray(targets.static))
    assert statics[NEITHER][..., colours].sum() == 0.0
    for row in range(len(targets)):
        assert statics[RICH_ONLY][row][..., colours].sum() == pytest.approx(1.0)
        assert statics[POOR_ONLY][row][..., colours].sum() == pytest.approx(1.0)
        rich = FIRST_FEATURE_CHANNEL + int(targets.rich_feature[row])
        assert statics[RICH_ONLY][row][..., rich].sum() == pytest.approx(1.0)
        assert statics[POOR_ONLY][row][..., rich].sum() == 0.0
    # everything that is not a colour plane is untouched in all four variants
    for variant in range(4):
        for channel in range(statics.shape[-1]):
            if channel not in (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1):
                assert statics[variant][..., channel] == pytest.approx(np.asarray(targets.static)[..., channel])


def test_the_decomposition_is_an_identity(drc):
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    sites = sites_of(policy.cfg)
    index = tuple(range(len(sites)))
    grids = make_grid_fn(policy, unravel, n_steps=2, site_index=index)
    out = grids(flat, np.ones(4, dtype=np.float32), erasure_statics(targets), targets.positions[:, 0])
    for site in out:
        a_rich, a_poor, total, inter = decompose(np.asarray(site[0]))
        assert total == pytest.approx(a_rich + a_poor + inter, abs=1e-5)


def test_erasing_both_objectives_leaves_no_objective_signal(drc):
    """A_c is what is there *because* objective c is: with neither drawn, both vanish."""
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    statics = erasure_statics(targets)
    blank = np.stack([statics[NEITHER]] * 4)
    grids = make_grid_fn(policy, unravel, n_steps=1, site_index=(0,))
    out = np.asarray(grids(flat, np.ones(4, dtype=np.float32), blank, targets.positions[:, 0])[0][0])
    a_rich, a_poor, _, inter = decompose(out)
    assert np.abs(a_rich).max() == pytest.approx(0.0, abs=1e-6)
    assert np.abs(a_poor).max() == pytest.approx(0.0, abs=1e-6)
    assert np.abs(inter).max() == pytest.approx(0.0, abs=1e-6)


def test_the_input_conv_only_sees_an_objective_within_its_kernel(drc):
    """The first layer is 3x3, so erasing an objective changes only its grid neighbours.

    Grid neighbours, not maze neighbours: a cell one step away through a wall is
    still inside the kernel. That gap between grid distance and BFS distance is
    what tells a propagated field from convolutional blur, so it is measured
    rather than assumed -- see :func:`chebyshev_distance`.
    """
    policy, params, targets = drc
    flat, unravel = ravel_pytree(params)
    grids = make_grid_fn(policy, unravel, n_steps=1, site_index=(0,))
    out = np.asarray(grids(flat, np.ones(4, dtype=np.float32), erasure_statics(targets), targets.positions[:, 0])[0][0])
    a_rich, _, _, _ = decompose(out)
    changed = np.abs(a_rich).sum(-1) > 1e-6
    grid = chebyshev_distance(targets)[:, 0]
    for row in range(len(targets)):
        assert not changed[row][grid[row] > 1].any()


def test_chebyshev_distance_is_zero_at_the_objective(drc):
    _, _, targets = drc
    grid = chebyshev_distance(targets)
    bfs = np.asarray(targets.fields)
    for row in range(len(targets)):
        for objective in range(2):
            assert grid[row, objective].min() == 0
            assert np.argmin(grid[row, objective]) == np.argmin(bfs[row, objective])


def test_combination_fit_recovers_a_known_rule():
    rng = np.random.default_rng(0)
    near = rng.normal(size=(3, 5, 5, 8))
    far = rng.normal(size=(3, 5, 5, 8))
    mask = np.ones((3, 5, 5), dtype=bool)
    superposed = combination_fit(near, far, near + far, mask)
    assert superposed["alpha"] == pytest.approx(1.0, abs=1e-8)
    assert superposed["gamma"] == pytest.approx(1.0, abs=1e-8)
    winner = combination_fit(near, far, near, mask)
    assert winner["alpha"] == pytest.approx(1.0, abs=1e-8)
    assert winner["gamma"] == pytest.approx(0.0, abs=1e-8)
    assert combination_fit(near, near, near, mask)["identifiable"] is False


def test_field_profile_recovers_an_exponential_decay():
    distance = np.tile(np.arange(11), (2, 11, 1))
    field = np.exp(-0.5 * distance)[..., None] * np.ones(4)
    out = field_profile(field, distance, np.ones_like(distance, dtype=bool), max_distance=10)
    assert out["log_slope"] == pytest.approx(-0.5, abs=1e-6)
    assert out["log_r2"] == pytest.approx(1.0, abs=1e-6)
    assert out["half_distance"] == pytest.approx(2.0)


def test_free_cells_excludes_walls(drc):
    _, _, targets = drc
    mask = free_cells(targets)
    assert mask.shape == (len(targets), SIZE, SIZE)
    for row in range(len(targets)):
        for branch in range(2):
            for step in range(int(targets.lengths[row, branch])):
                cell = targets.positions[row, branch, step]
                assert mask[row, cell[0], cell[1]]
