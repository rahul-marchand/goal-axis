"""The feedforward field decomposition's arithmetic, without a checkpoint.

The forward pass needs a trained network, so what is checkable here is the
plumbing around it: that a variant's observation is the maze it claims to be,
that the gain reaches only the plane it names, that "where the distances
disagree" means what it says, and that the linear fit answers the question
``field_profile`` leaves open.
"""

from __future__ import annotations

import numpy as np

from goalmisgen.analysis.feedforward_fields import differing_cells, linear_fit_r2, observations_for
from goalmisgen.envs.observation import AGENT_CHANNEL, FIRST_FEATURE_CHANNEL


def statics(n_levels: int = 3, size: int = 5, channels: int = 4) -> np.ndarray:
    """``(4, N, S, S, C)`` with one objective per colour plane, erased as the variants say."""
    base = np.zeros((n_levels, size, size, channels), dtype=np.float32)
    base[:, 1, 1, FIRST_FEATURE_CHANNEL] = 1.0
    base[:, 3, 3, FIRST_FEATURE_CHANNEL + 1] = 1.0
    out = np.stack([base.copy() for _ in range(4)])
    out[1, :, 3, 3, FIRST_FEATURE_CHANNEL + 1] = 0.0  # rich only
    out[2, :, 1, 1, FIRST_FEATURE_CHANNEL] = 0.0  # poor only
    out[3, :, 1, 1, FIRST_FEATURE_CHANNEL] = 0.0  # neither
    out[3, :, 3, 3, FIRST_FEATURE_CHANNEL + 1] = 0.0
    return out


def test_observations_are_nchw_with_the_agent_drawn():
    grids = statics()
    positions = np.array([[2, 2], [0, 4], [4, 0]], dtype=np.int32)
    observations = observations_for(grids, positions, np.ones(grids.shape[-1], dtype=np.float32))

    assert observations.shape == (4, 3, 4, 5, 5)  # (variants, levels, channels, H, W)
    for variant in range(4):
        for level, (row, column) in enumerate(positions):
            agent = observations[variant, level, AGENT_CHANNEL]
            assert agent[row, column] == 1.0
            assert agent.sum() == 1.0  # exactly one agent, in every variant


def test_the_agent_is_in_the_same_cell_in_every_variant():
    grids = statics()
    positions = np.array([[2, 2], [0, 4], [4, 0]], dtype=np.int32)
    observations = observations_for(grids, positions, np.ones(grids.shape[-1], dtype=np.float32))
    for variant in range(1, 4):
        assert np.array_equal(observations[0, :, AGENT_CHANNEL], observations[variant, :, AGENT_CHANNEL])


def test_gain_scales_only_its_own_plane():
    grids = statics()
    positions = np.zeros((3, 2), dtype=np.int32)
    gain = np.ones(grids.shape[-1], dtype=np.float32)
    gain[FIRST_FEATURE_CHANNEL + 1] = 0.5

    plain = observations_for(grids, positions, np.ones_like(gain))
    scaled = observations_for(grids, positions, gain)

    assert np.allclose(scaled[0, :, FIRST_FEATURE_CHANNEL + 1], 0.5 * plain[0, :, FIRST_FEATURE_CHANNEL + 1])
    for channel in range(grids.shape[-1]):
        if channel != FIRST_FEATURE_CHANNEL + 1:
            assert np.array_equal(scaled[0, :, channel], plain[0, :, channel])


def test_erasure_survives_the_observation_build():
    """A variant that erased an objective must not have it back in the observation."""
    grids = statics()
    positions = np.zeros((3, 2), dtype=np.int32)
    observations = observations_for(grids, positions, np.ones(grids.shape[-1], dtype=np.float32))

    assert observations[1, 0, FIRST_FEATURE_CHANNEL + 1].sum() == 0.0  # rich only: no poor objective
    assert observations[2, 0, FIRST_FEATURE_CHANNEL].sum() == 0.0  # poor only: no rich objective
    assert observations[3, 0, FIRST_FEATURE_CHANNEL].sum() == 0.0
    assert observations[3, 0, FIRST_FEATURE_CHANNEL + 1].sum() == 0.0
    assert observations[0, 0, FIRST_FEATURE_CHANNEL].sum() == 1.0  # both: untouched


def test_differing_cells_keeps_only_real_detours():
    bfs = np.array([[0, 1, 2], [1, 8, 3], [2, 3, 12]], dtype=np.float64)
    grid = np.array([[0, 1, 2], [1, 1, 3], [2, 3, 4]], dtype=np.float64)

    keep = differing_cells(bfs, grid, minimum=3)

    assert keep[1, 1]  # 8 against 1: a wall forced a detour
    assert keep[2, 2]  # 12 against 4
    assert not keep[0, 0]  # the two agree where the path is straight
    assert keep.sum() == 2


def test_linear_fit_r2_is_one_on_a_straight_line_and_low_on_a_curve():
    counts = [100] * 8
    straight = {"means": [10.0 - d for d in range(8)], "counts": counts}
    curved = {"means": [10.0 * 0.5**d for d in range(8)], "counts": counts}

    assert linear_fit_r2(straight) > 0.999
    assert linear_fit_r2(curved) < 0.95


def test_linear_fit_r2_ignores_thin_and_missing_distances():
    """Distances with too few cells, or none, must not enter the fit."""
    means = [10.0, 9.0, 8.0, 7.0, float("nan"), 5.0, 0.0, 1000.0]  # the kept points sit on 10 - d
    counts = [100, 100, 100, 100, 0, 100, 100, 2]  # last bin is too thin to trust
    assert linear_fit_r2({"means": means, "counts": counts}) > 0.99


def test_linear_fit_r2_gives_up_on_too_few_points():
    assert np.isnan(linear_fit_r2({"means": [1.0, 0.5, float("nan")], "counts": [100, 100, 0]}))
