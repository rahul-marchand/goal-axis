"""The route model's erasure decomposition: what it removes, and what it fits.

The decomposition itself is an algebraic identity, so the tests that earn their
keep are the ones on the observation surgery -- erasing an objective must remove
that objective and nothing else -- and on the fits, checked against data whose
answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.route_fields import (
    BOTH,
    NEITHER,
    POOR_ONLY,
    RICH_ONLY,
    cell_grids,
    erasure_observations,
    grid_distance_from,
    objective_channels,
    profile,
    scale_objective,
)
from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL, WALL_CHANNEL
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.demos import DemoSet

SIZE = 7


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(SIZE, SIZE)), n_levels=40, seed=0, block_size=20)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16)


def test_erasure_removes_exactly_one_objective(demos: DemoSet) -> None:
    indices = np.arange(8)
    variants = erasure_observations(demos, indices)
    rich, poor = objective_channels(demos, indices)
    rows = np.arange(len(indices))

    both = variants[BOTH]
    assert np.allclose(both, demos.observations(indices)), "the both-objectives variant is the untouched observation"
    for variant, present, absent in ((RICH_ONLY, rich, poor), (POOR_ONLY, poor, rich)):
        kept = variants[variant][rows, present[:, 1], present[:, 2], present[:, 0]]
        gone = variants[variant][rows, absent[:, 1], absent[:, 2], absent[:, 0]]
        assert np.all(kept == 1.0), "the surviving objective is untouched"
        assert np.all(gone == 0.0), "the erased objective's cell is empty in its own plane"
    for who in (rich, poor):
        assert np.all(variants[NEITHER][rows, who[:, 1], who[:, 2], who[:, 0]] == 0.0)


def test_erasure_touches_no_other_cell_and_no_other_plane(demos: DemoSet) -> None:
    indices = np.arange(8)
    variants = erasure_observations(demos, indices)
    rich, poor = objective_channels(demos, indices)
    value_channel = FIRST_FEATURE_CHANNEL + demos.n_features
    difference = np.abs(variants[NEITHER] - variants[BOTH])
    changed = np.argwhere(difference > 0)
    for level, row, column, channel in changed:
        cells = [tuple(rich[level])[1:], tuple(poor[level])[1:]]
        planes = [rich[level][0], poor[level][0]]
        # An objective's own colour plane, or the value channel at its cell:
        # a value-showing observation carries the objective in both places.
        assert (row, column) in cells, "only the objectives' own cells change"
        assert channel in planes or (not demos.hide_values and channel == value_channel)
    assert np.array_equal(variants[NEITHER][..., WALL_CHANNEL], variants[BOTH][..., WALL_CHANNEL])


def test_scale_objective_scales_one_cell(demos: DemoSet) -> None:
    indices = np.arange(6)
    observations = np.asarray(demos.observations(indices))
    _, poor = objective_channels(demos, indices)
    scaled = scale_objective(observations, poor, 0.5)
    rows = np.arange(len(indices))
    assert np.allclose(scaled[rows, poor[:, 1], poor[:, 2], poor[:, 0]], 0.5)
    untouched = scaled.copy()
    untouched[rows, poor[:, 1], poor[:, 2], poor[:, 0]] = observations[rows, poor[:, 1], poor[:, 2], poor[:, 0]]
    assert np.allclose(untouched, observations), "nothing but that one cell moves"


def test_scale_objective_handles_the_variant_stack(demos: DemoSet) -> None:
    indices = np.arange(5)
    variants = erasure_observations(demos, indices)
    _, poor = objective_channels(demos, indices)
    scaled = scale_objective(variants, poor, 2.0)
    rows = np.arange(len(indices))
    assert np.allclose(scaled[BOTH, rows, poor[:, 1], poor[:, 2], poor[:, 0]], 2.0)
    assert np.allclose(scaled[NEITHER, rows, poor[:, 1], poor[:, 2], poor[:, 0]], 0.0), (
        "scaling an absent objective is a no-op, which is what makes it a source change"
    )


def test_cell_grids_unflattens_prefix_tokens_in_row_major_order() -> None:
    depths, batch, width = 2, 3, 4
    cells = SIZE * SIZE
    streams = np.arange(depths * batch * (cells + 1) * width, dtype=np.float32)
    streams = streams.reshape(depths, batch, cells + 1, width)
    grids = cell_grids(streams, SIZE)
    assert grids.shape == (depths, batch, SIZE, SIZE, width)
    assert np.array_equal(grids[1, 2, 3, 4], streams[1, 2, 3 * SIZE + 4])


def test_cell_grids_rejects_a_short_prefix() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        cell_grids(np.zeros((1, 1, 5, 2)), SIZE)


def test_grid_distance_is_chebyshev_from_the_given_cell() -> None:
    positions = np.array([[0, 0], [3, 2]])
    distance = grid_distance_from(positions, SIZE)
    assert distance[0, 0, 0] == 0 and distance[0, 2, 5] == 5
    assert distance[1, 3, 2] == 0 and distance[1, 0, 0] == 3


def test_profile_recovers_a_known_exponential() -> None:
    size, cells = 15, 2000
    distance = np.tile(np.arange(size), (cells // size, 1))
    field = np.exp(-0.4 * distance)[..., None] * np.ones(3)
    mask = np.ones_like(distance, dtype=bool)
    out = profile(field, distance, mask, max_distance=12)
    assert out["log_slope"] == pytest.approx(-0.4, abs=1e-6)
    assert out["log_r2"] == pytest.approx(1.0, abs=1e-6)
    assert out["linear_r2"] < out["log_r2"], "a straight line is the worse description of a decaying exponential"


def test_profile_recovers_a_known_straight_line() -> None:
    size, cells = 15, 2000
    distance = np.tile(np.arange(size), (cells // size, 1))
    field = (10.0 - 0.5 * distance)[..., None] * np.ones(2) / np.sqrt(2)
    mask = np.ones_like(distance, dtype=bool)
    out = profile(field, distance, mask, max_distance=12)
    assert out["linear_slope"] == pytest.approx(-0.5, abs=1e-6)
    assert out["linear_r2"] == pytest.approx(1.0, abs=1e-6)
    assert out["log_r2"] < out["linear_r2"], "and the exponential is the worse description of a straight line"
