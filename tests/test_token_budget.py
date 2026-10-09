"""The token budget adds up, and reads a bigger part as bigger."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.token_budget import budget, cell_token_parts


def test_shares_sum_to_one():
    rng = np.random.default_rng(0)
    parts = {name: rng.normal(size=(4, 9, 16)) for name in ("a", "b", "c")}
    shares = budget(parts)
    assert sum(part.norm for part in shares.values()) == pytest.approx(1.0)
    assert sum(part.variation for part in shares.values()) == pytest.approx(1.0)


def test_a_dominant_part_owns_the_token():
    rng = np.random.default_rng(1)
    small = rng.normal(size=(4, 9, 16)) * 0.01
    parts = {"big": rng.normal(size=(4, 9, 16)), "small": small}
    shares = budget(parts)
    assert shares["big"].norm > 0.95
    assert shares["big"].after_norm > shares["small"].after_norm


def test_a_constant_part_carries_no_variation():
    rng = np.random.default_rng(2)
    constant = np.broadcast_to(rng.normal(size=(1, 1, 16)), (4, 9, 16))
    parts = {"varying": rng.normal(size=(4, 9, 16)), "constant": constant}
    shares = budget(parts)
    assert shares["constant"].variation == pytest.approx(0.0, abs=1e-12)
    assert shares["varying"].variation == pytest.approx(1.0)


def test_layer_norm_ignores_a_pure_rescale():
    """A part along the token itself is invisible after normalisation."""
    rng = np.random.default_rng(3)
    token = rng.normal(size=(4, 9, 16))
    parts = {"token": token, "stretch": 0.25 * token}
    shares = budget(parts)
    assert shares["stretch"].norm == pytest.approx(0.2)
    # Not exactly zero: LayerNorm's epsilon makes it scale-invariant only to
    # within eps / variance, which is ~2e-7 at unit variance.
    assert shares["stretch"].after_norm == pytest.approx(0.0, abs=1e-5)


def test_parts_must_be_the_same_shape():
    with pytest.raises(ValueError, match="different shapes"):
        budget({"a": np.zeros((2, 3)), "b": np.zeros((2, 4))})


def test_cell_token_parts_rebuild_the_embedding():
    rng = np.random.default_rng(4)
    size, width, channels = 3, 8, 4
    params = {
        "params": {
            "cell_in": {"kernel": rng.normal(size=(channels, width)), "bias": rng.normal(size=(width,))},
            "row_embedding": rng.normal(size=(size, width)),
            "col_embedding": rng.normal(size=(size, width)),
        }
    }
    cells = rng.normal(size=(2, size * size, channels))
    parts = cell_token_parts(params, cells, size=size, colour=(2, 3))

    tree = params["params"]
    rows = np.arange(size * size) // size
    cols = np.arange(size * size) % size
    expected = (
        cells @ tree["cell_in"]["kernel"] + tree["cell_in"]["bias"] + tree["row_embedding"][rows] + tree["col_embedding"][cols]
    )
    assert np.allclose(sum(parts.values()), expected)
    assert np.allclose(parts["colour"], cells[..., 2:] @ tree["cell_in"]["kernel"][2:])
