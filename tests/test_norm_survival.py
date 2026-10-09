"""A LayerNorm keeps the part of a perturbation that turns the token, and no more."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.norm_survival import decompose, layer_norm, relative_change, survival


def tokens(shape=(5, 16), seed=0):
    return np.random.default_rng(seed).normal(size=shape)


def test_shares_sum_to_one():
    x = tokens()
    d = tokens(seed=1)
    parts = decompose(x, d)
    assert parts.ones + parts.magnitude + parts.direction == pytest.approx(1.0)


def test_a_pure_scaling_is_all_magnitude():
    """Scaling a centred token is exactly the direction a LayerNorm discards."""
    x = tokens()
    centred = x - x.mean(axis=-1, keepdims=True)
    parts = decompose(centred, 0.1 * centred)
    assert parts.magnitude == pytest.approx(1.0, abs=1e-9)
    assert parts.direction == pytest.approx(0.0, abs=1e-9)


def test_a_constant_shift_is_all_ones():
    x = tokens()
    parts = decompose(x, np.full_like(x, 0.3))
    assert parts.ones == pytest.approx(1.0, abs=1e-9)


def test_an_orthogonal_turn_is_all_direction():
    x = tokens(shape=(1, 4))
    centred = x - x.mean(axis=-1, keepdims=True)
    # Orthogonal to both the token and the ones vector.
    d = np.array([[1.0, -1.0, 0.0, 0.0]])
    d = d - d.mean()
    d = d - centred * (centred @ d.T) / (centred @ centred.T)
    parts = decompose(x, d)
    assert parts.direction == pytest.approx(1.0, abs=1e-8)


def test_layer_norm_discards_what_decompose_calls_killed():
    """The first-order statement, checked numerically at a small dose."""
    x = tokens(shape=(64, 32), seed=3)
    centred = x - x.mean(axis=-1, keepdims=True)
    for delta, expected in [(1e-4 * centred, 0.0), (np.full_like(x, 1e-4), 0.0)]:
        change = np.linalg.norm(layer_norm(x + delta) - layer_norm(x))
        assert change / np.linalg.norm(layer_norm(x)) == pytest.approx(expected, abs=1e-6)


def test_a_turn_survives_the_norm():
    x = tokens(shape=(64, 32), seed=4)
    rng = np.random.default_rng(5)
    delta = 1e-3 * rng.normal(size=x.shape)
    before = relative_change(x, x + delta)
    after = relative_change(layer_norm(x), layer_norm(x + delta))
    assert after > 0.3 * before  # most of a random perturbation is a turn


def test_layer_norm_matches_flax():
    flax = pytest.importorskip("flax.linen")
    import jax.numpy as jnp

    x = tokens(shape=(3, 8), seed=6)
    module = flax.LayerNorm(epsilon=1e-6)
    variables = module.init(__import__("jax").random.PRNGKey(0), jnp.asarray(x))
    reference = np.asarray(module.apply(variables, jnp.asarray(x)))
    scale = np.asarray(variables["params"]["scale"])
    bias = np.asarray(variables["params"]["bias"])
    assert np.allclose(layer_norm(x, scale, bias), reference, atol=1e-5)


def test_survival_is_one_for_an_identity_stage():
    x = tokens()
    d = 1e-3 * tokens(seed=7)
    assert survival((x, x + d), (x, x + d)) == pytest.approx(1.0)


def test_survival_is_zero_when_the_stage_removes_it():
    x = tokens(shape=(32, 16), seed=8)
    centred = x - x.mean(axis=-1, keepdims=True)
    delta = 1e-4 * centred
    got = survival((x, x + delta), (layer_norm(x), layer_norm(x + delta)))
    assert got == pytest.approx(0.0, abs=1e-3)
