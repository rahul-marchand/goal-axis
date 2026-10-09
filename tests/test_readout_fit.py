"""The fitted readout, checked against data whose answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.readout_fit import (
    agent_vectors,
    auc,
    crossing_in_gap,
    evaluate_readout,
    logistic_direction,
    norm_readout,
    random_directions,
    unit,
)


def test_agent_vectors_picks_each_levels_own_cell():
    field = np.arange(2 * 3 * 3 * 4, dtype=np.float32).reshape(2, 3, 3, 4)
    cells = np.array([[0, 1], [2, 2]])
    got = agent_vectors(field, cells)
    assert np.allclose(got[0], field[0, 0, 1])
    assert np.allclose(got[1], field[1, 2, 2])


def test_auc_is_one_for_a_perfect_separator_and_a_half_for_a_constant():
    label = np.array([0, 0, 1, 1], dtype=bool)
    assert auc(np.array([0.0, 1.0, 2.0, 3.0]), label) == pytest.approx(1.0)
    assert auc(np.array([3.0, 2.0, 1.0, 0.0]), label) == pytest.approx(0.0)
    assert auc(np.zeros(4), label) == pytest.approx(0.5)


def test_logistic_direction_recovers_the_direction_that_generated_the_labels():
    rng = np.random.default_rng(0)
    truth = unit(rng.normal(size=8))
    x = rng.normal(size=(4000, 8))
    y = (x @ truth > 0).astype(float)
    fitted, _ = logistic_direction(x, y, ridge=1e-4)
    assert abs(float(fitted @ truth)) > 0.98


def test_logistic_direction_returns_a_unit_vector():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 5))
    y = (x[:, 0] > 0).astype(float)
    fitted, _ = logistic_direction(x, y)
    assert float(np.linalg.norm(fitted)) == pytest.approx(1.0, abs=1e-9)


def test_crossing_recovers_a_constructed_zero():
    gap = np.linspace(0.0, 20.0, 200)
    # a readout that is zero at gap 7 and falls with the gap
    score = -0.5 * (gap - 7.0)
    got = crossing_in_gap(score, gap)
    assert got["crossing"] == pytest.approx(7.0)
    assert got["slope"] == pytest.approx(-0.5)
    assert got["corr"] == pytest.approx(-1.0)


def test_crossing_is_nan_when_the_readout_says_nothing():
    gap = np.linspace(0.0, 20.0, 50)
    got = crossing_in_gap(np.zeros_like(gap), gap)
    assert np.isnan(got["crossing"])


def test_evaluate_readout_reports_auc_and_crossing_together():
    rng = np.random.default_rng(2)
    gap = rng.uniform(0, 20, size=500)
    score = -(gap - 7.0) + rng.normal(scale=0.5, size=500)
    took_rich = (score > 0).astype(float)
    out = evaluate_readout(score, took_rich, gap)
    assert out["crossing"] == pytest.approx(7.0, abs=0.5)
    assert out["auc"] > 0.99
    assert out["n"] == 500


def test_norm_readout_matches_the_log_length_ratio():
    a_rich = np.array([[3.0, 4.0], [1.0, 0.0]])
    a_poor = np.array([[0.0, 1.0], [0.0, 2.0]])
    got = norm_readout(a_rich, a_poor)
    assert got[0] == pytest.approx(np.log(5.0) - np.log(1.0))
    assert got[1] == pytest.approx(np.log(1.0) - np.log(2.0))


def test_random_directions_are_unit_and_reproducible():
    first = random_directions(16, 32, seed=3)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.allclose(first, random_directions(16, 32, seed=3))


def test_a_direction_fitted_on_shuffled_labels_does_not_predict_the_real_ones():
    rng = np.random.default_rng(4)
    truth = unit(rng.normal(size=10))
    x = rng.normal(size=(1500, 10))
    y = (x @ truth > 0).astype(float)
    shuffled = rng.permutation(y)
    fitted, _ = logistic_direction(x[:1000], shuffled[:1000], ridge=1e-2)
    scored = auc(x[1000:] @ fitted, y[1000:])
    assert 0.35 < scored < 0.65
