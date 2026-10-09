"""The separability fits recover what synthetic agents were built to do."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.separability import (
    Decisions,
    banded_crossings,
    bootstrap,
    field_shape,
    fit_binomial_logit,
    fit_model,
    hat_basis,
    tabulate,
)


def distances(rng, rows):
    return rng.integers(1, 22, rows), rng.integers(1, 22, rows)


def agent(rng, d_rich, d_poor, logit):
    took = rng.random(len(d_rich)) < 1 / (1 + np.exp(-logit))
    ones = np.ones(len(d_rich), bool)
    return Decisions(d_rich, d_poor, took, ones)


def test_hat_basis_interpolates_linearly_and_sums_to_one():
    knots = np.array([0.0, 2.0, 4.0])
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.5, 4.0, 9.0])
    basis = hat_basis(x, knots)
    np.testing.assert_allclose(basis.sum(1), 1.0)
    np.testing.assert_allclose(basis @ knots, np.clip(x, 0, 4))


def test_binomial_logit_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = np.stack([np.ones(4000), rng.normal(size=4000)], 1)
    n = np.full(4000, 20.0)
    k = rng.binomial(20, 1 / (1 + np.exp(-(x @ [0.5, -1.2]))))
    w, deviance = fit_binomial_logit(x, n, k)
    np.testing.assert_allclose(w, [0.5, -1.2], atol=0.03)
    assert deviance > 0


def test_additive_shift_is_separable_and_contours_move_rigidly():
    rng = np.random.default_rng(1)
    rows = 60_000
    dr, dp = distances(rng, rows)
    gap = dr - dp
    base = agent(rng, dr, dp, 0.6 * (7 - gap))
    shifted = agent(rng, dr, dp, 0.6 * (4 - gap))  # threshold 7 -> 4, same slope, no distance dependence
    table = tabulate([base, shifted])
    full = fit_model(table, "full")
    near, slope = full.interaction(1)
    assert abs(near) < 0.03 and abs(slope) < 0.03
    contour = full.contour(1, [3, 8, 13])
    np.testing.assert_allclose(contour, 4.0, atol=0.3)
    np.testing.assert_allclose(full.contour(0, [3, 8, 13]), 7.0, atol=0.3)
    separable = fit_model(table, "separable")
    assert separable.deviance - full.deviance < 15  # chi2 with 2 df


def test_gain_on_a_noisy_field_shows_the_distance_interaction():
    rng = np.random.default_rng(2)
    rows = 80_000
    dr, dp = distances(rng, rows)
    field = lambda d: np.exp(-d / 6.0)  # noqa: E731
    base = agent(rng, dr, dp, 12 * (1.0 * field(dr) - 0.5 * field(dp)))
    brighter_poor = agent(rng, dr, dp, 12 * (1.0 * field(dr) - 0.8 * field(dp)))
    table = tabulate([base, brighter_poor])
    full, separable = fit_model(table, "full"), fit_model(table, "separable")
    assert separable.deviance - full.deviance > 50
    # the gain's effect in log-odds shrinks as both objectives get further away
    near, _ = full.interaction(1)
    assert near > 0


def test_field_shape_calls_an_exponential_field_linear_and_a_saturating_one_not():
    rng = np.random.default_rng(3)
    rows = 150_000
    dr, dp = distances(rng, rows)
    exponential = agent(rng, dr, dp, 1.0 + 0.5 * (dp - dr))
    shape = field_shape(tabulate([exponential]))
    assert shape.deviance_linear - shape.deviance_free < 2 * shape.df + 30
    assert shape.max_departure_steps < 1.0
    assert shape.slope == pytest.approx(-0.5, abs=0.03)

    psi = lambda d: -3.0 * np.log1p(d)  # noqa: E731 - steep near, flat far
    saturating = agent(rng, dr, dp, 1.0 + psi(dr) - psi(dp))
    shape = field_shape(tabulate([saturating]))
    assert shape.deviance_linear - shape.deviance_free > 500
    assert shape.max_departure_steps > 2.0


def test_banded_crossings_follow_a_distance_dependent_threshold():
    rng = np.random.default_rng(4)
    rows = 120_000
    dr, dp = distances(rng, rows)
    near = np.minimum(dr, dp)
    dec = agent(rng, dr, dp, 0.8 * (4 + 0.5 * near - (dr - dp)))
    crossings = banded_crossings(tabulate([dec]), 0, [(1, 3), (8, 10)])
    assert crossings[1] - crossings[0] == pytest.approx(3.5, abs=0.8)


def test_bootstrap_resamples_rows_jointly_and_refuses_unpaired_conditions():
    rng = np.random.default_rng(5)
    dr, dp = distances(rng, 500)
    a = agent(rng, dr, dp, 7 - (dr - dp))
    stats = bootstrap([a, a], lambda t: [t.n[t.condition == 0].sum() - t.n[t.condition == 1].sum()], resamples=5)
    np.testing.assert_array_equal(stats, 0)
    short = Decisions(dr[:-1], dp[:-1], a.took_rich[:-1], a.usable[:-1])
    with pytest.raises(ValueError, match="paired"):
        bootstrap([a, short], lambda t: [0.0], resamples=1)


def test_unusable_rows_are_not_counted():
    d = Decisions.from_arrays([3, 4, -1], [5, 2, 6], [True, False, True], [0, 1, 0], [0, 0, 0])
    table = tabulate([d])
    assert table.n.sum() == 1 and table.k.sum() == 1


def test_unpaired_bootstrap_resamples_each_condition_on_its_own():
    rng = np.random.default_rng(6)
    dr, dp = distances(rng, 400)
    a = agent(rng, dr, dp, 7 - (dr - dp))
    b = Decisions(dr[:300], dp[:300], a.took_rich[:300], a.usable[:300])
    stats = bootstrap([a, b], lambda t: [t.n[t.condition == 0].sum(), t.n[t.condition == 1].sum()], resamples=3, paired=False)
    np.testing.assert_array_equal(stats, [[400, 300]] * 3)
    from goalmisgen.analysis.separability import paired

    assert paired([a, a]) and not paired([a, b])
