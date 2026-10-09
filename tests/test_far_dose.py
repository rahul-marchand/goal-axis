import numpy as np
import pytest

from goalmisgen.analysis.far_dose import (
    first_departure,
    fit_linear_gain,
    fit_log_law,
    interpolate_x,
    module_matched_direction,
    random_direction,
)


def test_log_law_recovers_its_parameters_and_inverts():
    gains = np.array([1.0, 1.5, 2.0, 3.0, 5.0])
    law = fit_log_law(gains, 6.9 - 6.3 * np.log(gains))
    assert law.a == pytest.approx(6.9) and law.b == pytest.approx(6.3) and law.rms < 1e-9
    assert law.gain(law.tau(2.7)) == pytest.approx(2.7)


def test_log_law_ignores_non_finite_points():
    law = fit_log_law([1.0, 2.0, 4.0, 8.0], [1.0, 0.0, -1.0, np.nan])
    assert law.b == pytest.approx(1 / np.log(2))


def test_linear_gain_through_the_base():
    assert fit_linear_gain([0.15, 0.3, 0.45], [1.25, 1.5, 1.75]) == pytest.approx(5 / 3)


def test_interpolate_x_finds_the_crossing_and_reports_none_outside():
    points = [(0.0, 6.0), (1.0, 2.0), (2.0, -2.0)]
    assert interpolate_x(points, 0.0) == pytest.approx(1.5)
    assert interpolate_x(points, -5.0) is None


def test_random_directions_have_the_asked_norms():
    rng = np.random.default_rng(0)
    assert np.linalg.norm(random_direction(1000, 3.5, rng)) == pytest.approx(3.5)
    reference = rng.standard_normal(10)
    groups = {"a": np.arange(0, 4), "b": np.arange(4, 8)}  # rows 8-9 in no group
    out = module_matched_direction(reference, groups, rng)
    for rows in (np.arange(0, 4), np.arange(4, 8), np.arange(8, 10)):
        assert np.linalg.norm(out[rows]) == pytest.approx(np.linalg.norm(reference[rows]))
    assert abs(out @ reference) < np.linalg.norm(out) * np.linalg.norm(reference)


def test_first_departure_is_the_smallest_excluding_offset():
    offsets = [2.0, 0.5, 1.0]
    observed = [0.0, 4.0, 3.0]
    low = [-0.2, 3.8, 2.8]
    high = [0.2, 4.2, 3.2]
    predicted = [-2.0, 4.1, 3.5]
    assert first_departure(offsets, observed, low, high, predicted, tolerance=0.5) == 2.0
    assert first_departure(offsets, observed, low, high, predicted, tolerance=0.2) == 1.0
    assert first_departure(offsets, observed, low, high, predicted, tolerance=5.0) is None
