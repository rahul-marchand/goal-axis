"""The matching and storage behind the instrument control."""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.control_writes import linear_pair, load_pairs, match_scale, rotation_pair, save_pairs


def test_linear_pair_is_symmetric_about_the_base():
    base = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.5, 0.0, -1.0])
    plus, minus = linear_pair(base, direction, 2.0)
    assert np.allclose(plus, base + 2.0 * direction)
    assert np.allclose(minus, base - 2.0 * direction)
    assert np.allclose((plus + minus) / 2, base)


def test_match_scale_finds_the_dose_of_a_proportional_knob():
    base = np.zeros(3)
    direction = np.array([1.0, 0.0, 0.0])
    # A knob whose shift is 3 steps per unit of scale: the target of 6 wants scale 2.
    matched = match_scale(
        lambda s: linear_pair(base, direction, s),
        lambda plus, minus: 3.0 * (plus - minus)[0] / 2,
        target=6.0,
        scale0=0.5,
        name="linear",
    )
    assert matched.scale == pytest.approx(2.0, rel=1e-6)
    assert matched.shift == pytest.approx(6.0, rel=1e-6)
    assert matched.error < 1e-6
    assert len(matched.history) == 2


def test_match_scale_keeps_the_closest_round_when_the_knob_is_not_proportional():
    base = np.zeros(1)
    direction = np.array([1.0])

    # Saturating: the second round overshoots in scale but not in shift.
    def shift_fn(plus, minus):
        return float(4.0 * np.tanh((plus - minus)[0] / 2))

    matched = match_scale(lambda s: linear_pair(base, direction, s), shift_fn, target=3.0, scale0=1.0, name="saturating")
    best = min(matched.history, key=lambda row: abs(row[1] - 3.0))
    assert matched.scale == pytest.approx(best[0])
    assert matched.shift == pytest.approx(best[1])


def test_match_scale_flips_a_knob_that_points_the_wrong_way():
    """A saved edit's sign is arbitrary; matching has to be free to negate it."""
    base = np.zeros(3)
    direction = np.array([1.0, 0.0, 0.0])
    matched = match_scale(
        lambda s: linear_pair(base, direction, s),
        lambda plus, minus: 3.0 * (plus - minus)[0] / 2,
        target=-6.0,  # the knob moves the threshold up; the axis moves it down
        scale0=0.5,
        name="wrong-way",
        rounds=3,
    )
    assert matched.scale == pytest.approx(-2.0, rel=1e-6)
    assert matched.shift == pytest.approx(-6.0, rel=1e-6)


def test_match_scale_clips_the_size_of_a_correction_but_not_its_sign():
    base = np.zeros(1)
    direction = np.array([1.0])
    # A knob that barely moves the threshold: the exact factor would be ~1e4.
    matched = match_scale(
        lambda s: linear_pair(base, direction, s),
        lambda plus, minus: -1e-4 * (plus - minus)[0] / 2,
        target=5.0,
        scale0=1.0,
        name="tiny",
        rounds=2,
        clip=(0.5, 10.0),
    )
    scales = [s for s, _ in matched.history]
    assert scales[1] == pytest.approx(-10.0)  # clipped in size, negated in sign


def test_match_scale_stops_when_the_knob_does_not_move_the_threshold():
    matched = match_scale(
        lambda s: (np.zeros(2), np.zeros(2)), lambda plus, minus: 0.0, target=5.0, scale0=1.0, name="dead", rounds=4
    )
    assert len(matched.history) == 1  # no second round to extrapolate from
    assert matched.shift == 0.0
    assert matched.error == 5.0


def test_match_scale_rejects_a_zero_target():
    with pytest.raises(ValueError):
        match_scale(lambda s: (np.zeros(1), np.zeros(1)), lambda p, m: 1.0, target=0.0, scale0=1.0)


class _Rotation:
    """A stand-in with the shape of RotationEdit.delta: non-linear in ``t``."""

    def delta(self, phi):
        t = np.asarray(phi["t"], dtype=np.float64)
        return np.concatenate([np.sin(t), np.cos(t) - 1.0])


def test_rotation_pair_turns_both_ways_rather_than_negating():
    base = np.zeros(4)
    family = _Rotation()
    plus, minus = rotation_pair(family, base, np.array([0.3, -0.2]), 2.0)
    assert np.allclose(plus[:2], np.sin([0.6, -0.4]))
    assert np.allclose(minus[:2], np.sin([-0.6, 0.4]))
    # The even part is what a linear direction would have missed: it does not cancel.
    even = (plus + minus) / 2 - base
    assert np.abs(even[2:]).max() > 1e-3


def test_pairs_round_trip_through_disk(tmp_path):
    base = np.arange(5.0)
    a_plus, a_minus = linear_pair(base, np.ones(5), 0.5)
    matched = [
        type(
            "M",
            (),
            {
                "name": "late",
                "plus": a_plus,
                "minus": a_minus,
                "scale": 0.5,
                "shift": 6.0,
                "target": 6.0,
                "history": [(0.5, 6.0)],
            },
        )()
    ]
    save_pairs(tmp_path / "pairs.npz", matched, meta={"base": "unit-test"})
    loaded = load_pairs(tmp_path / "pairs.npz")
    assert set(loaded) == {"late"}
    assert np.allclose(loaded["late"][0], a_plus)
    assert np.allclose(loaded["late"][1], a_minus)
    sidecar = (tmp_path / "pairs.json").read_text()
    assert "unit-test" in sidecar and "late" in sidecar
