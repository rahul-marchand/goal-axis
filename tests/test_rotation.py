"""The rotation edit turns a colour's input slice without changing its length."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from goalmisgen.analysis.input_gain import InputKernel
from goalmisgen.edits.rotation import RotationEdit, angles, length_drift, partner_map


def kernel_and_base(shape=(3, 3, 4, 8), seed=0):
    """An input kernel at the start of a flat parameter vector, plus junk after it."""
    rng = np.random.default_rng(seed)
    kernel = InputKernel(path="params/conv/kernel", start=0, shape=shape)
    size = int(np.prod(shape))
    return kernel, rng.normal(size=size + 37).astype(np.float32)


def family_of(channels=(2, 3), **kwargs):
    kernel, base = kernel_and_base(**kwargs)
    planes = {int(c): kernel.plane(int(c)) for c in channels}
    return RotationEdit(size=len(base), planes=planes, partners=partner_map(channels), base_flat=base), kernel, base


ROUNDING = 1e-6
"""``delta`` renormalises in float32, so an untouched slice lands within an eps of zero, not on it."""


def test_zero_parameters_are_no_edit():
    family, _, _ = family_of()
    assert np.abs(np.asarray(family.delta(family.init(None)))).max() < ROUNDING


@pytest.mark.parametrize("t", [0.3, 1.0, -0.7, 4.0])
def test_length_is_preserved(t):
    family, _, base = family_of()
    delta = np.asarray(family.delta({"t": jnp.array([t, 0.0])}), dtype=np.float64)
    for channel, drift in length_drift(family, {"t": jnp.array([t, 0.0])}).items():
        assert abs(drift) < 1e-5, f"channel {channel} changed length by {drift}"
    # The untouched plane stays put, and the edit itself is orders of magnitude larger.
    assert np.abs(delta[family.planes[3]]).max() < ROUNDING < np.abs(delta[family.planes[2]]).max()


def test_turning_moves_towards_the_partner():
    """A positive ``t`` reduces the angle to the partner slice."""
    family, _, base = family_of()
    partner = base[family.planes[3]]
    before = base[family.planes[2]]
    delta = np.asarray(family.delta({"t": jnp.array([1.5, 0.0])}), dtype=np.float64)
    after = before + delta[family.planes[2]]

    def angle(a, b):
        return np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))

    assert angle(after, partner) < angle(before, partner)


def test_angle_grows_with_t_and_is_reported():
    family, _, _ = family_of()
    small = angles(family, {"t": jnp.array([0.2, 0.0])})[2]
    large = angles(family, {"t": jnp.array([2.0, 0.0])})[2]
    assert 0 < small < large


def test_the_limit_is_the_partner_direction():
    """A huge ``t`` points the slice where its partner points."""
    family, _, base = family_of()
    delta = np.asarray(family.delta({"t": jnp.array([1e4, 0.0])}), dtype=np.float64)
    turned = base[family.planes[2]] + delta[family.planes[2]]
    partner = base[family.planes[3]]
    cosine = turned @ partner / (np.linalg.norm(turned) * np.linalg.norm(partner))
    assert cosine > 0.999
    # ...and with its own original length, not the partner's.
    assert np.linalg.norm(turned) == pytest.approx(np.linalg.norm(base[family.planes[2]]), rel=1e-5)


def test_two_parameters_like_the_gain_family():
    family, _, _ = family_of()
    assert family.n_parameters() == 2


def test_partner_map_rejects_anything_but_two_colours():
    with pytest.raises(ValueError, match="exactly two"):
        partner_map((1, 2, 3))


def test_rescale_refuses():
    family, _, _ = family_of()
    with pytest.raises(NotImplementedError, match="max-norm"):
        family.rescale({"t": jnp.zeros(2)}, 2.0)


def test_penalty_is_the_edit_length():
    family, _, _ = family_of()
    phi = {"t": jnp.array([0.8, -0.3])}
    delta = np.asarray(family.delta(phi), dtype=np.float64)
    assert float(family.penalty(phi)) == pytest.approx(float(delta @ delta), rel=1e-4)


def test_flat_dense_kernel_works_too():
    """ViT ``embed`` and the route model's ``cell_in`` are ``(channels, width)``."""
    family, _, base = family_of(shape=(4, 16))
    delta = np.asarray(family.delta({"t": jnp.array([0.5, 0.5])}), dtype=np.float64)
    assert np.linalg.norm(delta) > 0
    for drift in length_drift(family, {"t": jnp.array([0.5, 0.5])}).values():
        assert abs(drift) < 1e-5
