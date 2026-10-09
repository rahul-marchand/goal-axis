"""The arm-length decomposition has to be exact, not Pythagorean.

Built from parts whose geometry is known, so every share can be checked against
the construction rather than against another implementation of the same guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis.arm_length import decompose, slope, table_row, write_error
from goalmisgen.analysis.tables import parse_measure_table


def _orthonormal(n_vectors: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, n_vectors)))
    return q.T


def test_orthogonal_parts_give_exact_shares_and_no_cross_term():
    drift, axis, e1, e2 = _orthonormal(4, 200)
    offsets = np.array([0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0])
    # Every arm: 2*drift + offset*5*axis + a residual of length 1, orthogonal to
    # both. Mirrored arms share a residual and successive pairs alternate its
    # sign, so the residuals sum to zero and are uncorrelated with the offsets:
    # least squares then recovers the construction exactly.
    residuals = [e1, e1, -e1, -e1, e2, e2, -e2, -e2, e1]
    diffs = np.stack([2 * drift + o * 5 * axis + r for o, r in zip(offsets, residuals)])
    result = decompose(offsets, diffs, splits=20)
    assert result.drift_norm == pytest.approx(2.0)
    assert result.axis_norm == pytest.approx(5.0)
    for arm in result.arms:
        total = 4 + (arm.offset * 5) ** 2 + 1
        assert arm.norm**2 == pytest.approx(total)
        assert arm.drift == pytest.approx(4 / total)
        assert arm.write == pytest.approx((arm.offset * 5) ** 2 / total)
        assert arm.remainder == pytest.approx(1 / total)
        assert arm.cross == pytest.approx(0.0, abs=1e-9)
    assert result.reliability == pytest.approx(1.0)


def test_non_orthogonal_remainder_is_measured_not_assumed():
    drift, axis = _orthonormal(2, 100)
    offsets = np.array([0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0])
    # A remainder that leans along the drift: Pythagoras would misreport it.
    lean = 0.5 * drift
    signs = [1, 1, -1, -1, 1, 1, -1]
    diffs = np.stack([drift + o * axis + s * lean for s, o in zip(signs, offsets)])
    result = decompose(offsets, diffs, splits=10)
    shares = [a.drift + a.write + a.remainder + a.cross for a in result.arms]
    assert np.allclose(shares, 1.0)
    # The lean survives as a nonzero cross term on the arms whose remainder is
    # not orthogonal to the fitted drift.
    assert any(abs(a.cross) > 1e-3 for a in result.arms)


def test_table_row_reads_only_the_named_dose():
    drift, axis, e1 = _orthonormal(3, 50)
    offsets = np.array([0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
    signs = [1, 1, -1, -1, 1, 1, -1, -1]
    diffs = np.stack([drift + o * axis + s * e1 for s, o in zip(signs, offsets)])
    both = {"o0": decompose(offsets, diffs, splits=10), "o1": decompose(offsets, diffs, splits=10)}
    row = table_row(both, 0.3)
    assert row["arms"] == 4
    total = 1 + 0.09 + 1
    assert row["drift"] == pytest.approx(1 / total)
    assert row["write"] == pytest.approx(0.09 / total)
    with pytest.raises(ValueError):
        table_row(both, 0.35)


def test_refuses_too_few_arms_and_mismatched_shapes():
    with pytest.raises(ValueError):
        decompose(np.array([0.1, -0.1]), np.ones((2, 5)))
    with pytest.raises(ValueError):
        decompose(np.array([0.1, -0.1, 0.2]), np.ones((2, 5)))


SAMPLE = """
arms
  v=0.10  cp_097280  (update 19)  |delta| 2.513
  v=0.20  cp_097280  (update 19)  |delta| 2.466
=== graded and predictive? ===
    value  offset   |delta|  |residual|  held-out R^2  held-out cos  implied offset
     0.10   -0.40     2.513       2.209        -0.137        -0.301           -0.15
     0.90    0.40     2.492       2.226        -0.039         0.354           +0.19
     null    0.00     2.337       2.268             —        -0.025           -0.01
=== what the arms themselves do ===
                                   steps    95% interval    reached
                 base, untouched     6.9  [  6.6,   7.2]     100.0%
               v=0.10 fine-tuned    12.5  [ 12.0,  13.0]     100.0%
               v=0.90 fine-tuned     4.8  [  4.5,   5.1]      99.5%
=== writable? pushing the base weights along the axis ===
           v=0.10 written (grid)    11.9  [ 11.4,  12.5]     100.0%
           v=0.90 written (grid)     4.8  [  4.5,   5.0]     100.0%
         v=1.40 written (unseen)     3.0  [  2.6,   3.2]     100.0%
          random, matched to 0.3     7.0  [  6.6,   7.3]     100.0%
"""

LOO_SAMPLE = """
  v=0.10  cp_399360  (update 78)  |delta| 4.85
=== writable out of sample? each value from an axis that never saw it ===
                                 steps    95% interval    reached
       v=0.10 written, held out    12.0  [ 11.5,  12.4]     100.0%
              v=0.10 fine-tuned    12.7  [ 12.2,  13.1]     100.0%
     value   fine-tuned   written    error
      0.10         12.7      12.0     -0.7
      0.20         10.5      10.7     +0.2
"""


def test_parses_every_section_of_a_014_table():
    table = parse_measure_table(SAMPLE)
    assert table.read_at == 97280
    assert table.base.point == 6.9
    assert table.fine_tuned[0.9].reached == pytest.approx(0.995)
    assert table.written[0.1].low == 11.4
    assert table.unseen[1.4].point == 3.0
    assert table.random[0.3].point == 7.0
    assert table.implied == {-0.4: -0.15, 0.4: 0.19}
    assert slope({v: m.point for v, m in table.fine_tuned.items()}) == pytest.approx((4.8 - 12.5) / 0.8)
    assert write_error(table) == pytest.approx((0.3, -0.3))


def test_parses_a_leave_one_out_table_and_refuses_mixed_budgets():
    table = parse_measure_table(LOO_SAMPLE)
    assert table.read_at == 399360
    assert table.loo == {0.1: (12.7, 12.0), 0.2: (10.5, 10.7)}
    assert 0.1 in table.fine_tuned and not table.written
    with pytest.raises(ValueError):
        parse_measure_table(SAMPLE + "  v=0.30  cp_199680  (update 39)  |delta| 2.4\n")
