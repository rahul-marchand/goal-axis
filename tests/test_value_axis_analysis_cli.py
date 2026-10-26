"""``experiments/014_value_axis_analysis.py``'s command line and its "task says" column.

The column printed ``(1.0 - value) / step_penalty`` whatever the sweep and the
discount, and a sweep other than the legacy ``c`` silently measured its offsets
from 0.5.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from goalmisgen.analysis.behaviour import optimal_threshold

PATH = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "014_value_axis_analysis.py"


@pytest.fixture(scope="module")
def analysis():
    spec = importlib.util.spec_from_file_location("value_axis_analysis", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REQUIRED = ["--base", "cp", "--arms", "arms", "--levels", "levels"]


def test_only_the_legacy_sweeps_default_their_base_value(analysis):
    assert analysis.parse_args([*REQUIRED]).base_value == 0.5  # 'v', the recorded default invocation
    assert analysis.parse_args([*REQUIRED, "--prefix", "c"]).base_value == 1.0
    assert analysis.parse_args([*REQUIRED, "--prefix", "o0", "--base-value", "1.0"]).base_value == 1.0
    with pytest.raises(SystemExit):
        analysis.parse_args([*REQUIRED, "--prefix", "o0"])


def test_the_task_threshold_follows_the_swept_objective_and_the_discount(analysis):
    base = (1.0, 0.5)
    # colour 1 swept, gamma 1: the old formula's one correct case
    assert analysis.task_threshold(base, 1, 0.3, 0.05, 1.0) == pytest.approx((1.0 - 0.3) / 0.05)
    # colour 0 swept: colour 0 at 1.4 against colour 1 at 0.5, not (1.0 - 1.4) / 0.05
    assert analysis.task_threshold(base, 0, 1.4, 0.05, 1.0) == pytest.approx((1.4 - 0.5) / 0.05)
    # discounted
    assert analysis.task_threshold(base, 1, 0.5, 0.05, 0.995) == pytest.approx(float(optimal_threshold(1.0, 0.5, 0.05, 0.995)))
    assert analysis.task_threshold(base, 1, 0.5, 0.05, 0.995) < 10.0
