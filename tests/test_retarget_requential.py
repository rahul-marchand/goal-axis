"""Tests for requential coding.

Load-bearing: the first call costs nothing (teacher and student both start at
the base); the code is exactly the per-example KL estimates plus the per-call
overhead bound; the KL estimate is ``log Q - log P`` on the teacher's own
draws, so it is positive on average when the two differ; the student is
trained only on what the code paid for (the teacher's draws), and a fixed
teacher is never moved; the result reads back as written and passes through
every downstream reader (records, diagnostics).
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.retarget import PILOT_TAUS, ExchangeRates, RouteScorer
from goalmisgen.offline.train import initial_params
from goalmisgen.retarget import (
    EMA,
    FineTunedTeacher,
    FixedTeacher,
    Learner,
    OneTensor,
    Optimiser,
    Record,
    Requential,
    RequentialCoding,
    RunSpec,
    Split,
    diagnostics,
    example_bits,
    measure,
)
from goalmisgen.retarget.code import result_from_dict
from goalmisgen.retarget.requential import KAPPA, call_overhead, checkpoint_calls, requential

TINY = ModelConfig(size=7, n_channels=5, max_actions=16, d_model=32, n_layers=2, n_heads=2)
STUDENT = Learner(optimiser=Optimiser(learning_rate=3e-3, batch_size=16))


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=120, seed=0, block_size=60)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16)


@pytest.fixture(scope="module")
def model_and_params():
    model = RoutePrefixLM(TINY)
    return model, initial_params(model, jax.random.PRNGKey(0))


SPLIT = Split(rows=np.arange(120), test=np.arange(0, 20), train=np.arange(20, 100), valid=np.arange(100, 120))


def code(demos, model, params, teacher, calls=4, batch=16, seed=0) -> Requential:
    coding = RequentialCoding(teacher=teacher, calls=calls, batch=batch, checkpoints=3)
    return requential(RouteScorer(model), params, demos, SPLIT, STUDENT, coding, seed=seed, log=lambda _: None)


def test_the_overhead_constant_is_the_papers():
    assert KAPPA == pytest.approx(5.20985, abs=1e-5)  # below 5.21, as the paper states
    assert call_overhead(0.0) == pytest.approx(KAPPA)
    assert call_overhead(-3.0) == pytest.approx(KAPPA)  # a negative KL estimate is charged as zero
    assert call_overhead(3.0) == pytest.approx(2 * math.log2(4) + KAPPA)


def test_checkpoints_start_at_zero_and_end_at_the_last_call():
    assert checkpoint_calls(1, 5) == [0, 1]
    points = checkpoint_calls(64, 7)
    assert points[0] == 0 and points[-1] == 64 and points == sorted(set(points))
    with pytest.raises(ValueError):
        checkpoint_calls(0, 3)


def test_the_first_call_is_free_and_the_code_is_kl_plus_overhead(demos, model_and_params):
    model, params = model_and_params
    teacher = FineTunedTeacher(optimiser=Optimiser(learning_rate=1e-2, batch_size=16))
    result = code(demos, model, params, teacher)
    assert result.call_kl[0] == 0.0  # the teacher has not moved yet
    np.testing.assert_allclose(result.example_bits[:16], 0.0)
    assert result.total_bits == pytest.approx(result.example_bits.sum() + sum(call_overhead(k) for k in result.call_kl))
    assert result.bounds == [16 * c for c in checkpoint_calls(4, 3)]
    assert len(result.example_bits) == len(result.inputs) == len(result.units) == 4 * 16
    assert set(result.inputs) <= set(SPLIT.train)
    assert result.test_example_bits.shape == (len(result.bounds), len(SPLIT.test))
    np.testing.assert_array_equal(result.test_example_bits[0], result.teacher_example_curve[0])  # both the base


def test_a_teacher_that_differs_costs_bits(demos, model_and_params):
    """The KL estimate is log Q - log P on Q's draws: positive on average once Q and P differ."""
    model, params = model_and_params
    other = initial_params(RoutePrefixLM(TINY), jax.random.PRNGKey(7))
    result = code(demos, model, params, FixedTeacher(other, "another init"), calls=2)
    assert result.call_kl[0] > 0
    np.testing.assert_allclose(result.teacher_curve, result.teacher_curve[0])  # a fixed teacher never moves


def test_the_student_learns_only_from_what_was_coded(demos, model_and_params):
    """Replaying the coded draws (seeded by call and row) through the same steps rebuilds the student.

    The draws must match exactly; the bits to float tolerance, since a GPU's
    reductions are not bit-reproducible between runs (2e-6 bits seen).
    """
    model, params = model_and_params
    teacher = FixedTeacher(initial_params(RoutePrefixLM(TINY), jax.random.PRNGKey(7)), "another init")
    first = code(demos, model, params, teacher, calls=3)
    again = code(demos, model, params, teacher, calls=3)
    np.testing.assert_array_equal(first.inputs, again.inputs)
    np.testing.assert_array_equal(first.units, again.units)
    np.testing.assert_allclose(first.example_bits, again.example_bits, atol=1e-4)
    np.testing.assert_allclose(first.test_bits, again.test_bits, atol=1e-4)


def test_a_restricted_teacher_moves_only_what_it_trains(demos):
    goals = RoutePrefixLM(ModelConfig(**{**TINY.to_dict(), "n_goals": 3, "base_goal": 1}))
    params = initial_params(goals, jax.random.PRNGKey(0))
    scorer = RouteScorer(goals)
    spec = FineTunedTeacher(optimiser=Optimiser(learning_rate=1e-1, batch_size=16), trains=OneTensor("goal_embedding"))
    run = spec.begin(scorer, params, demos, SPLIT.train, steps_per_call=3, seed=0)
    run.advance()
    moved = run.params()
    for name, leaf in params["params"].items():
        changed = jax.tree_util.tree_map(
            lambda a, b: bool(np.any(np.asarray(a) != np.asarray(b))), leaf, moved["params"][name]
        )
        assert any(jax.tree_util.tree_leaves(changed)) == (name == "goal_embedding"), name


def test_an_averaged_teacher_stays_nearer_the_base_than_the_raw_one(demos, model_and_params):
    """Averaging is a promise about weights, not loss: a noisy raw teacher can travel far and score no better."""
    from jax.flatten_util import ravel_pytree

    model, params = model_and_params
    scorer = RouteScorer(model)
    optimiser = Optimiser(learning_rate=1e-2, batch_size=16)
    raw = FineTunedTeacher(optimiser=optimiser).begin(scorer, params, demos, SPLIT.train, 5, 0)
    averaged = FineTunedTeacher(optimiser=optimiser, ema=EMA(min_steps=5)).begin(scorer, params, demos, SPLIT.train, 5, 0)
    raw.advance()
    averaged.advance()
    base, _ = ravel_pytree(params)
    distance = {
        name: float(np.linalg.norm(ravel_pytree(run.params())[0] - base))
        for name, run in (("raw", raw), ("averaged", averaged))
    }
    assert 0 < distance["averaged"] < distance["raw"]


def test_a_requential_run_records_reads_back_and_lights_its_teacher_checks(demos, model_and_params, tmp_path):
    model, params = model_and_params
    coding = RequentialCoding(
        teacher=FineTunedTeacher(optimiser=Optimiser(learning_rate=1e-2, batch_size=16)), calls=3, batch=16, checkpoints=2
    )
    spec = RunSpec(
        base="tiny",
        pool="fixture",
        goal=2.5,
        study_goals=PILOT_TAUS,
        n=16,
        n_valid=8,
        n_test=8,
        learner=STUDENT,
        coding=coding,
    )
    record = measure(spec, ExchangeRates(), RouteScorer(model), params, demos, log=lambda _: None)
    assert record.spec["coding"]["kind"] == "requential"
    assert len(record.changed) == len(record.units) == record.result.n == 48
    record.save(tmp_path / "r.json")
    loaded = Record.load(tmp_path / "r.json")
    assert isinstance(loaded.result, Requential)
    assert loaded.result.total_bits == pytest.approx(record.result.total_bits)
    assert isinstance(result_from_dict(record.result.to_dict()), Requential)
    findings = {f.check: f.status for f in diagnostics.run_checks(loaded)}
    for check in (
        "teacher reached the finish line",
        "coded model kept up with its teacher",
        "no teacher drift on unchanged levels",
    ):
        assert findings[check] is not diagnostics.Status.SKIP, check
    assert findings["unpaid information small"] is diagnostics.Status.OK


def test_a_d_smaller_than_one_call_is_refused(demos, model_and_params):
    """Draws are keyed by row, so rows repeated within a call would get identical samples, not independent draws."""
    model, params = model_and_params
    with pytest.raises(ValueError, match="distinct rows"):
        code(demos, model, params, FixedTeacher(params, "the base"), calls=1, batch=len(SPLIT.train) + 1)
