"""Tests for the retargeting cost on mazes and route models, and for the learners and code run on them.

What is load-bearing: a retargeted demonstration must be the solver's expert at
the new values on the same level and colours; the code length must be the
trainer's own cross-entropy, in bits, per route; a fit must never return
something worse on V than the base it started from; block ``k`` must be
encoded by the model fitted on the blocks before it and nothing else; a
restricted learner must move only what it trains, a subspace learner only
along its directions with ``c = 0`` the base to the bit; and a run's arms must
share their levels whatever their goal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import solve
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM, cross_entropy, targets_from_routes
from goalmisgen.offline.retarget import LN2, PILOT_TAUS, ExchangeRates, RouteScorer, retargeted, tau_values, untied
from goalmisgen.offline.train import initial_params
from goalmisgen.retarget import (
    AllWeights,
    EarlyStopping,
    InSubspace,
    Learner,
    OneTensor,
    Optimiser,
    RunSpec,
    Subspace,
    arm_data,
    doubling_blocks,
    example_bits,
    make_fitter,
    measure,
    prequential,
)

TINY = ModelConfig(size=7, n_channels=5, max_actions=16, d_model=32, n_layers=2, n_heads=2)


def learner(rate=3e-5, eval_first=2, eval_ratio=1.5, patience=2, max_steps=6, trains=AllWeights()) -> Learner:
    return Learner(
        optimiser=Optimiser(learning_rate=rate, batch_size=32),
        stopping=EarlyStopping(eval_first=eval_first, eval_ratio=eval_ratio, patience=patience, max_steps=max_steps),
        trains=trains,
    )


QUICK = learner()


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=120, seed=0, block_size=60)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16)


@pytest.fixture(scope="module")
def model_and_params():
    model = RoutePrefixLM(TINY)
    return model, initial_params(model, jax.random.PRNGKey(0))


def test_tau_values_give_the_rate_back():
    richer, poorer = tau_values(6.5)
    assert (richer - poorer) / 0.05 == pytest.approx(6.5)
    assert tau_values(10.0) == pytest.approx((1.0, 0.5))


def test_retargeting_at_the_base_values_reproduces_the_stored_routes(demos):
    same = retargeted(demos, tau_values(10.0))
    np.testing.assert_array_equal(same.actions, demos.actions)
    np.testing.assert_array_equal(same.target, demos.target)


def test_retargeted_routes_are_the_solvers_expert_at_the_new_values(demos):
    shifted = retargeted(demos, tau_values(2.5))
    np.testing.assert_array_equal(shifted.feature_ids, demos.feature_ids)
    for index in range(len(shifted)):
        level = shifted.level(index)
        assert sorted(o.value for o in level.objectives) == pytest.approx(sorted(tau_values(2.5)))
        assert int(shifted.target[index]) == solve(level, 0.05, step_limit=120).optimal_index
    assert (shifted.target != demos.target).any(), "a rate of 2.5 should flip some choices"


def test_retargeting_past_parity_is_refused(demos):
    with pytest.raises(ValueError, match="parity"):
        retargeted(demos, (0.5, 1.0))


def test_untied_drops_exactly_the_levels_at_the_rate(demos):
    kept = untied(demos, [10.0])
    richer = np.argmax(demos.values, axis=1)
    rows = np.arange(len(demos))
    gap = demos.distances[rows, richer] - demos.distances[rows, 1 - richer]
    np.testing.assert_array_equal(kept, np.flatnonzero(gap != 10))
    assert not demos.ambiguous[kept].any()


def test_route_bits_are_the_trainers_cross_entropy_in_bits(demos, model_and_params):
    model, params = model_and_params
    index = np.arange(8)
    observations = jnp.asarray(demos.observations(index))
    actions = jnp.asarray(demos.routes(index))
    lengths = jnp.asarray(demos.lengths[index], dtype=jnp.int32)
    per_route = example_bits(RouteScorer(model), params, demos, index, chunk=3)
    logits, _ = model.apply(params, observations, actions)
    targets = targets_from_routes(actions, lengths, model.config.eos)
    tokens = float((targets >= 0).sum())
    assert per_route.sum() == pytest.approx(float(cross_entropy(logits, targets)) * tokens / LN2, rel=1e-5)


def test_validation_is_log_spaced_and_ends_on_the_last_step():
    stopping = EarlyStopping(eval_first=5, eval_ratio=2.0, max_steps=50)
    assert stopping.eval_steps() == [5, 10, 20, 40, 50]
    assert stopping.candidates == 6


def test_doubling_blocks():
    assert doubling_blocks(1) == [0, 1]
    assert doubling_blocks(8) == [0, 1, 2, 4, 8]
    assert doubling_blocks(10) == [0, 1, 2, 4, 8, 10]


def test_prequential_refuses_blocks_that_do_not_cover_the_stream(demos, model_and_params):
    model, params = model_and_params
    with pytest.raises(ValueError, match="block boundaries"):
        prequential(
            RouteScorer(model),
            params,
            demos,
            np.arange(8),
            np.arange(8, 12),
            np.arange(12, 16),
            QUICK,
            blocks=lambda n: [0, 4],
            log=lambda _: None,
        )


def test_a_fit_never_scores_worse_than_its_starting_point(demos, model_and_params):
    model, params = model_and_params
    scorer = RouteScorer(model)
    valid = np.arange(80, 120)
    start = example_bits(scorer, params, demos, valid).mean()
    fitted = make_fitter(scorer, QUICK)(params, demos, np.arange(40), valid, seed=0)
    assert fitted.valid_bits <= start + 1e-9
    assert fitted.best_step in [0, *QUICK.stopping.eval_steps()]
    empty = make_fitter(scorer, QUICK)(params, demos, np.arange(0), valid, seed=0)
    assert empty.params is params and empty.best_step == 0


def test_the_first_block_is_encoded_by_the_base(demos, model_and_params):
    model, params = model_and_params
    train, valid, test = np.arange(0, 16), np.arange(80, 100), np.arange(100, 120)
    result = prequential(RouteScorer(model), params, demos, train, valid, test, QUICK, log=lambda _: None)
    assert result.bounds == [0, 1, 2, 4, 8, 16]
    assert len(result.example_bits) == 16 and len(result.test_bits) == 6
    first = example_bits(RouteScorer(model), params, demos, train[:1])
    assert result.example_bits[0] == pytest.approx(first[0])
    assert result.total_bits == pytest.approx(result.example_bits.sum())


# ----------------------------------------------------------------------
# The subspace learner
# ----------------------------------------------------------------------


def in_subspace(subspace: Subspace, eval_first: int = 1, patience: int = 3) -> Learner:
    return learner(rate=0.05, eval_first=eval_first, patience=patience, max_steps=8, trains=InSubspace(subspace))


def subspace_for(params, k: int, seed: int = 0) -> Subspace:
    flat, _ = ravel_pytree(params)
    vectors = np.random.default_rng(seed).standard_normal((k, flat.shape[0]))
    return Subspace.spanning(vectors, name=f"random{k}")


def test_spanning_is_orthonormal_and_keeps_the_first_vector(model_and_params):
    _, params = model_and_params
    flat, _ = ravel_pytree(params)
    vectors = np.random.default_rng(1).standard_normal((3, flat.shape[0])) * [[5.0], [0.1], [2.0]]
    subspace = Subspace.spanning(vectors)
    gram = subspace.directions.astype(np.float64) @ subspace.directions.T
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-5)
    first = vectors[0] / np.linalg.norm(vectors[0])
    assert float(subspace.directions[0] @ first) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="dependent"):
        Subspace.spanning(np.stack([vectors[0], 2 * vectors[0]]))
    with pytest.raises(ValueError, match="orthonormal"):
        Subspace(vectors.astype(np.float32))


def test_zero_coefficients_are_the_base_to_the_bit(model_and_params):
    _, params = model_and_params
    subspace = subspace_for(params, 2)
    at_zero = subspace.point(params, np.zeros(2))
    for got, want in zip(jax.tree_util.tree_leaves(at_zero), jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    moved, _ = ravel_pytree(subspace.point(params, [0.5, -1.0]))
    flat, _ = ravel_pytree(params)
    expected = np.asarray(flat) + 0.5 * subspace.directions[0] - 1.0 * subspace.directions[1]
    np.testing.assert_allclose(np.asarray(moved), expected, atol=1e-6)


def test_a_subspace_fit_moves_only_its_coefficients(demos, model_and_params):
    model, params = model_and_params
    subspace = subspace_for(params, 2)
    fit = make_fitter(RouteScorer(model), in_subspace(subspace))
    fitted = fit(params, demos, np.arange(64), np.arange(80, 120), seed=0)
    assert fitted.recorded is not None and fitted.recorded.shape == (2,)
    flat, _ = ravel_pytree(params)
    got, _ = ravel_pytree(fitted.params)
    # Whatever the fit chose, it is theta0 + c @ U for the c it reports, so
    # every weight moved in lockstep with the coefficients and nothing else.
    np.testing.assert_allclose(np.asarray(got) - np.asarray(flat), fitted.recorded @ subspace.directions, atol=1e-6)
    if fitted.best_step == 0:
        np.testing.assert_array_equal(fitted.recorded, 0)
    else:
        assert np.abs(fitted.recorded).max() > 0


def test_a_one_tensor_direction_leaves_every_other_tensor_bitwise_alone(demos, model_and_params):
    model, params = model_and_params
    flat, unravel = ravel_pytree(params)
    marker = jax.tree_util.tree_map(np.zeros_like, params)
    marker["params"]["head"] = jax.tree_util.tree_map(np.ones_like, params["params"]["head"])
    support, _ = ravel_pytree(marker)
    vector = np.random.default_rng(2).standard_normal(flat.shape[0]) * np.asarray(support)
    subspace = Subspace.spanning(vector[None])
    fitted = make_fitter(RouteScorer(model), in_subspace(subspace, patience=8))(
        params, demos, np.arange(64), np.arange(80, 120), seed=0
    )
    assert fitted.best_step > 0, "with this rate the fit should improve on V at least once"
    for name, tensor in params["params"].items():
        same = jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda a, b: bool(np.array_equal(np.asarray(a), np.asarray(b))), tensor, fitted.params["params"][name]
            )
        )
        assert same == (name != "head"), name


def test_a_subspace_fit_never_scores_worse_than_its_start(demos, model_and_params):
    model, params = model_and_params
    scorer = RouteScorer(model)
    valid = np.arange(80, 120)
    start = example_bits(scorer, params, demos, valid).mean()
    fitted = make_fitter(scorer, in_subspace(subspace_for(params, 1)))(params, demos, np.arange(40), valid, seed=0)
    assert fitted.valid_bits <= start + 1e-9


def test_prequential_records_the_coefficients_per_block(demos, model_and_params):
    model, params = model_and_params
    subspace = subspace_for(params, 2)
    train, valid, test = np.arange(0, 8), np.arange(80, 100), np.arange(100, 120)
    scorer = RouteScorer(model)
    result = prequential(scorer, params, demos, train, valid, test, in_subspace(subspace), log=lambda _: None)
    assert result.recorded.shape == (len(result.bounds), 2)
    np.testing.assert_array_equal(result.recorded[0], 0)  # no data: the base, c = 0
    assert "recorded" in result.to_dict()
    assert "recorded" not in prequential(scorer, params, demos, train, valid, test, QUICK, log=lambda _: None).to_dict()


def test_subspace_round_trips_and_refuses_a_mismatched_model(tmp_path, model_and_params):
    _, params = model_and_params
    subspace = subspace_for(params, 2)
    subspace = Subspace(subspace.directions, name="pca2", meta={"sweep": "o1"})
    subspace.save(tmp_path / "s.npz")
    loaded = Subspace.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(loaded.directions, subspace.directions)
    assert loaded.name == "pca2" and loaded.meta == {"sweep": "o1"}
    short = Subspace.spanning(np.ones((1, 10)))
    with pytest.raises(ValueError, match="spans 10"):
        short.point(params, [0.0])


def test_a_one_tensor_learner_moves_that_tensor_alone(demos, model_and_params):
    model, params = model_and_params
    fit = make_fitter(RouteScorer(model), learner(rate=1e-2, eval_first=1, patience=8, max_steps=4, trains=OneTensor("head")))
    fitted = fit(params, demos, np.arange(64), np.arange(80, 120), seed=0)
    assert fitted.best_step > 0
    for name, tensor in params["params"].items():
        moved = jax.tree_util.tree_map(
            lambda a, b: bool(np.any(np.asarray(a) != np.asarray(b))), tensor, fitted.params["params"][name]
        )
        assert any(jax.tree_util.tree_leaves(moved)) == (name == "head"), name
    with pytest.raises(ValueError, match="no parameter"):
        make_fitter(RouteScorer(model), learner(trains=OneTensor("nope")))(
            params, demos, np.arange(4), np.arange(80, 84), seed=0
        )


# ----------------------------------------------------------------------
# The maze goal family, and a run end to end
# ----------------------------------------------------------------------


def test_every_arm_of_a_study_codes_the_same_levels(demos):
    family = ExchangeRates()
    arms = {tau: arm_data(family, demos, tau, PILOT_TAUS, 16, 8, 8) for tau in PILOT_TAUS}
    for arm in arms.values():
        np.testing.assert_array_equal(arm.split.rows, arms[family.null].split.rows)
    assert not arms[family.null].changed.any()
    with pytest.raises(ValueError, match="study's goals"):
        arm_data(family, demos, 3.0, PILOT_TAUS, 16, 8, 8)


def test_a_study_drops_the_ties_of_every_goal_it_holds(demos):
    """An integer goal adds ties, and they leave every arm's pool, the null's included."""
    family = ExchangeRates()
    with_integer = (*PILOT_TAUS, 3.0)
    rows = family.deterministic(demos, with_integer)
    np.testing.assert_array_equal(rows, untied(demos, with_integer))
    null = arm_data(family, demos, family.null, with_integer, 16, 8, 8)
    assert set(null.split.rows) <= set(rows)


def test_the_pool_must_be_demonstrated_at_the_null_goal(demos):
    with pytest.raises(ValueError, match="null goal"):
        ExchangeRates().check_pool(retargeted(demos, tau_values(2.5)))


def test_measure_records_what_its_spec_asked_for(demos, model_and_params, tmp_path):
    from goalmisgen.retarget import Record

    model, params = model_and_params
    spec = RunSpec(
        base="tiny", pool="fixture", goal=2.5, study_goals=PILOT_TAUS, n=8, n_valid=8, n_test=8, ordering=1, learner=QUICK
    )
    record = measure(spec, ExchangeRates(), RouteScorer(model), params, demos, log=lambda _: None)
    assert record.spec == spec.to_dict()
    assert len(record.changed) == len(record.units) == record.result.n == 8
    arm = arm_data(ExchangeRates(), demos, 2.5, PILOT_TAUS, 8, 8, 8, ordering=1)
    np.testing.assert_array_equal(record.units, np.asarray(arm.data.lengths[arm.split.train]) + 1)
    assert record.result.test_example_bits.shape == (len(record.result.bounds), 8)
    np.testing.assert_array_equal(record.test_changed, arm.changed[arm.split.test])
    np.testing.assert_array_equal(record.test_units, np.asarray(arm.data.lengths[arm.split.test]) + 1)
    own = example_bits(RouteScorer(model), params, demos, arm.split.rows[arm.split.test])  # the pool is the null goal
    assert record.own_goal_test_bits == pytest.approx(own.mean())
    record.save(tmp_path / "r.json")
    loaded = Record.load(tmp_path / "r.json")
    np.testing.assert_array_equal(loaded.result.test_example_bits, record.result.test_example_bits)
    assert loaded.own_goal_test_units == record.own_goal_test_units
    np.testing.assert_array_equal(loaded.result.example_bits, record.result.example_bits)
    assert loaded.spec == record.spec
