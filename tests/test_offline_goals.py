"""Tests for goal vectors in the route model and the goal mixture that trains them.

Load-bearing: a model without goal vectors must be exactly the old model (same
parameter tree, configs written before the field still load); a model with
them must take its goal from the vector alone, defaulting to its base goal; a
mixture's goal id must be the set a demonstration came from; and a learner
restricted to the goal vector must leave every other weight alone.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.goals import GoalMixture, exchange_rate, split_by_goal
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM
from goalmisgen.offline.retarget import RouteScorer, retargeted, tau_values
from goalmisgen.offline.train import initial_params
from goalmisgen.retarget import EarlyStopping, Learner, OneTensor, Optimiser, make_fitter

TINY = ModelConfig(size=7, n_channels=4, max_actions=16, d_model=32, n_layers=2, n_heads=2)
GOALS = dataclasses.replace(TINY, n_goals=3, base_goal=1)


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 7)), n_levels=60, seed=0, block_size=60)
    return DemoSet.generate(dataset, np.arange(len(dataset)), rho=1.0, seed=0, max_actions=16).with_hidden_values()


@pytest.fixture(scope="module")
def mixture(demos) -> GoalMixture:
    return GoalMixture(tuple(retargeted(demos, tau_values(tau)) for tau in (2.5, 10.0, 18.5)))


def batch(demos, n=4):
    index = np.arange(n)
    return jnp.asarray(demos.observations(index)), jnp.asarray(demos.routes(index))


def test_a_model_without_goals_has_no_goal_parameter_and_refuses_ids(demos):
    model = RoutePrefixLM(TINY)
    params = initial_params(model, jax.random.PRNGKey(0))
    assert "goal_embedding" not in params["params"]
    observations, actions = batch(demos)
    with pytest.raises(ValueError, match="no goal vectors"):
        model.apply(params, observations, actions, goals=jnp.zeros(4, dtype=jnp.int32))


def test_configs_written_before_goals_still_load():
    old = {k: v for k, v in TINY.to_dict().items() if k not in ("n_goals", "base_goal")}
    assert ModelConfig.from_dict(old) == TINY


def test_the_goal_vector_is_the_only_difference_between_goals(demos):
    model = RoutePrefixLM(GOALS)
    params = initial_params(model, jax.random.PRNGKey(0))
    assert params["params"]["goal_embedding"].shape == (3, 32)
    observations, actions = batch(demos)
    default, _ = model.apply(params, observations, actions)
    base, _ = model.apply(params, observations, actions, goals=jnp.full(4, 1, dtype=jnp.int32))
    other, _ = model.apply(params, observations, actions, goals=jnp.full(4, 2, dtype=jnp.int32))
    np.testing.assert_array_equal(default, base)
    assert not np.allclose(base, other)
    swapped = jax.tree_util.tree_map(lambda x: x, params)
    table = params["params"]["goal_embedding"]
    swapped["params"]["goal_embedding"] = table.at[1].set(table[2])
    np.testing.assert_allclose(model.apply(swapped, observations, actions)[0], other, rtol=1e-6)


def test_goal_ids_name_the_set_each_demonstration_came_from(mixture, demos):
    n = len(demos)
    assert len(mixture) == 3 * n
    index = np.array([0, n - 1, n, 2 * n + 5, 5])
    np.testing.assert_array_equal(mixture.goal_ids(index), [0, 0, 1, 2, 0])
    assert mixture.taus == pytest.approx((2.5, 10.0, 18.5))
    assert [g["tau"] for g in mixture.meta["goals"]] == pytest.approx([2.5, 10.0, 18.5])


def save_goals(demos, tmp_path, rows_per_goal) -> list:
    paths = []
    for tau, rows in zip((2.5, 10.0, 18.5), rows_per_goal):
        path = tmp_path / f"tau{tau}"
        retargeted(demos.subset(rows), tau_values(tau)).save(path)
        paths.append(path)
    return paths


def test_split_by_goal_partitions_every_row_and_is_fixed_by_its_seed():
    split = split_by_goal(1000, 25, seed=3)
    assert len(split) == 25
    np.testing.assert_array_equal(np.sort(np.concatenate(split)), np.arange(1000))
    assert all(len(rows) > 10 for rows in split)
    assert all(np.array_equal(a, b) for a, b in zip(split, split_by_goal(1000, 25, seed=3)))
    assert not all(np.array_equal(a, b) for a, b in zip(split, split_by_goal(1000, 25, seed=4)))


def test_disjoint_goal_sets_load_only_when_asked_for(demos, tmp_path):
    paths = save_goals(demos, tmp_path, split_by_goal(len(demos), 3, seed=0))
    with pytest.raises(ValueError, match="different levels"):
        GoalMixture.load(paths)
    mixture = GoalMixture.load(paths, levels="disjoint")
    assert mixture.taus == pytest.approx((2.5, 10.0, 18.5))
    assert len(mixture) == len(demos)
    sizes = [len(p) for p in mixture.parts]
    np.testing.assert_array_equal(mixture.goal_ids([0, sizes[0], len(demos) - 1]), [0, 1, 2])


def test_disjoint_goal_sets_refuse_a_shared_level(demos, tmp_path):
    paths = save_goals(demos, tmp_path, [np.arange(0, 20), np.arange(20, 40), np.arange(39, 60)])
    with pytest.raises(ValueError, match="shares 1 levels"):
        GoalMixture.load(paths, levels="disjoint")


def test_same_level_goal_sets_still_load_by_default(demos, tmp_path):
    rows = np.arange(len(demos))
    paths = save_goals(demos, tmp_path, [rows, rows, rows])
    assert GoalMixture.load(paths).taus == pytest.approx((2.5, 10.0, 18.5))
    with pytest.raises(ValueError, match="shares"):
        GoalMixture.load(paths, levels="disjoint")


def test_exchange_rate():
    assert exchange_rate((1.0, 0.5), 0.05) == pytest.approx(10.0)
    assert exchange_rate((0.5, 1.0), 0.05) == pytest.approx(10.0)


def test_a_learner_restricted_to_the_goal_vector_moves_nothing_else(demos):
    model = RoutePrefixLM(GOALS)
    params = initial_params(model, jax.random.PRNGKey(0))
    shifted = retargeted(demos, tau_values(2.5))
    learner = Learner(
        optimiser=Optimiser(learning_rate=1e-2, batch_size=16),
        stopping=EarlyStopping(eval_first=1, eval_ratio=2.0, patience=100, max_steps=4),
        trains=OneTensor("goal_embedding"),
    )
    fitted = make_fitter(RouteScorer(model), learner)(params, shifted, np.arange(30), np.arange(30, 60), seed=0)
    for name, leaf in params["params"].items():
        moved = jax.tree_util.tree_map(lambda a, b: bool(np.any(a != b)), leaf, fitted.params["params"][name])
        assert any(jax.tree_util.tree_leaves(moved)) == (name == "goal_embedding" and fitted.best_step > 0), name
