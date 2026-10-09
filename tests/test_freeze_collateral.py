"""Frozen fine-tunes and the collateral readouts of an edit."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from goalmisgen.edits.collateral import decisions_away_from, navigation, relative_change, single_objective
from goalmisgen.edits.freeze import freeze, frozen_leaves, frozen_modules
from goalmisgen.envs.level import Level, Objective


def drc_like_params():
    return {
        "params": {
            "network_params": {
                "conv_list_0": {"kernel": jnp.ones((3, 3, 4, 2)), "bias": jnp.ones(2)},
                "conv_list_1": {"kernel": jnp.ones((3, 3, 2, 2))},
                "cell_list_2": {"hh": {"kernel": jnp.ones((3, 3, 2, 2))}},
                "dense_list_0": {"kernel": jnp.ones((4, 3))},
            },
            "actor_params": {"Output": {"kernel": jnp.ones((3, 4))}},
        }
    }


def test_frozen_modules_receive_no_update_through_momentum_and_the_rest_do():
    params = drc_like_params()
    tx = freeze(optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(0.1, weight_decay=0.5)), ["input conv", "conv 1"])
    state = tx.init(params)
    grads = jax_tree_ones(params)
    for _ in range(3):
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    net = params["params"]["network_params"]
    np.testing.assert_array_equal(np.asarray(net["conv_list_0"]["kernel"]), 1.0)
    np.testing.assert_array_equal(np.asarray(net["conv_list_0"]["bias"]), 1.0)
    np.testing.assert_array_equal(np.asarray(net["conv_list_1"]["kernel"]), 1.0)
    assert not np.allclose(np.asarray(net["dense_list_0"]["kernel"]), 1.0)
    assert not np.allclose(np.asarray(params["params"]["actor_params"]["Output"]["kernel"]), 1.0)


def jax_tree_ones(tree):
    import jax

    return jax.tree_util.tree_map(jnp.ones_like, tree)


def test_a_module_name_that_matches_nothing_refuses_to_train():
    with pytest.raises(ValueError, match="no parameter"):
        freeze(optax.sgd(0.1), ["input kernel"]).init(drc_like_params())
    assert len(frozen_leaves(drc_like_params(), ["input conv"])) == 2


def test_the_patch_keeps_cleanbas_optimizer_state_readable_and_is_undone():
    import dataclasses

    import cleanba.cleanba_impala as impala

    from goalmisgen.configs.presets import maze_drc33

    args = dataclasses.replace(maze_drc33(min_size=5, max_size=5), learning_rate=0.1, final_learning_rate=0.1)
    params = drc_like_params()
    original = impala.make_optimizer
    with frozen_modules(["input conv"]):
        tx = impala.make_optimizer(args, params, total_updates=10)
        state = tx.init(params)
        updates, state = tx.update(jax_tree_ones(params), state, params)
        assert "learning_rate" in state.hyperparams
    assert impala.make_optimizer is original
    net = updates["params"]["network_params"]
    assert float(jnp.abs(net["conv_list_0"]["kernel"]).sum()) == 0.0
    assert float(jnp.abs(net["conv_list_1"]["kernel"]).sum()) > 0.0


def corridor_level() -> Level:
    # The poorer objective sits in the corridor between the start and the richer one.
    walls = np.ones((3, 7), dtype=bool)
    walls[1, :] = False
    walls[0, 2] = False
    walls[0, 3] = False
    return Level(
        walls=walls,
        agent_start=(1, 0),
        objectives=(Objective(position=(1, 6), value=1.0, feature_id=0), Objective(position=(1, 3), value=0.5, feature_id=1)),
    )


def test_single_objective_levels_drop_the_other_and_recompute_the_distance():
    rich = single_objective([corridor_level()], "rich", size=7)
    poor = single_objective([corridor_level()], "poor", size=7)
    assert rich.distance.tolist() == [6] and rich.feature.tolist() == [0]
    assert poor.distance.tolist() == [3] and poor.feature.tolist() == [1]
    assert rich.static[0, :, :, 2].sum() == 1 and rich.static[0, :, :, 3].sum() == 0
    assert poor.static[0, :, :, 3].sum() == 1 and poor.static[0, :, :, 2].sum() == 0
    assert tuple(rich.start[0]) == (1, 0)

    summary = navigation(rich, reached=np.array([0]), steps=np.array([8]))
    assert summary == {"success": 1.0, "shortest": 0.0, "excess_steps": 2.0}
    assert navigation(rich, reached=np.array([-1]), steps=np.array([120]))["success"] == 0.0


def test_decisions_away_from_the_threshold_score_the_clear_cases_only():
    gap = np.array([-5.0, 3.0, 9.0, 15.0, 1.0])
    rich = np.zeros(5, dtype=int)
    reached = np.array([0, 1, 1, 0, -1])  # right, excluded (near tau), right, wrong, timeout
    out = decisions_away_from(gap, rich, reached, tau=4.0, margin=3.0)
    assert out == {"accuracy": pytest.approx(2 / 3), "n": 3}


def test_relative_change_is_per_module_and_over_the_support():
    base = np.array([3.0, 4.0, 1.0, 0.0])
    delta = np.array([0.3, 0.4, 0.0, 0.0])
    groups = {"a": np.array([0, 1]), "b": np.array([2, 3])}
    out = relative_change(delta, base, groups, ["a", "b"])
    assert out["a"] == pytest.approx(0.1) and out["b"] == 0.0 and out["support"] == pytest.approx(0.1)
