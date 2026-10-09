"""The cheapest-edit search: targets, loss, edit families, and the JAX rollout.

The two tests that would matter most if they broke are the equivalences: the
teacher-forced logits must be the ones the policy produces stepping the route
one observation at a time, and the in-JAX rollout must walk the maze by the
environment's rules. Either failing silently would make every threshold the
search reports a threshold of something else.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from goalmisgen.analysis import closed_loop
from goalmisgen.analysis.input_gain import gain_part, input_kernel
from goalmisgen.analysis.route_targets import BRANCH, FORK, PAD, PREFIX, route_targets, usable
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33
from goalmisgen.edits import bc as bc_edit
from goalmisgen.edits import rl as rl_edit
from goalmisgen.edits.core import (
    DenseEdit,
    GainEdit,
    LowRankEdit,
    SearchConfig,
    choice_curve,
    curve_loss,
    curve_targets,
    fork_margins,
    margin_curve,
    margin_loss,
    margin_offsets,
    route_loss,
    rule_labels,
    search,
    threshold_targets,
    top_k_indices,
    weight_kernels,
)
from goalmisgen.envs.level import Level, Objective
from goalmisgen.envs.solver import MOVES
from goalmisgen.offline.decode import replay
from goalmisgen.offline.model import ModelConfig, RoutePrefixLM

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
SIZE = 7


def plus_level() -> Level:
    """A corridor from the left that forks at the centre: up to the rich goal, right to the poor one."""
    walls = np.ones((5, 5), dtype=bool)
    walls[2, :] = False
    walls[:, 2] = False
    return Level(
        walls=walls,
        agent_start=(2, 0),
        objectives=(Objective((0, 2), 1.0, 0), Objective((2, 4), 0.5, 1)),
    )


def test_targets_mark_the_fork_and_the_moves_toward_each_goal():
    targets = route_targets([plus_level()], size=5)
    assert targets.gap.tolist() == [0.0] and targets.distances.tolist() == [[4, 4]]
    assert targets.rich_above.tolist() == [True]
    assert targets.role[0, 0].tolist() == [PREFIX, PREFIX, FORK, BRANCH]
    assert targets.role[0, 1].tolist() == [PAD, PAD, PAD, BRANCH]
    assert targets.toward_rich[0, 0, 2].tolist() == [True, False, False, False]  # up at the fork
    assert targets.toward_poor[0, 0, 2].tolist() == [False, False, False, True]  # right at the fork
    assert targets.toward_rich[0, 0, 0].tolist() == targets.toward_poor[0, 0, 0].tolist() == [False, False, False, True]
    assert targets.moves[0, 0].tolist() == [RIGHT, RIGHT, UP, UP]
    assert targets.moves[0, 1].tolist() == [RIGHT, RIGHT, RIGHT, RIGHT]


def test_loss_is_small_when_the_policy_follows_the_target_and_the_fork_error_reads_the_split():
    t = route_targets([plus_level()], size=5)
    # Both goals' moves up to and at the fork; past it, only the moves toward that branch's own goal.
    past_fork = (t.role == BRANCH)[..., None]
    branch_goal = np.stack([t.toward_rich[:, 0], t.toward_poor[:, 1]], axis=1)
    toward = np.where(past_fork, branch_goal, t.toward_rich | t.toward_poor).astype(np.float32)
    logits = 10.0 * toward
    loss, metrics = route_loss(jnp.asarray(logits), t.toward_rich, t.toward_poor, t.role, jnp.asarray([0.5]))
    assert float(metrics["fork_error"]) < 1e-3  # up and right tie at the fork, as p = 0.5 asks
    assert float(metrics["nav"]) < 0.01
    greedy_rich = 10.0 * np.where(t.toward_rich, 1.0, 0.0).astype(np.float32)
    _, rich_metrics = route_loss(jnp.asarray(greedy_rich), t.toward_rich, t.toward_poor, t.role, jnp.asarray([0.5]))
    assert abs(float(rich_metrics["fork_error"]) - 0.5) < 1e-3


def test_margin_loss_charges_the_missing_offset_in_steps_and_nothing_for_matching_the_base():
    t = route_targets([plus_level(), plus_level()], size=5)
    rng = np.random.default_rng(0)
    base = jnp.asarray(rng.normal(size=(2, 2, t.steps, 4)).astype(np.float32))
    offset = jnp.asarray([1.0, -0.5])
    _, metrics = margin_loss(base, base, t.toward_rich, t.toward_poor, t.role, offset, slope=0.5)
    assert float(metrics["nav"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["fork_error"]) == pytest.approx(1.5)  # |1| / 0.5 and |-0.5| / 0.5, averaged

    # Move the up-move logit at the fork so the margin lands exactly on base + offset.
    moved = np.asarray(base).copy()
    m0 = np.asarray(fork_margins(base, t.toward_rich, t.toward_poor, t.role))
    for i, o in enumerate([1.0, -0.5]):
        moved[i, 0, 2, UP] = moved[i, 0, 2, RIGHT] + m0[i] + o  # margin = logit(up) - logit(right) at a two-move fork
    _, metrics = margin_loss(jnp.asarray(moved), base, t.toward_rich, t.toward_poor, t.role, offset, slope=0.5)
    assert float(metrics["fork_error"]) < 1e-4


def test_margin_curve_is_monotone_and_translation_moves_the_crossing():
    rng = np.random.default_rng(0)
    gaps = rng.integers(-15, 16, size=5000).astype(float)
    margins = 3 * np.tanh((7 - gaps) / 2) + rng.normal(scale=0.3, size=gaps.size)
    curve = margin_curve(gaps, margins)
    grid = np.arange(-15, 16)
    assert np.all(np.diff(curve(grid)) <= 1e-9)
    offsets = margin_offsets(curve, np.array([9.0, 30.0]), np.array([True, False]), shifts=[2.0], rule="threshold")
    # At gap 9 a threshold moved from 7 to 9 puts the margin at the old crossing value, about zero.
    assert float(curve(9.0) + offsets[0, 0]) == pytest.approx(float(curve(7.0)), abs=1e-6)
    assert abs(float(curve(7.0))) < 0.3
    assert offsets[1, 0] == pytest.approx(0.0, abs=1e-6)  # flat beyond the data


def test_rule_labels_find_the_base_threshold_and_flip_exactly_the_band_the_shift_crosses():
    gaps = np.arange(-10, 11).astype(float).repeat(3)
    margins = 7.5 - gaps  # decides rich below 7.5
    margins[gaps == -5] *= -1  # three noisy levels
    tau_m, y, weight = rule_labels(gaps, margins, np.zeros(len(gaps), bool), shifts=[2.0, -2.0], rule="threshold")
    assert tau_m == 7.5
    assert weight[gaps == -5].sum() == 0 and weight[gaps != -5].min() == 1
    flipped = (y[:, 0] > 0) != (gaps < 7.5)
    assert set(gaps[flipped]) == {8.0, 9.0}
    assert set(gaps[(y[:, 1] > 0) != (gaps < 7.5)]) == {6.0, 7.0}


def test_curve_loss_accepts_any_levels_that_produce_the_rate():
    t = route_targets([plus_level()] * 4, size=5)
    logits = np.zeros((4, 2, t.steps, 4), np.float32)
    # two levels decide rich strongly, two poor: rate 0.5 in the one bin
    logits[:2, 0, 2, UP] = 5.0
    logits[2:, 0, 2, RIGHT] = 5.0
    swapped = logits[::-1].copy()
    for candidate in (logits, swapped):
        _, metrics = curve_loss(jnp.asarray(candidate), jnp.asarray(candidate), t.toward_rich, t.toward_poor, t.role,
                                jnp.zeros(4, jnp.int32), jnp.full(4, 0.5), n_bins=1, temperature=0.1)
        assert float(metrics["fork_error"]) < 1e-3


def test_curve_targets_translate_the_choice_curve():
    gaps = np.arange(-10, 11).astype(float).repeat(10)
    curve = choice_curve(gaps, gaps < 7)
    _, n_bins, rate = curve_targets(curve, np.array([8.0, 8.0]), np.array([True, False]), [2.0], "contingent")
    assert rate[0, 0] == pytest.approx(1.0) and rate[1, 0] == pytest.approx(0.0)
    assert n_bins == 2


def test_threshold_targets_cross_one_half_at_tau():
    p = threshold_targets(jnp.asarray([4.0, 7.0, 10.0]), 7.0, 0.5)
    assert float(p[1]) == pytest.approx(0.5) and float(p[0]) > 0.99 and float(p[2]) < 0.01


# --- the DRC -------------------------------------------------------------------------


def tiny_drc():
    config = MazeConfig(
        max_episode_steps=40,
        num_envs=2,
        min_size=SIZE,
        max_size=SIZE,
        asynchronous=False,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        objective_values=(1.0, 0.5),
        feature_value_correlation=1.0,
    )
    args = maze_drc33(min_size=SIZE, max_size=SIZE)
    net = dataclasses.replace(args.net, n_recurrent=1, repeats_per_step=2)
    policy, _, params = net.init_params(config.make(), jax.random.PRNGKey(0))
    return config, policy, params


def sample_levels(config, count, seed=0):
    sampler = config.sampler()
    rng = np.random.default_rng(seed)
    levels = []
    while len(levels) < count:
        level = sampler.sample(rng)
        if usable(level):
            levels.append(level)
    return levels


@pytest.fixture(scope="module")
def drc():
    config, policy, params = tiny_drc()
    levels = sample_levels(config, 6)
    return config, policy, params, levels, route_targets(levels, size=SIZE)


def test_teacher_forced_logits_equal_stepping_the_route(drc):
    config, policy, params, levels, targets = drc
    flat, unravel = ravel_pytree(params)
    batch = {"static": jnp.asarray(targets.static), "positions": jnp.asarray(targets.positions)}
    logits = np.asarray(rl_edit.make_logits_fn(policy, unravel)(flat, batch))

    encoder = config.encoder()
    get_action = partial(policy.apply, method=policy.get_action)
    for branch in (0, 1):
        carry = policy.apply(params, jax.random.PRNGKey(0), (len(levels), 4, SIZE, SIZE), method=policy.initialize_carry)
        for t in range(targets.steps):
            obs = np.stack([encoder.encode(level, tuple(targets.positions[i, branch, t])) for i, level in enumerate(levels)])
            starts = np.full(len(levels), t == 0)
            carry, _, step_logits, _ = get_action(params, carry, obs.transpose(0, 3, 1, 2), starts, jax.random.PRNGKey(0), temperature=0.0)
            np.testing.assert_allclose(logits[:, branch, t], np.asarray(step_logits), rtol=1e-4, atol=1e-4)


def test_rollout_walks_the_maze_by_the_environments_rules(drc):
    config, policy, params, levels, targets = drc
    flat, unravel = ravel_pytree(params)
    rollout = closed_loop.make_rollout_fn(policy, unravel, step_limit=30)
    reached, steps = rollout(flat, jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0]), jnp.ones(4))

    # Replay the same greedy policy in Python, one level at a time, and score it with the environment's replay.
    encoder = config.encoder()
    get_action = partial(policy.apply, method=policy.get_action)
    for i, level in enumerate(levels):
        carry = policy.apply(params, jax.random.PRNGKey(0), (1, 4, SIZE, SIZE), method=policy.initialize_carry)
        position, actions = level.agent_start, []
        goals = {o.position for o in level.objectives}
        for t in range(30):
            obs = encoder.encode(level, position)[None].transpose(0, 3, 1, 2)
            carry, action, _, _ = get_action(params, carry, obs, np.array([t == 0]), jax.random.PRNGKey(0), temperature=0.0)
            action = int(action[0])
            actions.append(action)
            r, c = position[0] + MOVES[action][0], position[1] + MOVES[action][1]
            if 0 <= r < level.shape[0] and 0 <= c < level.shape[1] and not level.is_wall((r, c)):
                position = (r, c)
            if position in goals:
                break
        info = replay(level, actions, step_penalty=0.05, step_limit=30)
        assert int(steps[i]) == info["episode_steps"]
        assert int(reached[i]) == (info["reached_feature_id"] if info["reached_objective"] else -1)


def test_onpolicy_logits_are_the_teacher_forced_logits_of_the_agents_own_path(drc):
    from goalmisgen.analysis.route_targets import FORK as FORK_ROLE

    config, policy, params, levels, targets = drc
    flat, unravel = ravel_pytree(params)
    batch = {"static": jnp.asarray(targets.static), "fields": jnp.asarray(targets.fields),
             "start": jnp.asarray(targets.positions[:, 0, 0])}
    out = rl_edit.make_onpolicy_fn(policy, unravel, horizon=12)(flat, batch)
    forced = rl_edit.make_logits_fn(policy, unravel)(flat, {"static": batch["static"], "positions": out["positions"][:, None]})
    np.testing.assert_allclose(np.asarray(out["logits"]), np.asarray(forced)[:, 0], rtol=1e-4, atol=1e-4)

    # Positions follow the greedy moves: consecutive cells differ by at most one step.
    steps = np.abs(np.diff(np.asarray(out["positions"]), axis=1)).sum(-1)
    assert steps.max() <= 1
    # Where the agent commits, the move it takes is exclusive to one objective.
    role = np.asarray(out["role"])
    for i in np.flatnonzero(np.asarray(out["committed"])):
        t = int(np.flatnonzero(role[i] == FORK_ROLE)[0])
        action = int(np.argmax(np.asarray(out["logits"])[i, t]))
        assert np.asarray(out["toward_rich"])[i, t, action] or np.asarray(out["toward_poor"])[i, t, action]


def test_the_gain_family_is_the_gain_part_of_the_input_kernel(drc):
    _, _, params, _, _ = drc
    flat = np.asarray(ravel_pytree(params)[0], dtype=np.float64)
    kernel = input_kernel(params)
    family = GainEdit(size=flat.size, planes={2: kernel.plane(2), 3: kernel.plane(3)}, base_flat=flat)
    g = jnp.asarray([0.3, -0.7])
    axis = np.zeros_like(flat)
    axis[kernel.plane(2)] = 0.3 * flat[kernel.plane(2)]
    axis[kernel.plane(3)] = -0.7 * flat[kernel.plane(3)]
    np.testing.assert_allclose(np.asarray(family.delta({"g": g})), gain_part(axis, flat, kernel, (2, 3)), rtol=1e-5, atol=1e-6)


def test_lowrank_delta_lands_in_each_kernel_in_its_own_axis_order():
    params = {"params": {"dense": {"kernel": np.zeros((3, 5), np.float32), "bias": np.zeros(5, np.float32)},
                         "conv": {"kernel": np.zeros((2, 2, 3, 4), np.float32)}}}
    flat, unravel = ravel_pytree(params)
    kernels = weight_kernels(params)
    family = LowRankEdit(size=flat.size, kernels=kernels, rank=1)
    phi = family.init(jax.random.PRNGKey(0))
    phi = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), phi)
    delta = unravel(family.delta(phi))["params"]
    np.testing.assert_allclose(np.asarray(delta["dense"]["kernel"]), np.ones((3, 5)))
    np.testing.assert_allclose(np.asarray(delta["dense"]["bias"]), np.zeros(5))
    np.testing.assert_allclose(np.asarray(delta["conv"]["kernel"]), np.ones((2, 2, 3, 4)))


def test_dense_family_on_a_subset_and_top_k():
    family = DenseEdit(size=6, indices=np.array([1, 4]))
    delta = np.asarray(family.delta({"u": jnp.asarray([2.0, -3.0])}))
    assert delta.tolist() == [0, 2, 0, 0, -3, 0]
    assert top_k_indices(delta, 1).tolist() == [4]
    assert top_k_indices(delta, 1, within=np.array([0, 1, 2])).tolist() == [1]


def test_search_lowers_the_loss_on_a_tiny_drc(drc):
    _, policy, params, _, targets = drc
    flat, unravel = ravel_pytree(params)
    arrays = {
        "static": targets.static, "positions": targets.positions, "toward_rich": targets.toward_rich,
        "toward_poor": targets.toward_poor, "role": targets.role, "gap": targets.gap, "rich_above": targets.rich_above, "margin_offset": np.tile(np.float32([-1.0, 1.0]), (len(targets), 1)), "rule_sign": np.tile(np.float32([-1.0, 1.0]), (len(targets), 1)), "rule_weight": np.ones(len(targets), np.float32), "gap_bin": np.zeros(len(targets), np.int32), "curve_rate": np.tile(np.float32([0.0, 1.0]), (len(targets), 1)),
    }
    family = DenseEdit(size=flat.size)
    result = search(
        family, rl_edit.make_logits_fn(policy, unravel), flat, arrays,
        SearchConfig(shifts=(-2.0, 2.0), tau_base=0.0, scale=1.0, steps=30, batch=6, lr=1e-2, log_every=29, margin_slope=0.3),
        log=lambda _: None,
    )
    assert result.history[-1]["loss"] < result.history[0]["loss"]
    assert result.delta.shape == (flat.size,)


# --- the route model --------------------------------------------------------------------


def test_route_model_logits_follow_each_branch_and_end_in_eos():
    level = plus_level()
    targets = route_targets([level], size=5)
    config = ModelConfig(size=5, d_model=16, n_layers=1, n_heads=2, max_actions=8, n_channels=4)
    model = RoutePrefixLM(config, remat=False)
    arrays = bc_edit.with_eos(targets, config.max_actions)
    assert arrays["role"][0, 0].tolist() == [PREFIX, PREFIX, FORK, BRANCH, BRANCH]
    assert arrays["toward_rich"][0, 0, 4].tolist() == [False, False, False, False, True]
    assert arrays["toward_poor"][0, 1, 4].tolist() == [False, False, False, False, True]

    obs = closed_loop.draw_agent(jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0]))
    params = model.init(jax.random.PRNGKey(0), obs, jnp.zeros((1, 8), jnp.int32))
    flat, unravel = ravel_pytree(params)
    batch = {k: jnp.asarray(v) for k, v in arrays.items()}
    logits = np.asarray(bc_edit.make_logits_fn(model, unravel)(flat, batch))
    for branch in (0, 1):
        direct, _ = model.apply(params, obs, jnp.asarray(arrays["actions"][:, branch]))
        np.testing.assert_allclose(logits[:, branch], np.asarray(direct)[:, :5], rtol=1e-5, atol=1e-5)
