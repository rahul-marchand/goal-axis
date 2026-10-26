"""Tests for behavioural measurement.

The autoreset test is the one that matters. A gymnasium vector environment
returns the *next* episode's info at a terminating step, so reading outcomes
from the top level misattributes every episode by one — silently, with
plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.analysis import bin_by_margin, collect_episode_outcomes, summarise
from goalmisgen.analysis.behaviour import (
    indifference_point,
    on_shortest_route,
    optimal_threshold,
    value_distance_decisions,
)
from goalmisgen.configs.env import MazeConfig


def random_policy(rng):
    def policy(observations, starts):
        del starts  # a stateless policy has nothing to reset
        return rng.integers(0, 4, size=len(observations))

    return policy


def test_a_recurrent_policy_is_told_when_a_new_episode_begins():
    """Without this, a stateful agent carries the finished level's plan forward.

    Every episode after the first would then be measured on an agent that
    started with a plan for a different maze — quietly, and only in the runs
    long enough to reuse an environment.
    """
    config = MazeConfig(max_episode_steps=20, num_envs=4, min_size=5, max_size=5, asynchronous=False)
    envs = config.make()

    seen: list[np.ndarray] = []
    rng = np.random.default_rng(0)

    def policy(observations, starts):
        seen.append(np.asarray(starts).copy())
        return rng.integers(0, 4, size=len(observations))

    collect_episode_outcomes(envs, policy, n_episodes=20, seed=0)

    assert seen[0].all(), "the first step after reset begins an episode in every environment"
    assert not all(s.any() for s in seen[1:]), "mid-episode steps must not clear the recurrent state"
    # Collection stops the moment it has enough episodes, so the final batch of
    # terminations is never shown to the policy; everything earlier must be.
    flagged = sum(int(s.sum()) for s in seen[1:])
    assert flagged >= 20 - envs.num_envs, f"only {flagged} episode starts were reported to the policy"


def test_outcomes_come_from_the_finished_episode_not_the_next_one():
    """final_info describes the episode that ended; top-level info does not."""
    config = MazeConfig(max_episode_steps=20, num_envs=8, min_size=5, max_size=5, asynchronous=False)
    envs = config.make()

    outcomes = collect_episode_outcomes(envs, random_policy(np.random.default_rng(0)), n_episodes=30, seed=0)
    assert len(outcomes) == 30
    # Every outcome is a completed episode, so it carries an episode length.
    assert all("episode_steps" in o for o in outcomes)
    assert all(o["episode_steps"] >= 1 for o in outcomes)
    # And the ground truth that came with that same level.
    assert all("optimal_index" in o for o in outcomes)


def test_summary_excludes_ambiguous_levels_from_optimality():
    """Either choice is optimal on a tie, so those episodes would inflate
    accuracy with coin flips."""
    outcomes = [
        {"reached_objective": True, "chose_optimal": True, "is_ambiguous": True, "reached_feature_id": 0},
        {"reached_objective": True, "chose_optimal": False, "is_ambiguous": False, "reached_feature_id": 1},
    ]
    summary = summarise(outcomes)
    assert summary.chose_optimal == 0.0, "the ambiguous episode must not count as a success"
    assert summary.ambiguous == 0.5


def test_summary_reports_proxy_following_separately_from_optimality():
    outcomes = [
        {"reached_objective": True, "chose_optimal": False, "is_ambiguous": False, "reached_feature_id": 0} for _ in range(4)
    ]
    summary = summarise(outcomes)
    assert summary.followed_feature_zero == 1.0
    assert summary.chose_optimal == 0.0


def test_timeouts_count_against_reaching_but_not_against_optimality():
    outcomes = [
        {"reached_objective": False, "episode_steps": 20},
        {"reached_objective": True, "chose_optimal": True, "is_ambiguous": False, "reached_feature_id": 0},
    ]
    summary = summarise(outcomes)
    assert summary.reached_objective == 0.5
    assert summary.chose_optimal == 1.0


def test_summarising_nothing_is_an_error():
    with pytest.raises(ValueError, match="no episodes"):
        summarise([])


def test_every_environment_contributes_the_same_number_of_episodes():
    """Otherwise fast environments dominate and short episodes are oversampled.

    Timeouts run the full step limit, so they are exactly the episodes a
    total-count loop misses - inflating the reach rate and deflating the mean
    length, in the direction that flatters the agent.
    """
    config = MazeConfig(max_episode_steps=20, num_envs=4, min_size=5, max_size=5, asynchronous=False)
    envs = config.make()

    # `starts` is the previous step's done flags, so counting it counts the
    # episodes each environment actually finished while collection ran.
    finished = np.zeros(4, dtype=int)
    rng = np.random.default_rng(0)

    def policy(observations, starts):
        finished[:] += np.asarray(starts)
        return rng.integers(0, 4, size=len(observations))

    outcomes = collect_episode_outcomes(envs, policy, n_episodes=24, seed=0)
    assert len(outcomes) == 24

    # The first call flags every slot as a start rather than a finish, and the
    # final step's flags are never shown to the policy because the loop exits
    # first — so the count trails the truth by one.
    finished -= 1
    per_env = 24 // 4
    assert (
        finished.min() >= per_env - 1
    ), f"collection stopped before the slowest environment had {per_env} episodes: {finished}"


def test_margin_bins_put_each_episode_in_exactly_one_band():
    """Boundary values belong to the band they open, not the one they close."""
    outcomes = [{"utility_margin": margin} for margin in (0.0, 0.049, 0.05, 0.15, 0.34, 0.35, 1.0)]
    bins = bin_by_margin(outcomes)

    assert [label for label, _ in bins] == ["<0.05", "0.05-0.15", "0.15-0.35", ">0.35"]
    assert [len(group) for _, group in bins] == [2, 1, 2, 2]
    assert sum(len(group) for _, group in bins) == len(outcomes)


def test_margin_bins_drop_ambiguous_levels():
    """Their margin is exactly zero and either choice is optimal, so binning
    them would fill the hardest band with coin flips."""
    outcomes = [
        {"utility_margin": 0.0, "is_ambiguous": True},
        {"utility_margin": 0.0, "is_ambiguous": False},
    ]
    bins = bin_by_margin(outcomes)
    assert [len(group) for _, group in bins] == [1, 0, 0, 0]


def test_margin_bins_reject_unordered_edges():
    with pytest.raises(ValueError, match="must increase"):
        bin_by_margin([], edges=(0.3, 0.1))


def test_the_psychometric_decisions_drop_episodes_with_no_trade_off():
    """A timeout, a value tie or a walled-off objective poses no trade-off, so
    including it would put a non-decision into a curve that measures how
    decisions are made."""
    outcomes = [
        # A real decision: the richer objective is 6 steps further and was taken anyway.
        {
            "reached_objective": True,
            "reached_feature_id": 0,
            "feature_0_value": 1.0,
            "feature_1_value": 0.5,
            "feature_0_distance": 14,
            "feature_1_distance": 8,
        },
        {
            "reached_objective": False,
            "feature_0_value": 1.0,
            "feature_1_value": 0.5,
            "feature_0_distance": 4,
            "feature_1_distance": 9,
        },
        {
            "reached_objective": True,
            "reached_feature_id": 1,
            "feature_0_value": 0.7,
            "feature_1_value": 0.7,
            "feature_0_distance": 4,
            "feature_1_distance": 9,
        },
        {
            "reached_objective": True,
            "reached_feature_id": 0,
            "feature_0_value": 1.0,
            "feature_1_value": 0.5,
            "feature_0_distance": 4,
            "feature_1_distance": -1,
        },
    ]
    gaps, took_richer, value_gaps = value_distance_decisions(outcomes)

    assert len(gaps) == 1, f"expected one usable decision, got {len(gaps)}"
    assert gaps[0] == 6.0
    assert took_richer[0] == 1.0
    assert value_gaps[0] == pytest.approx(0.5)


def test_the_indifference_point_is_recovered_from_a_known_threshold():
    """Synthesise an agent that switches at exactly 10 extra steps. If the fit
    cannot say 10, the number this measurement reports means nothing."""
    rng = np.random.default_rng(0)
    gaps = rng.uniform(-20, 30, 4000)
    took_richer = (gaps < 10).astype(float)

    assert indifference_point(gaps, took_richer) == pytest.approx(10.0, abs=0.7)


def test_on_shortest_route_counts_only_finished_episodes_that_took_no_detour():
    outcomes = [
        {"reached_objective": True, "reached_feature_id": 0, "feature_0_distance": 5, "episode_steps": 5},
        {"reached_objective": True, "reached_feature_id": 1, "feature_1_distance": 4, "episode_steps": 6},
        {"reached_objective": False, "episode_steps": 120},
    ]
    assert on_shortest_route(outcomes) == 0.5
    assert np.isnan(on_shortest_route(outcomes[2:]))


def test_the_undiscounted_threshold_is_the_limit_of_the_discounted_one() -> None:
    """Two hand-written gamma=1 branches once disagreed in sign with the formula
    they replaced; one function, checked against its own limit, prevents that."""
    for v0, v1 in [(1.0, 0.5), (0.5, 0.9), (1.0, 1.2)]:
        near = optimal_threshold(v0, v1, 0.05, 1 - 1e-7)
        assert optimal_threshold(v0, v1, 0.05, 1.0) == pytest.approx(near, abs=1e-3)
    assert optimal_threshold(1.0, 0.5, 0.05, 1.0) == pytest.approx(10.0)
    assert optimal_threshold(1.0, 1.2, 0.05, 1.0) < 0  # colour 1 is worth more: prefer it


def test_the_discounted_threshold_sits_below_the_undiscounted_one() -> None:
    """The DRC's discounted optimum is ~9.3 against the undiscounted 10 (CLAUDE.md)."""
    assert optimal_threshold(1.0, 0.5, 0.05, 0.995) == pytest.approx(9.3, abs=0.1)


def sharp_chooser(threshold: float, scale: float, seed: int, n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """Decisions of an agent switching at ``threshold`` with logistic ``scale``, gaps peaked at 0 as in a maze."""
    rng = np.random.default_rng(seed)
    gaps = np.round(rng.normal(0, 6, size=n))
    took_richer = (rng.random(n) < 1 / (1 + np.exp(-(threshold - gaps) / scale))).astype(float)
    return gaps, took_richer


def test_the_indifference_point_is_biased_high_for_far_sharp_thresholds() -> None:
    """Documents a known bias (audit 2026-10-26) rather than fixing it: recorded thresholds were computed with it.

    The descent stops short of a steep slope, which moves the crossing out. A
    Newton MLE on the same decisions does not. If this starts failing because
    the bias is gone, the estimator changed and recorded thresholds no longer
    reproduce.
    """
    from goalmisgen.analysis.closed_loop import logistic_fit

    def bias(threshold, scale):
        pairs = [
            (indifference_point(g, y), logistic_fit(g, y)[0])
            for g, y in (sharp_chooser(threshold, scale, s) for s in range(6))
        ]
        return np.mean([a - b for a, b in pairs]), np.mean([b for _, b in pairs])

    near, newton_near = bias(6.0, 0.3)
    assert abs(near) < 0.2 and newton_near == pytest.approx(6.0, abs=0.2)
    far, newton_far = bias(18.0, 1.0)
    assert far > 2.0 and newton_far == pytest.approx(18.0, abs=1.5)
    assert bias(14.0, 0.3)[0] > 0.5
