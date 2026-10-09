"""Tests for the legend observation, the positive control for a factorised value circuit.

The claims the control rests on are checked directly: colour is gone from the
objectives, the cue sits where no first-layer kernel over the objective can see
it, which objective carries value changes only the legend, and nothing about
the stored levels (and so their fingerprint) moves.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from goalmisgen.envs.legend import LEGEND_CHANNEL, OBJECTIVE_CHANNEL, LegendEncoder
from goalmisgen.envs.level import Level, Objective
from goalmisgen.envs.observation import AGENT_CHANNEL, WALL_CHANNEL

ROOT = Path(__file__).resolve().parent.parent
ROOM = np.ones((7, 7), dtype=np.bool_)
ROOM[1:6, 1:6] = False


def level(rich=(1, 4), poor=(4, 2)) -> Level:
    return Level(
        walls=ROOM,
        agent_start=(3, 3),
        objectives=(Objective(rich, value=1.0, feature_id=0), Objective(poor, value=0.5, feature_id=1)),
    )


def test_shape_is_one_wider_with_four_channels():
    encoder = LegendEncoder(max_size=11)
    assert encoder.shape == (12, 12, 4)
    assert encoder.space().shape == (12, 12, 4)
    assert encoder.space().contains(encoder.encode(level(), (3, 3)))


def test_objectives_share_one_uncoloured_plane():
    obs = LegendEncoder(max_size=11).encode(level(), (3, 3))
    assert obs[1, 4, OBJECTIVE_CHANNEL] == 1.0 and obs[4, 2, OBJECTIVE_CHANNEL] == 1.0
    assert obs[:, :, OBJECTIVE_CHANNEL].sum() == 2
    assert obs[3, 3, AGENT_CHANNEL] == 1.0 and obs[:, :, AGENT_CHANNEL].sum() == 1


def test_legend_marks_row_and_column_of_the_feature_zero_objective_in_the_border():
    obs = LegendEncoder(max_size=11).encode(level(rich=(1, 4)), (3, 3))
    marks = {tuple(int(i) for i in cell) for cell in np.argwhere(obs[:, :, LEGEND_CHANNEL] > 0)}
    assert marks == {(1, 11), (11, 4)}
    # Both marks sit in wall padding, outside the maze.
    assert all(obs[r, c, WALL_CHANNEL] == 1.0 for r, c in marks)


def test_legend_is_beyond_a_3x3_kernel_of_the_objective_it_names():
    """A mark inside one kernel of the objective would let a first-layer gain act on it."""
    encoder = LegendEncoder(max_size=11)
    for rich in [(1, 1), (1, 5), (5, 5), (5, 1)]:
        obs = encoder.encode(level(rich=rich, poor=(3, 1) if rich != (3, 1) else (1, 3)), (3, 3))
        for r, c in np.argwhere(obs[:, :, LEGEND_CHANNEL] > 0):
            assert max(abs(r - rich[0]), abs(c - rich[1])) > 1


def test_swapping_which_objective_pays_changes_only_the_legend():
    encoder = LegendEncoder(max_size=11)
    a = encoder.encode(level(rich=(1, 4), poor=(4, 2)), (3, 3))
    b = encoder.encode(level(rich=(4, 2), poor=(1, 4)), (3, 3))
    for channel in (WALL_CHANNEL, AGENT_CHANNEL, OBJECTIVE_CHANNEL):
        np.testing.assert_array_equal(a[:, :, channel], b[:, :, channel])
    assert not np.array_equal(a[:, :, LEGEND_CHANNEL], b[:, :, LEGEND_CHANNEL])


def test_no_value_reaches_the_observation():
    encoder = LegendEncoder(max_size=11)
    base = level()
    richer = Level(
        walls=ROOM,
        agent_start=(3, 3),
        objectives=(Objective((1, 4), value=0.7, feature_id=0), Objective((4, 2), value=0.2, feature_id=1)),
    )
    np.testing.assert_array_equal(encoder.encode(base, (3, 3)), encoder.encode(richer, (3, 3)))


def test_rejects_levels_without_a_single_feature_zero_objective():
    bad = Level(
        walls=ROOM,
        agent_start=(3, 3),
        objectives=(Objective((1, 4), value=1.0, feature_id=1), Objective((4, 2), value=0.5, feature_id=1)),
    )
    with pytest.raises(ValueError, match="exactly one"):
        LegendEncoder(max_size=11).encode(bad, (3, 3))


def hidden_config(**overrides):
    from goalmisgen.configs.env import MazeConfig

    settings = dict(
        num_envs=1,
        min_size=7,
        max_size=7,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        asynchronous=False,
        max_episode_steps=60,
    )
    settings.update(overrides)
    return MazeConfig(**settings)


def test_install_switches_every_maze_config_and_leaves_the_sampler_alone(monkeypatch):
    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.envs import legend
    from goalmisgen.envs.dataset import dataset_fingerprint

    config = hidden_config()
    fingerprint = dataset_fingerprint(config.live_sampler())
    monkeypatch.setattr(MazeConfig, "encoder", MazeConfig.encoder)  # restored after the test
    legend.install()
    assert isinstance(config.encoder(), LegendEncoder)
    assert dataset_fingerprint(config.live_sampler()) == fingerprint

    env = config.build_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, 8, 4)
    assert obs[:, :, LEGEND_CHANNEL].sum() == 2
    assert obs[:, :, OBJECTIVE_CHANNEL].sum() == 2

    envs = config.make()
    batch, _ = envs.reset()
    assert batch.shape == (1, 4, 8, 8)  # NCHW after cleanba's wrapper


def test_install_refuses_a_visible_value_channel(monkeypatch):
    from goalmisgen.configs.env import MazeConfig
    from goalmisgen.envs import legend

    monkeypatch.setattr(MazeConfig, "encoder", MazeConfig.encoder)
    legend.install()
    with pytest.raises(ValueError, match="hidden-value control"):
        hidden_config(value_encoding="at_objective", colour_is_the_only_value_cue=False).encoder()


def run_python(code: str, with_site: bool) -> str:
    env = dict(os.environ)
    path = [str(ROOT / "scripts" / "legend_site")] if with_site else []
    env["PYTHONPATH"] = os.pathsep.join(path + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=ROOT).stdout


def test_site_hook_reaches_a_fresh_interpreter_only_when_on_the_path():
    code = (
        "import sys\n"
        "assert 'jax' not in sys.modules, 'the hook must not import JAX at start-up'\n"
        "from goalmisgen.configs.env import MazeConfig\n"
        "c = MazeConfig(max_episode_steps=60, num_envs=1, min_size=7, max_size=7, value_encoding='none', colour_is_the_only_value_cue=True)\n"
        "print(type(c.encoder()).__name__)\n"
    )
    assert run_python(code, with_site=True).strip() == "LegendEncoder"
    assert run_python(code, with_site=False).strip() == "ObservationEncoder"
