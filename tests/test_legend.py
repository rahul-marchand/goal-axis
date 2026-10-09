"""Tests for the legend observation, the positive control for a factorised value circuit.

The claims the control rests on are checked directly: colour is gone from the
objectives, the cue sits where no first-layer kernel over the objective can see
it, which objective carries value changes only the legend, and nothing about
the stored levels (and so their fingerprint) moves.
"""

from __future__ import annotations

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


def test_the_config_field_switches_the_encoder_and_leaves_the_sampler_alone():
    from goalmisgen.envs.dataset import dataset_fingerprint
    from goalmisgen.envs.observation import ObservationEncoder

    planes, legend = hidden_config(), hidden_config(observation="legend")
    assert isinstance(planes.encoder(), ObservationEncoder)
    assert isinstance(legend.encoder(), LegendEncoder)
    assert dataset_fingerprint(legend.live_sampler()) == dataset_fingerprint(planes.live_sampler())

    env = legend.build_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, 8, 4)
    assert obs[:, :, LEGEND_CHANNEL].sum() == 2
    assert obs[:, :, OBJECTIVE_CHANNEL].sum() == 2

    envs = legend.make()
    batch, _ = envs.reset()
    assert batch.shape == (1, 4, 8, 8)  # NCHW after cleanba's wrapper


def test_the_legend_refuses_a_visible_value_channel():
    with pytest.raises(ValueError, match="hidden-value control"):
        hidden_config(observation="legend", value_encoding="at_objective", colour_is_the_only_value_cue=False).encoder()


def test_an_unknown_observation_is_refused():
    with pytest.raises(ValueError, match="observation"):
        hidden_config(observation="pixels")


def save_run(root: Path, observation: str | None, marker: str | None = None) -> Path:
    """A run's checkpoint with cfg.json as 001 writes it; ``observation=None`` drops the field, as old runs lack it."""
    import json

    import farconf
    from cleanba.config import Args

    from goalmisgen.configs.presets import maze_drc33

    config = maze_drc33(min_size=11, max_size=11, hide_values=True, observation=observation or "planes")
    payload = farconf.to_dict(config, Args)
    if observation is None:
        del payload["train_env"]["observation"]
    checkpoint = root / "local-files" / "cp_150000000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "cfg.json").write_text(json.dumps({"cfg": payload, "update_step": 1}))
    if marker is not None:
        (root / "OBSERVATION").write_text(marker + "\n")
    return checkpoint


def test_the_observation_round_trips_through_cfg_json(tmp_path):
    from goalmisgen.configs import saved

    checkpoint = save_run(tmp_path / "legend", "legend")
    assert saved.train_env(checkpoint).observation == "legend"
    assert saved.observation(checkpoint) == "legend"
    assert isinstance(saved.train_env(checkpoint).encoder(), LegendEncoder)


def test_a_record_from_before_the_field_still_loads(tmp_path):
    """Every run trained before the field existed has no ``observation`` in its cfg.json."""
    from goalmisgen.configs import saved

    checkpoint = save_run(tmp_path / "old", None)
    assert saved.train_env(checkpoint).observation == "planes"
    assert saved.observation(checkpoint) == "planes"


def test_a_hook_era_legend_run_is_read_through_its_observation_file(tmp_path):
    """The legend runs made with the import hook recorded it only in OBSERVATION beside BASE.json."""
    from goalmisgen.configs import saved

    agent = tmp_path / "legend11g1.s1234"
    checkpoint = save_run(agent, None, marker="legend")
    assert saved.train_env(checkpoint).observation == "legend"
    arm = agent / "arms" / "o1+020@400k" / "local-files" / "cp_400000"
    arm.mkdir(parents=True)
    (arm / "cfg.json").write_text((checkpoint / "cfg.json").read_text())
    assert saved.observation(arm) == "legend"


def test_a_legend_base_makes_legend_arms(tmp_path):
    from goalmisgen.finetune import finetune_config, parse_args

    agent = tmp_path / "legend11g1.s1234"
    checkpoint = save_run(agent, None, marker="legend")
    args = parse_args([str(checkpoint), "--value", "0.7", "--levels", "/tmp/levels", "--run-dir", str(tmp_path / "arm")])
    assert finetune_config(args).train_env.observation == "legend"
    plain = save_run(tmp_path / "plain", None)
    args = parse_args([str(plain), "--value", "0.7", "--levels", "/tmp/levels", "--run-dir", str(tmp_path / "arm2")])
    assert finetune_config(args).train_env.observation == "planes"
