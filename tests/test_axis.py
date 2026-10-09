"""The RL axis library: evaluation configs built from a run's own record, and the base marker.

The scripts this library replaced built their evaluation ``MazeConfig`` from
literals -- size 11, 120 steps, no value channel, correlation 1.0 -- that were
true of every base they were pointed at. :func:`goalmisgen.axis.eval_config`
builds it from the run's saved ``train_env`` instead, and a migrated script has
to reproduce what it measured before. So for the saved configurations of the
bases actually swept -- written the way ``001`` and ``013`` write them -- the
two must be the *same* config, field for field.
"""

from __future__ import annotations

import json
from pathlib import Path

import farconf
import pytest
from cleanba.config import Args

from goalmisgen.axis import checkpoint_eval_config, eval_config
from goalmisgen.configs import saved
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import preset_for, with_final_checkpoint
from goalmisgen.volume import read_base_marker

TRAIN_LEVELS = "/workspace/data/levels/values/1.00-0.50@1M"
LEVELS = "/workspace/data/levels/values/1.00-0.50@500k"
VALUES = (1.0, 0.5)


def g1_base(net: str, seed: int = 1234) -> Args:
    """A g1 base as 001 built it: ``--net NET --correlation 1.0 --min-size 11 --max-size 11 --hide-values --gamma 1.0``."""
    return with_final_checkpoint(
        preset_for(net)(
            feature_value_correlation=1.0,
            gamma=1.0,
            min_size=11,
            max_size=11,
            total_timesteps=150_000_000,
            seed=seed,
            level_dataset=TRAIN_LEVELS,
            hide_values=True,
        )
    )


def save(tmp_path: Path, name: str, config: Args) -> Path:
    """A checkpoint directory holding cfg.json exactly as cleanba writes it."""
    checkpoint = tmp_path / name / "local-files" / "cp_140206080"
    checkpoint.mkdir(parents=True)
    (checkpoint / "cfg.json").write_text(json.dumps({"cfg": farconf.to_dict(config, Args), "update_step": 1}))
    return checkpoint


def literal(num_envs: int, split: str, seed: int, values=VALUES, max_episode_steps: int = 120) -> MazeConfig:
    """The literal every migrated script wrote (restricted_writes, input_gain, axis_far_dose, ...)."""
    return MazeConfig(
        max_episode_steps=max_episode_steps,
        num_envs=num_envs,
        min_size=11,
        max_size=11,
        n_objectives=len(values),
        objective_values=values,
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=LEVELS,
        dataset_split=split,
        asynchronous=False,
        seed=seed,
    )


def literal_014(num_envs: int = 64, seed: int = 0) -> MazeConfig:
    """014's eval_config: no n_objectives, the values written out."""
    return MazeConfig(
        max_episode_steps=120,
        num_envs=num_envs,
        min_size=11,
        max_size=11,
        feature_value_correlation=1.0,
        objective_values=(1.0, 0.5),
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=LEVELS,
        dataset_split="test",
        asynchronous=False,
        seed=seed,
    )


@pytest.mark.parametrize("net", ["drc33", "resnet", "vit"])
def test_the_g1_bases_evaluate_in_the_literal_config(tmp_path, net) -> None:
    checkpoint = save(tmp_path, f"{net}.g1", g1_base(net))
    train_env = saved.train_env(checkpoint)
    for num_envs, split, seed in ((64, "test", 0), (2, "valid", 3), (8, "test", 0), (128, "train", 7)):
        assert eval_config(train_env, LEVELS, split=split, num_envs=num_envs, seed=seed, values=VALUES) == literal(
            num_envs, split, seed
        )
    assert checkpoint_eval_config(checkpoint, LEVELS, num_envs=64, seed=0) == literal_014()
    assert eval_config(train_env, LEVELS, num_envs=1, max_episode_steps=200) == literal(1, "test", 0, max_episode_steps=200)


def test_a_tail_lr_extension_evaluates_the_same(tmp_path) -> None:
    """``s3456e``-style bases are a 013 continuation; their cfg.json is 013's, not 001's."""
    from goalmisgen.finetune import finetune_config, parse_args

    base = save(tmp_path, "novalue11g1.s3456", g1_base("drc33", seed=3456))
    args = parse_args(
        [str(base), "--objective-values", "1.0", "0.5", "--levels", TRAIN_LEVELS, "--run-dir", str(tmp_path / "e")]
        + ["--steps", "100000000", "--lr", "4e-6", "--checkpoints", "8", "--seed", "42"]
    )
    extension = save(tmp_path, "novalue11g1.s3456e", finetune_config(args))
    assert checkpoint_eval_config(extension, LEVELS, num_envs=64, seed=0) == literal_014()


def test_a_literal_without_a_seed_differs_only_in_the_seed(tmp_path) -> None:
    """cheapest_edit and its siblings set no seed, so cleanba drew one; nothing they built ever stepped on it."""
    train_env = saved.train_env(save(tmp_path, "b", g1_base("drc33")))
    built = eval_config(train_env, LEVELS, num_envs=1, seed=None)
    assert built == literal(1, "test", built.seed)


def test_a_base_with_a_value_channel_keeps_it(tmp_path) -> None:
    """The literal would have stripped it; the run's own record does not."""
    config = with_final_checkpoint(preset_for("drc33")(min_size=11, max_size=11, level_dataset=TRAIN_LEVELS))
    train_env = saved.train_env(save(tmp_path, "maze11", config))
    assert eval_config(train_env, LEVELS).value_encoding == "at_objective"


def test_the_base_marker_names_the_checkpoint(tmp_path) -> None:
    run = tmp_path / "novalue11g1.s1234"
    run.mkdir()
    (run / "BASE.json").write_text(json.dumps({"checkpoint": "local-files/cp_120000000", "values": [1.0, 0.5], "steps": 1}))
    marker = read_base_marker(run)
    assert marker.checkpoint == run / "local-files" / "cp_120000000"
    assert marker.values == (1.0, 0.5)
    assert marker.payload["steps"] == 1
    with pytest.raises(FileNotFoundError, match="BASE.json"):
        read_base_marker(tmp_path / "nothing")
