"""Tests for the fine-tune script's command line.

These exist because the three-objective sweep died on its first arm: --value was
still required while the three-objective path passes only --objective-values, so
every invocation failed in argparse before doing anything. The configuration
builder had been tested by handing it a Namespace directly, which is exactly the
path that cannot catch this — it skips the parser entirely.

So the rule these encode is that the *entry point* gets tested, not just the
function behind it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from goalmisgen.configs import saved

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "013_value_axis.py"
spec = importlib.util.spec_from_file_location("value_axis", SCRIPT)
assert spec is not None and spec.loader is not None
value_axis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(value_axis)


def run(argv: list[str]):
    sys.argv = ["013_value_axis.py", *argv]
    return value_axis.parse_args()


BASE = ["cp", "--levels", "/tmp/levels", "--run-dir", "/tmp/run"]


def test_two_objectives_by_value_alone():
    args = run([*BASE, "--value", "0.7"])
    assert value_axis.finetune_config(args).train_env.objective_values == (1.0, 0.7)


def test_three_objectives_without_value():
    """The invocation the sweep actually makes, and that used to fail outright."""
    args = run([*BASE, "--objective-values", "1.0", "0.65", "0.3"])
    env = value_axis.finetune_config(args).train_env
    assert env.objective_values == (1.0, 0.65, 0.3)
    assert env.n_objectives == 3


def test_objective_values_wins_over_value():
    args = run([*BASE, "--value", "0.7", "--objective-values", "1.0", "0.5", "0.2"])
    assert value_axis.finetune_config(args).train_env.objective_values == (1.0, 0.5, 0.2)


def test_neither_is_refused():
    with pytest.raises(SystemExit):
        run(BASE)


def test_a_single_objective_value_is_refused():
    with pytest.raises(SystemExit):
        run([*BASE, "--objective-values", "1.0"])


def test_the_agent_never_sees_a_value_channel():
    """The whole design: colour is the only cue to what an objective is worth."""
    args = run([*BASE, "--objective-values", "1.0", "0.65", "0.3"])
    env = value_axis.finetune_config(args).train_env
    assert env.value_encoding == "none"
    assert env.colour_is_the_only_value_cue is True


def test_the_value_channel_is_read_from_the_base_not_assumed(tmp_path) -> None:
    """013 was written for novalue11 and hardcoded hide_values=True, so
    fine-tuning maze11 -- which has a value channel -- built a four-channel
    network against five-channel weights and died on a shape error."""
    import json

    novalue = tmp_path / "novalue"
    novalue.mkdir()
    (novalue / "cfg.json").write_text(json.dumps({"cfg": {"train_env": {"colour_is_the_only_value_cue": True}}}))
    withvalue = tmp_path / "withvalue"
    withvalue.mkdir()
    (withvalue / "cfg.json").write_text(json.dumps({"cfg": {"train_env": {"colour_is_the_only_value_cue": False}}}))

    assert saved.hides_values(novalue) is True
    assert saved.hides_values(withvalue) is False


def test_an_unreadable_base_config_keeps_the_old_behaviour(tmp_path) -> None:
    """Every existing sweep was on an agent without a value channel."""
    assert saved.hides_values(tmp_path) is True


def test_an_agent_recording_only_value_encoding_is_read_correctly(tmp_path) -> None:
    """maze11's config has no colour_is_the_only_value_cue key at all, so a
    lookup with a default silently returned the wrong answer. value_encoding is
    recorded by every run and says directly whether there is a value channel."""
    import json

    withvalue = tmp_path / "withvalue"
    withvalue.mkdir()
    (withvalue / "cfg.json").write_text(json.dumps({"cfg": {"train_env": {"value_encoding": "at_objective"}}}))
    novalue = tmp_path / "novalue"
    novalue.mkdir()
    (novalue / "cfg.json").write_text(json.dumps({"cfg": {"train_env": {"value_encoding": "none"}}}))

    assert saved.hides_values(withvalue) is False
    assert saved.hides_values(novalue) is True


def test_the_network_is_read_from_the_base_not_assumed(tmp_path) -> None:
    """013 built every arm with maze_drc33. For a ResNet or transformer base the
    reset checkpoint would then pair DRC configuration with non-DRC weights.
    The saved spec is used as is, so a non-default transformer keeps its size."""
    import json

    import farconf
    from cleanba.convlstm import ConvLSTMConfig

    from goalmisgen.configs.presets import maze_resnet, maze_transformer_large
    from goalmisgen.nets.scaled import ScaledInputSpec
    from goalmisgen.nets.transformer import TransformerSpec

    for name, net in {
        "resnet": maze_resnet().net,
        "vitl": maze_transformer_large().net,
        "drc": ConvLSTMConfig(),
    }.items():
        base = tmp_path / name
        base.mkdir()
        payload = {"cfg": {"net": farconf.to_dict(net), "train_env": {"value_encoding": "none"}}}
        (base / "cfg.json").write_text(json.dumps(payload))
        args = run([str(base), "--value", "0.7", "--levels", "/tmp/levels", "--run-dir", "/tmp/run"])
        config = value_axis.finetune_config(args)
        assert config.net == net, name
        assert config.train_env.objective_values == (1.0, 0.7)
        assert config.train_env.value_encoding == "none"
    assert isinstance(maze_resnet().net, ScaledInputSpec)
    assert isinstance(maze_transformer_large().net, TransformerSpec)
    assert maze_transformer_large().net.d_model != TransformerSpec().d_model


def test_an_unreadable_base_config_builds_the_drc(tmp_path) -> None:
    from cleanba.convlstm import ConvLSTMConfig

    assert saved.trained_net(tmp_path) is None
    args = run([str(tmp_path), "--value", "0.7", "--levels", "/tmp/levels", "--run-dir", "/tmp/run"])
    assert isinstance(value_axis.finetune_config(args).net, ConvLSTMConfig)


def test_gamma_is_read_from_the_base_not_the_preset(tmp_path) -> None:
    """An arm is a nudge to the base's own objective: fine-tuning a gamma=1.0
    agent at the preset's 0.995 would quietly optimise a different exchange
    rate than the one being measured, and reset_checkpoint rewrites cfg.json
    so cleanba's config-diff at resume cannot catch it."""
    import json

    base = tmp_path / "undiscounted"
    base.mkdir()
    (base / "cfg.json").write_text(json.dumps({"cfg": {"loss": {"gamma": 1.0}}}))

    args = run([str(base), "--levels", "/tmp/levels", "--run-dir", "/tmp/run", "--value", "0.7"])
    assert value_axis.finetune_config(args).loss.gamma == 1.0


def test_a_base_without_a_recorded_gamma_keeps_the_preset_default(tmp_path) -> None:
    args = run([str(tmp_path), "--levels", "/tmp/levels", "--run-dir", "/tmp/run", "--value", "0.7"])
    from goalmisgen.configs.presets import maze_drc33

    assert value_axis.finetune_config(args).loss.gamma == maze_drc33().loss.gamma
