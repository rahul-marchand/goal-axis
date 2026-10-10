"""What a saved agent was trained with, read from the ``cfg.json`` beside its weights.

Anything that rebuilds an agent -- to fine-tune it, probe it or decode it --
has to agree with the weights about the observation format and the network,
and a fine-tune also has to keep the objective. Those belong to the agent, not
to whichever script is loading it, so they are read from the run's own record
rather than assumed. Every reader here returns a conservative answer when the
record is missing, and says which.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import farconf
from cleanba.network import PolicySpec


def saved_config(checkpoint: Path) -> dict[str, Any] | None:
    """The configuration saved beside a checkpoint, or ``None`` if it is unreadable.

    cleanba nests it under ``cfg``; older copies written by hand do not.
    """
    try:
        payload = json.loads((Path(checkpoint) / "cfg.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("cfg", payload)


def hides_values(checkpoint: Path) -> bool:
    """Whether the agent was trained without a value channel.

    A config with the wrong channel count builds a network the saved weights
    cannot fill -- fine-tuning ``maze11``, which has a value channel, as if it
    were ``novalue11`` died on
    ``ScopeParamShapeError: expected (3, 3, 5, 32) but got (3, 3, 4, 32)``.

    ``True`` when the record is unreadable: every agent swept before this was
    read had no value channel.
    """
    config = saved_config(checkpoint)
    if config is None:
        return True
    env = config.get("train_env", {})
    # ``value_encoding`` is the field to read: every run records it, and it says
    # directly whether the observation carries a value channel. The first attempt
    # at this read ``colour_is_the_only_value_cue``, which maze11's config does
    # not contain at all -- so the lookup returned its default and the fix
    # silently kept the behaviour it was meant to correct.
    encoding = env.get("value_encoding")
    if encoding is not None:
        return encoding == "none"
    return bool(env.get("colour_is_the_only_value_cue", True))


def discount(checkpoint: Path) -> float | None:
    """The discount the agent was trained with, or ``None`` if unrecorded.

    A fine-tune is a small nudge to the base's own objective: fine-tuning a
    gamma=1.0 agent at the preset's 0.995 would quietly optimise a different
    exchange rate than the one the base learned -- the quantity a value sweep
    exists to measure.
    """
    config = saved_config(checkpoint)
    if config is None:
        return None
    gamma = config.get("loss", {}).get("gamma")
    return float(gamma) if gamma is not None else None


def trained_net(checkpoint: Path) -> PolicySpec | None:
    """The network spec the agent was trained with, or ``None`` if unrecorded.

    The saved spec itself rather than a preset looked up by class, so a
    transformer of a non-default size comes back at its own size. A spec naming
    a different network than the weights were trained with dies on a parameter
    shape error -- or, for a head of the same shape, does not die.
    """
    config = saved_config(checkpoint)
    if config is None or config.get("net") is None:
        return None
    return farconf.from_dict(config["net"], PolicySpec)


def legacy_observation(checkpoint: Path) -> str | None:
    """The observation an ``OBSERVATION`` file records for a run that predates the config field.

    The legend runs were trained with an import hook switching every
    ``MazeConfig`` to the legend, so their ``cfg.json`` says nothing about it;
    ``scripts/pods/legend_chain.sh`` wrote ``OBSERVATION`` into the agent's
    directory instead. A checkpoint sits at ``<agent>/local-files/cp_*``, or at
    ``<agent>/arms/<arm>/local-files/cp_*`` for one of its arms.
    """
    checkpoint = Path(checkpoint)
    candidates = [checkpoint.parent.parent]
    if len(checkpoint.parents) > 3 and checkpoint.parents[2].name == "arms":
        candidates.append(checkpoint.parents[3])
    for directory in candidates:
        marker = directory / "OBSERVATION"
        if marker.is_file():
            return marker.read_text().strip()
    return None


def observation(checkpoint: Path) -> str:
    """How the agent's levels were drawn: ``MazeConfig.observation``, from the record or its legacy marker."""
    config = saved_config(checkpoint)
    recorded = (config or {}).get("train_env", {}).get("observation")
    return recorded or legacy_observation(checkpoint) or "planes"


def train_env(checkpoint: Path):
    """The training environment's :class:`~goalmisgen.configs.env.MazeConfig`, or ``None`` if unrecorded.

    Decoded the way ``load_train_state`` decodes the whole record, so it is the
    configuration the weights were trained under -- observation format, value
    encoding, sizes, step penalty, episode limit -- rather than one rebuilt from
    literals in whichever script is reading the agent. A record from before
    ``observation`` was a field takes it from :func:`legacy_observation`.
    """
    import dataclasses

    from cleanba.config import Args

    config = saved_config(checkpoint)
    if config is None or config.get("train_env") is None:
        return None
    env = farconf.from_dict(config, Args).train_env
    if "observation" not in config["train_env"]:
        legacy = legacy_observation(checkpoint)
        if legacy is not None:
            env = dataclasses.replace(env, observation=legacy)
    return env
