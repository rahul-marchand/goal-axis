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
