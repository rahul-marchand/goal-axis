"""Angle between the two colours' input-kernel signatures, per base.

The rotation edit turns one colour's slice towards the other's, so how much turn
is available before they coincide is set by the base angle between them. Weights
only: no environment, no rollout.
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/GoalMisgeneralisation")

from goalmisgen.analysis.input_gain import input_kernel  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402


def slices(params):
    k = input_kernel(params)
    flat = np.concatenate([np.asarray(leaf).ravel() for leaf in __import__("jax").tree_util.tree_leaves(params)])
    return k, {c: flat[k.plane(c)] for c in (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)}


def report(run: Path, offline: bool):
    payload = json.loads((run / "BASE.json").read_text())
    checkpoint = run / payload["checkpoint"]
    if offline:
        from goalmisgen.offline.model import load_model
        params = load_model(checkpoint).params
    else:
        from cleanba.cleanba_impala import load_train_state
        from goalmisgen.configs.env import MazeConfig
        values = tuple(payload["values"])
        cfg = MazeConfig(max_episode_steps=120, num_envs=2, min_size=11, max_size=11, n_objectives=len(values),
                         objective_values=values, feature_value_correlation=1.0, value_encoding="none",
                         colour_is_the_only_value_cue=True, asynchronous=False, seed=0)
        params = load_train_state(checkpoint, env_cfg=cfg)[3].params
    k, planes = slices(params)
    a, b = list(planes.values())
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"{run.name:28s} {k.path:52s} n={a.size:5d}  |K0|={np.linalg.norm(a):7.3f} |K1|={np.linalg.norm(b):7.3f}"
          f"  cos={cos:+.3f}  angle={np.degrees(np.arccos(np.clip(cos, -1, 1))):6.2f} deg")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        offline = arg.startswith("offline:")
        try:
            report(Path(arg.removeprefix("offline:")), offline)
        except Exception as exc:  # keep going across bases
            print(f"{arg}: {type(exc).__name__}: {exc}")
