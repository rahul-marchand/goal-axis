"""Does the rich/poor ratio of input-kernel signature lengths track the trained values?

If value is carried as magnitude in the input code, an arm trained at a lower
colour-1 value should write a shorter colour-1 signature. Weights only.
"""
import json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/GoalMisgeneralisation")

from goalmisgen.analysis.input_gain import input_kernel  # noqa: E402
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL  # noqa: E402


def load(checkpoint, values):
    from cleanba.cleanba_impala import load_train_state

    from goalmisgen.configs.env import MazeConfig
    cfg = MazeConfig(max_episode_steps=120, num_envs=2, min_size=11, max_size=11, n_objectives=len(values),
                     objective_values=values, feature_value_correlation=1.0, value_encoding="none",
                     colour_is_the_only_value_cue=True, asynchronous=False, seed=0)
    return load_train_state(checkpoint, env_cfg=cfg)[3].params


def measure(params):
    import jax
    k = input_kernel(params)
    flat = np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree_util.tree_leaves(params)])
    a, b = (flat[k.plane(c)] for c in (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb))
    return float(na), float(nb), cos


def latest(path: Path):
    cps = sorted(path.glob("local-files/cp_*")) or sorted(path.glob("cp_*"))
    return cps[-1] if cps else None


if __name__ == "__main__":
    run = Path(sys.argv[1])
    payload = json.loads((run / "BASE.json").read_text())
    base_values = tuple(payload["values"])
    na, nb, cos = measure(load(run / payload["checkpoint"], base_values))
    print(f"{'arm':16s} {'v1':>5s} {'|K0|':>8s} {'|K1|':>8s} {'ratio':>7s} {'cos':>7s} {'angle':>7s}")
    print(f"{'base':16s} {base_values[1]:5.2f} {na:8.3f} {nb:8.3f} {na / nb:7.3f} {cos:+7.3f} {np.degrees(np.arccos(cos)):7.2f}")
    for arm in sorted((run / "arms").glob("o1*@400k")):
        m = re.search(r"o1([+-])(\d+)", arm.name)
        offset = (1 if m.group(1) == "+" else -1) * int(m.group(2)) / 100
        v1 = base_values[1] + offset
        cp = latest(arm)
        if cp is None:
            print(f"{arm.name:16s} {v1:5.2f}   no checkpoint")
            continue
        na, nb, cos = measure(load(cp, (base_values[0], v1)))
        print(f"{arm.name:16s} {v1:5.2f} {na:8.3f} {nb:8.3f} {na / nb:7.3f} {cos:+7.3f} {np.degrees(np.arccos(cos)):7.2f}")
