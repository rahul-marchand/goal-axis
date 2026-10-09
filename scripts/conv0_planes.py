"""Which input planes the DRC's input-conv axis acts on: squared length by plane, and the sign of the mean weight change per plane."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/GoalMisgeneralisation")
from cleanba.cleanba_impala import load_train_state
from jax.flatten_util import ravel_pytree
from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.configs.env import MazeConfig
from goalmisgen.volume import discover_arms
PLANES = ["wall", "agent", "colour 0 (1.0 goal)", "colour 1 (0.5 goal)"]
out = {}
for base_name in sys.argv[1:]:
    run = Path("/workspace/data/runs") / base_name
    payload = json.loads((run / "BASE.json").read_text()); values = tuple(payload["values"])
    config = MazeConfig(max_episode_steps=120, num_envs=8, min_size=11, max_size=11, n_objectives=2, objective_values=values,
        feature_value_correlation=1.0, value_encoding="none", colour_is_the_only_value_cue=True,
        level_dataset="/workspace/data/levels/values/1.00-0.50@500k", dataset_split="test", asynchronous=False, seed=0)
    _, _, _, base_state, _ = load_train_state(run / payload["checkpoint"], env_cfg=config)
    base_k = np.asarray(base_state.params["params"]["network_params"]["conv_list_0"]["kernel"], np.float64)
    out[base_name] = {}
    for sweep in (0, 1):
        arms = discover_arms(run / "arms", objective=sweep, base_value=values[sweep], steps=400_000)
        offs, diffs = [], []
        for v, cp in sorted(arms.items()):
            o = v - values[sweep]
            if abs(o) < 1e-9 or abs(o) > 0.45: continue
            _, _, _, st, _ = load_train_state(cp, env_cfg=config)
            k = np.asarray(st.params["params"]["network_params"]["conv_list_0"]["kernel"], np.float64)
            offs.append(o); diffs.append((k - base_k).ravel())
        axis, _ = fit_axis_and_drift(np.array(offs), np.stack(diffs))
        axis = axis.reshape(base_k.shape)  # (3,3,in,out)
        by_plane = (axis**2).sum(axis=(0,1,3)); by_plane /= by_plane.sum()
        centre = axis[1,1]  # centre tap, (in, out)
        # alignment of each plane's change with the base kernel's same plane: + = amplify, - = attenuate
        align = [float((axis[:,:,p,:]*base_k[:,:,p,:]).sum() / np.sqrt((axis[:,:,p,:]**2).sum()*(base_k[:,:,p,:]**2).sum())) for p in range(4)]
        out[base_name][f"o{sweep}"] = {"share_by_plane": by_plane.tolist(), "cos_with_base_by_plane": align, "n_arms": len(offs)}
        print(f"{base_name} sweep o{sweep} ({len(offs)} arms): share of conv-0 axis by input plane " + ", ".join(f"{n} {s:.0%}" for n, s in zip(PLANES, by_plane)))
        print(f"    cos(axis, base kernel) per plane: " + ", ".join(f"{n} {a:+.2f}" for n, a in zip(PLANES, align)))
Path("/workspace/GoalMisgeneralisation/figures/data/restricted/conv0_planes.json").write_text(json.dumps(out, indent=2))
