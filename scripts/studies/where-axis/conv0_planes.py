"""Which input planes the DRC's input-conv axis acts on: squared length by plane, and the sign of the mean weight change per plane.

    uv run python scripts/studies/where-axis/conv0_planes.py novalue11g1.s1234 novalue11g1.s2468 novalue11g1.s3456e

Per base and sweep, the axis is fitted on the input convolution alone, from the
o arms at 400k steps inside |offset| <= 0.45; the shares and per-plane cosines
are written to figures/data/restricted/conv0_planes.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.paths import level_dataset, repo_root, runs_dir
from goalmisgen.volume import discover_arms

PLANES = ["wall", "agent", "colour 0 (1.0 goal)", "colour 1 (0.5 goal)"]


def input_conv(params) -> np.ndarray:
    return np.asarray(params["params"]["network_params"]["conv_list_0"]["kernel"], np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bases", nargs="+", help="DRC agent names under --runs.")
    parser.add_argument("--runs", type=Path, default=runs_dir())
    parser.add_argument("--levels", type=str, default=str(level_dataset()))
    parser.add_argument("--out", type=Path, default=repo_root() / "figures" / "data" / "restricted" / "conv0_planes.json")
    args = parser.parse_args()

    from goalmisgen.axis import load_base

    out = {}
    for base_name in args.bases:
        run = args.runs / base_name
        base = load_base(run, args.levels, split="test", num_envs=8, seed=0)
        values = base.values
        base_k = input_conv(base.params)
        out[base_name] = {}
        for sweep in (0, 1):
            arms = discover_arms(run / "arms", objective=sweep, base_value=values[sweep], steps=400_000)
            offs, diffs = [], []
            for v, cp in sorted(arms.items()):
                o = v - values[sweep]
                if abs(o) < 1e-9 or abs(o) > 0.45:
                    continue
                k = input_conv(base.other_params(cp))
                offs.append(o)
                diffs.append((k - base_k).ravel())
            axis, _ = fit_axis_and_drift(np.array(offs), np.stack(diffs))
            axis = axis.reshape(base_k.shape)  # (3,3,in,out)
            by_plane = (axis**2).sum(axis=(0, 1, 3))
            by_plane /= by_plane.sum()
            # alignment of each plane's change with the base kernel's same plane: + = amplify, - = attenuate
            align = [
                float(
                    (axis[:, :, p, :] * base_k[:, :, p, :]).sum()
                    / np.sqrt((axis[:, :, p, :] ** 2).sum() * (base_k[:, :, p, :] ** 2).sum())
                )
                for p in range(4)
            ]
            out[base_name][f"o{sweep}"] = {
                "share_by_plane": by_plane.tolist(),
                "cos_with_base_by_plane": align,
                "n_arms": len(offs),
            }
            print(
                f"{base_name} sweep o{sweep} ({len(offs)} arms): share of conv-0 axis by input plane "
                + ", ".join(f"{n} {s:.0%}" for n, s in zip(PLANES, by_plane))
            )
            print("    cos(axis, base kernel) per plane: " + ", ".join(f"{n} {a:+.2f}" for n, a in zip(PLANES, align)))
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
