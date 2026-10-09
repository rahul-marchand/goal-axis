"""Colour-signature length and angle for the route model, base and arms.

    uv run python scripts/bc_angle.py /workspace/data/offline/runs/bcnv11.s1

Same measurement as arm_norms.py, through the offline loader: `cell_in` is the
route model's input kernel, shape (4, width), so a colour's signature is one
row of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL


def signature(params) -> tuple[float, float, float]:
    kernel = np.asarray(params["params"]["cell_in"]["kernel"])
    a = kernel[FIRST_FEATURE_CHANNEL].ravel()
    b = kernel[FIRST_FEATURE_CHANNEL + 1].ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb))
    return float(na), float(nb), cos


def line(label, v1, params) -> None:
    na, nb, cos = signature(params)
    print(
        f"{label:22s} {v1:>5} {na:8.4f} {nb:8.4f} {na / nb:7.3f} {cos:+7.3f} "
        f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):7.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="Route-model run directories, arms under <run>/arms/.")
    args = parser.parse_args()

    from goalmisgen.offline.axis import load_base
    from goalmisgen.offline.train import list_checkpoints, load_checkpoint

    for run in args.runs:
        print(f"### {run.name}")
        print(f"{'arm':22s} {'v1':>5s} {'|K0|':>8s} {'|K1|':>8s} {'ratio':>7s} {'cos':>7s} {'angle':>7s}")
        try:
            base = load_base(run)
            line("base", base.values[1], base.params)
            arms = sorted((run / "arms").glob("*")) if (run / "arms").is_dir() else []
            for arm in arms:
                cps = list_checkpoints(arm)
                if not cps:
                    continue
                _, params = load_checkpoint(cps[-1][1])
                line(arm.name, arm.name, params)
        except Exception as exc:
            print(f"  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
