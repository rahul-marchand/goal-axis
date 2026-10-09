"""Angle between the two colours' input-kernel signatures, per base.

    uv run python scripts/colour_angle.py /workspace/data/runs/novalue11g1.s1234 \
        offline:/workspace/data/offline/runs/bcnv11.s1

The rotation edit turns one colour's slice towards the other's, so how much turn
is available before they coincide is set by the base angle between them. Weights
only: no environment, no rollout. A base prefixed ``offline:`` is a route model,
read at its last checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.analysis.input_gain import input_kernel
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL


def slices(params):
    import jax

    k = input_kernel(params)
    flat = np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree_util.tree_leaves(params)])
    return k, {c: flat[k.plane(c)] for c in (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)}


def report(run: Path, offline: bool) -> None:
    if offline:
        # The route model's own loader; this branch used to ask for an
        # offline.model.load_model that never existed.
        from goalmisgen.offline.axis import load_base as load_route_model

        params = load_route_model(run).params
    else:
        from goalmisgen.axis import load_base

        # Weights only: two live environments, built to load and never stepped.
        params = load_base(run, None, split="train", num_envs=2, seed=0).params
    k, planes = slices(params)
    a, b = list(planes.values())
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(
        f"{run.name:28s} {k.path:52s} n={a.size:5d}  |K0|={np.linalg.norm(a):7.3f} |K1|={np.linalg.norm(b):7.3f}"
        f"  cos={cos:+.3f}  angle={np.degrees(np.arccos(np.clip(cos, -1, 1))):6.2f} deg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bases", nargs="+", help="Agent directories; prefix a route model's with offline:.")
    args = parser.parse_args()
    for arg in args.bases:
        offline = arg.startswith("offline:")
        try:
            report(Path(arg.removeprefix("offline:")), offline)
        except Exception as exc:  # keep going across bases
            print(f"{arg}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
