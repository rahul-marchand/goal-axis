"""Does the rich/poor ratio of input-kernel signature lengths track the trained values?

    uv run python scripts/arm_norms.py /workspace/data/runs/novalue11g1.s1234

If value is carried as magnitude in the input code, an arm trained at a lower
colour-1 value should write a shorter colour-1 signature. Weights only: the
o1 arms at 400k steps, each at its last checkpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from goalmisgen.analysis.input_gain import input_kernel
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL


def measure(params) -> tuple[float, float, float]:
    """Lengths of the two colour planes' input signatures, and the cosine between them."""
    import jax

    k = input_kernel(params)
    flat = np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree_util.tree_leaves(params)])
    a, b = (flat[k.plane(c)] for c in (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb))
    return float(na), float(nb), cos


def latest(path: Path) -> Path | None:
    cps = sorted(path.glob("local-files/cp_*")) or sorted(path.glob("cp_*"))
    return cps[-1] if cps else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Agent directory holding BASE.json and arms/.")
    args = parser.parse_args()

    from goalmisgen.axis import load_base

    # Weights only: two live environments, built to load and never stepped.
    base = load_base(args.run, None, split="train", num_envs=2, seed=0)
    base_values = base.values
    na, nb, cos = measure(base.params)
    print(f"{'arm':16s} {'v1':>5s} {'|K0|':>8s} {'|K1|':>8s} {'ratio':>7s} {'cos':>7s} {'angle':>7s}")
    print(f"{'base':16s} {base_values[1]:5.2f} {na:8.3f} {nb:8.3f} {na / nb:7.3f} {cos:+7.3f} {np.degrees(np.arccos(cos)):7.2f}")
    for arm in sorted((args.run / "arms").glob("o1*@400k")):
        m = re.search(r"o1([+-])(\d+)", arm.name)
        offset = (1 if m.group(1) == "+" else -1) * int(m.group(2)) / 100
        v1 = base_values[1] + offset
        cp = latest(arm)
        if cp is None:
            print(f"{arm.name:16s} {v1:5.2f}   no checkpoint")
            continue
        na, nb, cos = measure(base.other_params(cp))
        print(f"{arm.name:16s} {v1:5.2f} {na:8.3f} {nb:8.3f} {na / nb:7.3f} {cos:+7.3f} {np.degrees(np.arccos(cos)):7.2f}")


if __name__ == "__main__":
    main()
