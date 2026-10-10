"""Per-gap choice curves for axis-written models: the s-curve panel's data.

    uv run python scripts/measure/written_scurves.py --agent /workspace/data/runs/novalue11.s1234 \
        --axis /workspace/data/figures-data/drc/basin/axis.novalue11.s1234.o1@400000.npz \
        --alphas -0.2 0 0.2 --out /workspace/data/figures-data/drc/scurves

The threshold tables say where the written threshold sits; this dumps the whole
psychometric curve around it. For each alpha (in units of the swept objective's
value) the base weights get ``theta + alpha * axis`` and are decoded over the
same episodes — one env, one seed, so the levels are paired across alphas and
the curves differ only by the write. Saves decode_h1_drc's five arrays per
alpha, plus a printed binned curve as a sanity check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.paths import level_dataset, run_dir
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", type=Path, default=run_dir("novalue11.s1234"))
    parser.add_argument("--axis", type=Path, required=True, help="Cached axis npz from scripts/measure/basin_map.py.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[-0.2, 0.0, 0.2])
    parser.add_argument("--levels", type=str, default=str(level_dataset(n_levels=1_000_000)))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--episodes", type=int, default=8192)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    from goalmisgen.analysis.behaviour import decision_arrays
    from goalmisgen.axis import Greedy, load_base
    from goalmisgen.volume import read_base_marker

    checkpoint = read_base_marker(args.agent).checkpoint
    axis = np.load(args.axis)["axis"].astype(np.float64)
    print(f"checkpoint {checkpoint}")
    print(f"axis       {args.axis.name}  |axis| {np.linalg.norm(axis):.1f} per unit value")

    base = load_base(args.agent, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed)
    base_flat = base.flat
    greedy = Greedy.of(base)

    args.out.mkdir(parents=True, exist_ok=True)
    for alpha in args.alphas:
        outcomes = greedy.outcomes(base.params_of(base_flat + alpha * axis), args.episodes, args.seed)
        arrays = decision_arrays(outcomes)
        reached = arrays["reached"]
        tag = f"{'+' if alpha >= 0 else '-'}{abs(alpha):.2f}"
        out = args.out / f"scurve.{args.agent.name}.a{tag}.npz"
        np.savez_compressed(out, alpha=alpha, **arrays)

        gap = arrays["d_rich"] - arrays["d_poor"]
        took = (arrays["reached_fid"] == arrays["colour_of_rich"]) & arrays["reached"]
        line = []
        for g in range(-4, 21, 4):
            m = (gap >= g) & (gap < g + 4) & arrays["reached"]
            if m.sum() >= 50:
                line.append(f"gap {g:>3}..{g+3}: {took[m].mean():.2f}")
        print(f"alpha {alpha:+.2f}  reached {reached.mean():.1%}  " + "  ".join(line))
    print("\nSCURVES_DONE")


if __name__ == "__main__":
    main()
