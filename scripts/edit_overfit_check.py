"""Does a searched edit move the threshold on the levels it was trained on more than on held-out ones?

    uv run python scripts/edit_overfit_check.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --deltas /workspace/data/cheapest_edit/TAG.npz --n-edit 4096

A dense edit has a million and a half coordinates and the search sees a few
thousand levels, so it can buy loss with level-specific margin changes that do
not transfer. The fine-tuned axis was shaped on millions of levels. If the
threshold shift on the edit levels is close to the target while the held-out
shift falls short, the gap is memorisation, not a missing mechanism.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.provenance import header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--n-edit", type=int, default=4096)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--shifts", type=float, nargs="+", default=[-4, -2, 2, 4])
    args = parser.parse_args()
    print(header())

    from goalmisgen.analysis.closed_loop import read_outcomes
    from goalmisgen.edits.backends import rl_backend

    backend = rl_backend(args.run, args.levels, n_edit=args.n_edit, n_eval=args.n_eval)
    flat = backend.base_flat
    sets = {"edit": backend.edit_targets, "held out": backend.eval_targets}
    base = {name: read_outcomes(t, *backend.evaluate(flat, t)).tau for name, t in sets.items()}
    print("\nbase threshold: " + ", ".join(f"{k} {v:.2f}" for k, v in base.items()))

    deltas = np.load(args.deltas)
    for name in deltas.files:
        direction = deltas[name].astype(np.float64)
        print(f"\n=== {name} ===\n  {'shift':>6}{'edit shift':>12}{'held-out shift':>16}")
        for shift in args.shifts:
            moved = {k: read_outcomes(t, *backend.evaluate(flat + shift * direction, t)).tau - base[k] for k, t in sets.items()}
            print(f"  {shift:>+6.1f}{moved['edit']:>+12.2f}{moved['held out']:>+16.2f}", flush=True)


if __name__ == "__main__":
    main()
