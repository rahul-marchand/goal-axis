"""Pairwise comparison of saved edit directions: weight space, function space over all steps, and at forks only.

    uv run python scripts/edit_compare.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k \
        --direction v=/workspace/data/cheapest_edit/p.gain.novalue11g1.s1234.npz:v_step \
        --direction gain=/workspace/data/cheapest_edit/p.gain.novalue11g1.s1234.npz:seed0 ...

Every direction is compared with every other on the same held-out teacher-forced
batch, so edits from different searches (and the fine-tuned axis they all saved
as ``v_step``) sit in one matrix. Directions should be per step of threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.provenance import header  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--direction", action="append", required=True, help="NAME=PATH.npz:KEY")
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(header())

    from goalmisgen.analysis.layers import module_groups
    from goalmisgen.analysis.weights import cosine
    from goalmisgen.edits import compare
    from goalmisgen.edits.targets import FORK

    sys.argv = [sys.argv[0], str(args.run), "--levels", args.levels, "--n-edit", "16", "--n-eval", str(args.n)]
    from scripts.cheapest_edit import parse_args as edit_args
    from scripts.cheapest_edit import rl_backend

    backend = rl_backend(edit_args())
    directions = {}
    for spec in args.direction:
        name, rest = spec.split("=", 1)
        path, key = rest.rsplit(":", 1)
        directions[name] = np.load(path)[key].astype(np.float64)
    names = list(directions)
    batch = {k: np.asarray(v) for k, v in backend.eval_arrays.items()}

    tangents = compare.function_tangents(backend.logits_fn, backend.base_flat, batch, directions)
    forks = compare.function_tangents(backend.logits_fn, backend.base_flat, batch, directions, roles=(FORK,))

    def matrix(title, value):
        print(f"\n{title}\n  {'':<14}" + "".join(f"{n[:12]:>13}" for n in names))
        for a in names:
            print(f"  {a[:14]:<14}" + "".join(f"{value(a, b):>+13.3f}" for b in names))

    matrix("weight-space cosine", lambda a, b: cosine(directions[a], directions[b]))
    matrix("function-space cosine, all route steps", lambda a, b: cosine(tangents[a], tangents[b]))
    matrix("function-space cosine, forks only", lambda a, b: cosine(forks[a], forks[b]))
    print("\nnorms: " + ", ".join(f"{n} |d| {np.linalg.norm(directions[n]):.3f} |J d| {np.linalg.norm(tangents[n]):.2f} "
                                  f"|J d| fork {np.linalg.norm(forks[n]):.2f}" for n in names))

    groups, order = module_groups(backend.params)
    print(f"\nper-module share of |d|^2\n  {'module':<20}" + "".join(f"{n[:12]:>13}" for n in names))
    for module in order:
        print(f"  {module:<20}" + "".join(
            f"{100 * float(directions[n][groups[module]] @ directions[n][groups[module]]) / float(directions[n] @ directions[n]):>12.1f}%"
            for n in names))


if __name__ == "__main__":
    main()
