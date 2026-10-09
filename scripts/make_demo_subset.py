"""The first N demonstrations of a set, saved as a set of their own.

    uv run python scripts/make_demo_subset.py --demos <in> --n 1000 --out <out>
    uv run python scripts/make_demo_subset.py --demos <test set> --n 2048 \
        --exclude <another pool's train set> --out <a set held out from both>

For the overfitting study, where a base and every one of its arms must train on
*the same levels* and differ only in what the objectives are worth. Every
demonstration set at a given values tag is the same level pool in the same
order -- fixed values consume no randomness, so the layouts are identical
across tags, and the train/valid/test partition is decided by hashing the maze
rather than by partitioning the pool -- so "the first N" names the same N levels
in each of them. ``--check`` asserts exactly that against another set.

The slice is a prefix rather than a random draw so that a subset at N is a
subset of one at 10N: the overfitting ladder is then nested, and a difference
between its rungs is the amount of data and not also which data.
``level_index`` and ``source_fingerprint`` are carried through untouched, so
``shared_levels`` still catches a subset that overlaps an evaluation set.

**``--exclude`` exists because splits are global only at a fixed pool size.**
``split_by_layout`` decides a maze's split by hashing the maze, which makes the
partition the same for every pool generated at the same size -- but the
thresholds it hashes against are fractions of the pool, and the campaign's two
pools are different sizes. So ``1.00-0.50@150k``'s train split is *not* inside
``1.00-0.50@1M``'s: 2,732 of the 50k demonstrations the arms train on sit in
the 1M pool's test split, and 1,824 of the 2,048 the arms are scored on sit in
the 1M pool's training set. An evaluation set for this study has to be held out
from both pools at once, which is what excluding one from the other produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.offline.demos import DemoSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", type=Path, required=True, help="Demonstration set to slice.")
    parser.add_argument("--n", type=int, required=True, help="How many demonstrations to keep, from the front.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="Another demonstration set whose first N levels must be the same levels. Use it to "
        "confirm that a base's subset and an arm's subset really are the same mazes.",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="Demonstration sets whose levels to leave out, before taking the first N. They must "
        "come from the same values tag, since a level index only means anything relative to "
        "the pool that produced it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check is not None and args.exclude:
        raise SystemExit(
            "--check compares two sets position by position, which excluding levels from one of "
            "them makes meaningless; use one or the other"
        )
    demos = DemoSet.load(args.demos)

    keep = np.ones(len(demos), dtype=bool)
    for path in args.exclude:
        other = DemoSet.load(path)
        if other.meta.get("source_fingerprint") != demos.meta.get("source_fingerprint"):
            raise SystemExit(
                f"{path} was generated from a different sampler configuration than {args.demos}, so its "
                "level indices name different levels and excluding them would exclude nothing"
            )
        dropped = np.isin(np.asarray(demos.level_index), np.asarray(other.level_index))
        print(f"excluding {int((dropped & keep).sum()):,} levels that are also in {path}")
        keep &= ~dropped
    available = np.flatnonzero(keep)
    if args.n > len(available):
        raise SystemExit(
            f"{args.demos} leaves {len(available):,} demonstrations after exclusions, " f"fewer than the {args.n:,} asked for"
        )
    indices = available[: args.n]

    if args.check is not None:
        other = DemoSet.load(args.check)
        mine = np.asarray(demos.level_index[indices])
        theirs = np.asarray(other.level_index[indices])
        if not np.array_equal(mine, theirs):
            raise SystemExit(
                f"the first {args.n:,} levels of {args.demos} and {args.check} are different levels "
                f"({int((mine != theirs).sum()):,} differ); a subset of one is not a subset of the other"
            )
        print(f"checked: same {args.n:,} levels as {args.check}")

    subset = demos.subset(indices)
    subset.save(args.out)
    values = np.asarray(subset.values)
    print(
        f"{args.demos} [{len(demos):,}] -> {args.out} [{len(subset):,}], "
        f"values {tuple(values[0].round(3).tolist())}, rho {subset.rho}"
    )
    print(json.dumps({k: v for k, v in subset.meta.items() if k in ("source", "split", "values", "n")}, indent=2))


if __name__ == "__main__":
    main()
