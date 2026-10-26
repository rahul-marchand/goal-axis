"""Split one demonstration set between several goals: each level re-solved at one goal drawn at random.

    uv run python scripts/data/make_goal_split.py --demos /workspace/data/offline/demos/train.rho100 \
        --goals-like /workspace/data/offline/demos/arms/1.00-*.train.rho100 \
        --seed 0 --out /workspace/data/offline/demos/retarget/goals-random.s0

Training data for a goal-vector route model that sees as many distinct mazes as
``bcnv11`` (``scripts/pods/goal_vector_base.sh --random-tau``). ``bcgv11``'s 25
sets hold the same 50k mazes at 25 value pairs; here every level of ``--demos``
goes to one of those value pairs, uniformly at random (``split_by_goal``), and
is re-demonstrated by the expert at it (``retargeted``). The output is one set
per goal, named like the ``--goals-like`` set it copies its values from, so a
sorted listing gives the goals in the same order, and so the same goal ids, as
``bcgv11``. Load them with ``GoalMixture.load(..., levels="disjoint")``.

About a millisecond per level; the goals are re-solved in parallel.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.goals import exchange_rate, split_by_goal
from goalmisgen.offline.retarget import retargeted
from goalmisgen.parallel import usable_cpus, worker_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", type=Path, required=True, help="The set whose levels to split.")
    parser.add_argument(
        "--goals-like", type=Path, nargs="+", required=True, help="One set per goal; only its values are read."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seeds which goal each level gets.")
    parser.add_argument("--out", type=Path, required=True, help="Directory to write one set per goal into.")
    parser.add_argument("--workers", type=int, default=usable_cpus())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    like = sorted(args.goals_like)
    values = [tuple(float(v) for v in DemoSet.load(path).meta["values"]) for path in like]
    demos = DemoSet.load(args.demos, mmap=False)
    step_penalty = float(demos.meta["step_penalty"])
    split = split_by_goal(len(demos), len(like), args.seed)
    print(f"{args.demos}: {len(demos):,} levels -> {len(like)} goals, seed {args.seed}")

    start = time.time()
    tasks = [(demos.subset(rows), pair) for rows, pair in zip(split, values)]
    with worker_pool(min(args.workers, len(tasks))) as pool:
        sets = pool.starmap(retargeted, tasks)
    for path, pair, rows, goal_set in zip(like, values, split, sets):
        meta = {**goal_set.meta, "goal_split": {"from": str(args.demos), "seed": args.seed, "n_goals": len(like)}}
        dataclasses.replace(goal_set, meta=meta).save(args.out / path.name)
        print(f"  tau {exchange_rate(pair, step_penalty):5.2f}  values {pair}  {len(rows):>7,} levels -> {path.name}")
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
