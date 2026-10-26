"""The best a one-dimensional subspace learner could do: the loss along a direction, and its best point.

    uv run python scripts/studies/retarget/oracle_subspace.py /workspace/data/offline/runs/bcnv11.s1 \
        --pool /workspace/data/offline/demos/retarget/pool.rho100 \
        --subspace /workspace/data/offline/subspaces/bcnv11.s1/axis.npz --out <dir>

The twin of ``oracle_dial.py`` for an agent whose dial, if it has one, is a
direction in weight space rather than a vector it was built with. For each
exchange rate the base is moved to ``theta0 + c u`` for every ``c`` on a grid,
and the ``c`` with the lowest loss on V is the *oracle setting*. Its bits on
the same D and T as ``retarget_cost.py`` codes (``arm_data``) are what the agent
pays if it only turns that one knob, as well as it can be turned. A subspace
learner's SDL above that is the cost of *finding* the setting by SGD; the
oracle's own excess over the null arm is what the direction cannot do at any
setting. The profile on V also shows the landscape the learner walks, so a
stalled fit can be told from a direction that runs out.

``--subspace`` may be a directory of per-rate files (``axis-heldout/``), as in
``retarget_chain.sh``. Choosing among G grid points on V is worth log2(G) bits,
once. Only one-dimensional subspaces: a grid in k dimensions is G^k fits.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.retarget import PILOT_TAUS, ExchangeRates, RouteScorer, load_agent
from goalmisgen.provenance import header
from goalmisgen.retarget import InSubspace, Subspace, arm_data, example_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--pool", type=pathlib.Path, required=True)
    parser.add_argument(
        "--subspace", type=pathlib.Path, required=True, help="A 1-D Subspace file, or a directory of tau<tau>.npz."
    )
    parser.add_argument("--taus", type=float, nargs="+", default=list(PILOT_TAUS))
    parser.add_argument("--study-taus", type=float, nargs="+", default=list(PILOT_TAUS))
    parser.add_argument("--grid", type=float, nargs=3, default=[-1.5, 4.0, 0.05], metavar=("LO", "HI", "STEP"))
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--n-valid", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(header())

    agent = load_agent(args.run)
    params0 = agent.params
    scorer = RouteScorer(agent.model)
    family = ExchangeRates()
    study = sorted({*args.study_taus, family.null})
    lo, hi, step = args.grid
    grid = np.round(np.arange(lo, hi + step / 2, step), 10)

    pool = DemoSet.load(args.pool, mmap=False, hide_values=agent.hide_values)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{'tau':>5} {'c*':>7} {'V at c*':>8} {'V at 0':>8} {'T at c*':>8} {'D total':>9}  profile minima")
    for tau in args.taus:
        path = args.subspace / f"tau{tau:g}.npz" if args.subspace.is_dir() else args.subspace
        subspace = Subspace.load(path)
        if subspace.k != 1:
            raise SystemExit(f"{path} is {subspace.k}-dimensional; the grid oracle is for one direction")
        arm = arm_data(family, pool, tau, study, args.n, args.n_valid, args.n_test)
        demos, split = arm.data, arm.split

        profile = np.array(
            [float(example_bits(scorer, subspace.point(params0, [c]), demos, split.valid).mean()) for c in grid]
        )
        best = int(np.argmin(profile))
        oracle = subspace.point(params0, [grid[best]])
        d_bits = example_bits(scorer, oracle, demos, split.train)
        t_bits = float(example_bits(scorer, oracle, demos, split.test).mean())
        interior = np.flatnonzero((profile[1:-1] < profile[:-2]) & (profile[1:-1] < profile[2:])) + 1
        record = {
            "tau": tau,
            "subspace": InSubspace(subspace, path=str(path)).describe(),
            "grid": grid.tolist(),
            "valid_profile": profile.tolist(),
            "best": float(grid[best]),
            "valid_bits": float(profile[best]),
            "test_bits": t_bits,
            "train_bits_total": float(d_bits.sum()),
            "train_bits_by_example": d_bits.tolist(),
            "local_minima": [float(grid[i]) for i in interior],
        }
        (args.out / f"tau{tau:g}.json").write_text(json.dumps(record))
        zero = profile[int(np.argmin(np.abs(grid)))]
        minima = ", ".join(f"{grid[i]:+.2f} ({profile[i]:.3f})" for i in interior)
        print(
            f"{tau:>5g} {grid[best]:>+7.2f} {profile[best]:>8.3f} {zero:>8.3f} {t_bits:>8.3f} {d_bits.sum():>9,.0f}  {minima}",
            flush=True,
        )


if __name__ == "__main__":
    main()
