"""The goal-vector agent with its dial set as well as it can be: a reference no subtraction can bias.

    uv run python scripts/studies/retarget/oracle_dial.py /workspace/data/offline/runs/bcgv11.s1 \
        --pool /workspace/data/offline/demos/retarget/pool.rho100 --out <dir>

For each exchange rate, the candidate settings of the base goal's vector are
the 25 trained vectors and the 24 midpoints between neighbours (by tau). The
one with the lowest loss on V is the *oracle dial*, and its bits on the same
training stream D and test set T as ``retarget_cost.py`` codes (``arm_data``)
are what an agent pays if it only turns its dial. A retargeting run's cost above
that is then the cost of finding the dial, with no null arm and no
route-composition bias in it: the oracle encodes the very routes the run
encodes.

Choosing among 49 candidates on V is worth log2(49), about 6 bits, once.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.goals import with_base_goal_vector
from goalmisgen.offline.retarget import PILOT_TAUS, ExchangeRates, RouteScorer, load_agent
from goalmisgen.offline.train import load_run_config
from goalmisgen.provenance import header
from goalmisgen.retarget import arm_data, example_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--pool", type=pathlib.Path, required=True)
    parser.add_argument("--taus", type=float, nargs="+", default=list(PILOT_TAUS))
    parser.add_argument("--study-taus", type=float, nargs="+", default=list(PILOT_TAUS))
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--n-valid", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(header())

    agent = load_agent(args.run)
    model, params = agent.model, agent.params
    goal_taus = np.array([g["tau"] for g in load_run_config(args.run)["demos"]["goals"]])
    table = np.asarray(params["params"]["goal_embedding"])
    order = np.argsort(goal_taus)
    candidates = [(f"tau {goal_taus[k]:g}", table[k]) for k in order]
    candidates += [(f"mid {goal_taus[a]:g}..{goal_taus[b]:g}", (table[a] + table[b]) / 2) for a, b in zip(order, order[1:])]
    scorer = RouteScorer(model)
    family = ExchangeRates()
    study = sorted({*args.study_taus, family.null})

    pool = DemoSet.load(args.pool, mmap=False, hide_values=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for tau in args.taus:
        arm = arm_data(family, pool, tau, study, args.n, args.n_valid, args.n_test)
        demos, split = arm.data, arm.split
        settings = [with_base_goal_vector(params, model, v) for _, v in candidates]
        scores = [float(example_bits(scorer, p, demos, split.valid).mean()) for p in settings]
        best = int(np.argmin(scores))
        d_bits = example_bits(scorer, settings[best], demos, split.train)
        t_bits = float(example_bits(scorer, settings[best], demos, split.test).mean())
        record = {
            "tau": tau,
            "best": candidates[best][0],
            "valid_bits": scores[best],
            "test_bits": t_bits,
            "train_bits_total": float(d_bits.sum()),
            "train_bits_by_example": d_bits.tolist(),
            "candidates": {name: score for (name, _), score in zip(candidates, scores)},
        }
        (args.out / f"tau{tau:g}.json").write_text(json.dumps(record))
        print(
            f"tau {tau:>5g}: oracle {candidates[best][0]:<16} V {scores[best]:.3f}  T {t_bits:.3f}  D total {d_bits.sum():,.0f} bits",
            flush=True,
        )


if __name__ == "__main__":
    main()
