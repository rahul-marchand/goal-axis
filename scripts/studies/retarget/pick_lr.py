"""Which learning rate to give the retargeting learner, chosen on a rate no arm uses.

    uv run python scripts/studies/retarget/pick_lr.py /workspace/data/offline/runs/bcnv11.s1 \
        --pool /workspace/data/offline/demos/retarget/pool.rho100 --tau 4.5 --lr 1e-5 3e-5 1e-4

For each learning rate and training-set size, fits once from the base with early
stopping and reports the best validation loss, split into the levels whose
choice the goal flips and those it leaves alone, and the step it was reached
at. The learner is part of the code, so its rate is fixed for a whole study;
choosing it on an exchange rate outside the arm grid keeps the choice from
having seen any arm's goal. ``--subspace`` picks the rate of a subspace learner
(``retarget_cost.py --subspace``) the same way.

The levels are its own shuffle of the pool (seed 1), V first, not a study split:
the recorded ``pick_lr.*`` tables were made that way.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.retarget import ExchangeRates, RouteScorer, load_agent
from goalmisgen.provenance import header
from goalmisgen.retarget import AllWeights, InSubspace, Learner, Optimiser, example_bits, make_fitter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=pathlib.Path)
    parser.add_argument("--pool", type=pathlib.Path, required=True)
    parser.add_argument("--tau", type=float, default=4.5)
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4])
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 256, 4096])
    parser.add_argument("--n-valid", type=int, default=1024)
    parser.add_argument("--subspace", type=pathlib.Path, default=None, help="A saved Subspace to restrict the learner to.")
    args = parser.parse_args()
    print(header())

    agent = load_agent(args.base)
    params0 = agent.params
    family = ExchangeRates()
    pool = DemoSet.load(args.pool, mmap=False, hide_values=agent.hide_values)
    family.check_pool(pool)
    keep = family.deterministic(pool, [family.null, args.tau])
    chosen = keep[np.random.default_rng(1).permutation(len(keep))[: args.n_valid + max(args.sizes)]]
    demos = family.demonstrations(pool, chosen, args.tau)
    flipped = family.changed(pool, chosen, demos)
    valid = np.arange(args.n_valid)
    train = np.arange(args.n_valid, len(chosen))

    scorer = RouteScorer(agent.model)
    start = example_bits(scorer, params0, demos, valid)
    print(
        f"tau {args.tau}: {flipped[valid].mean():.1%} of V flipped; base scores {start.mean():.3f} bits/route "
        f"({start[flipped[valid]].mean():.2f} flipped, {start[~flipped[valid]].mean():.3f} unflipped)\n"
    )
    trains = AllWeights() if args.subspace is None else InSubspace.load(args.subspace)
    if args.subspace is not None:
        print(f"learner restricted to subspace {trains.subspace.name!r} (k = {trains.subspace.k}) from {args.subspace}\n")
    print(f"{'lr':>8} {'t':>6} {'best step':>10} {'stopped':>8} {'V':>7} {'flipped':>8} {'unflipped':>10}")
    for lr in args.lr:
        fit = make_fitter(scorer, Learner(optimiser=Optimiser(learning_rate=lr), trains=trains))
        for t in args.sizes:
            fitted = fit(params0, demos, train[:t], valid, seed=0)
            scored = example_bits(scorer, fitted.params, demos, valid)
            print(
                f"{lr:>8.0e} {t:>6} {fitted.best_step:>10} {fitted.steps_run:>8} {scored.mean():>7.3f} "
                f"{scored[flipped[valid]].mean():>8.2f} {scored[~flipped[valid]].mean():>10.3f}"
                + ("" if fitted.recorded is None else "  c " + " ".join(f"{x:+.3f}" for x in fitted.recorded)),
                flush=True,
            )


if __name__ == "__main__":
    main()
