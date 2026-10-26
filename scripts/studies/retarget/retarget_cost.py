"""Retargeting cost of a route model: the block-prequential code of one arm.

    uv run python scripts/studies/retarget/retarget_cost.py <run dir | random:<run dir>> \
        --pool /workspace/data/offline/demos/retarget/pool.rho100 --tau 6.5 \
        --out /workspace/data/offline/results/retarget/<agent>/<arm>.json

How many bits does it take to move this base onto the expert at exchange rate
``--tau``? The pool is split once for the whole study (``goalmisgen.retarget.splits``)
into the training stream D, a validation set V (early stopping) and a test set T
(the learning curve), the same for every agent and arm: levels tied at any of
``--study-taus`` are dropped first, so ``--tau`` must be one of them.
``--ordering`` permutes D only; ``--seed`` drives the learner's batches.
``random:<run dir>`` starts from a fresh initialisation of that run's
architecture, the "learn everything" anchor. ``--only`` trains one top-level
tensor and ``--subspace <file.npz>`` only the coefficients along a saved
``Subspace`` (``make_subspace.py``).

``--coding requential`` codes the goal by requential coding instead
(``goalmisgen.retarget.requential``): ``--calls`` REC calls of ``--call-batch``
routes each, the student being the learner above (``--lr``, ``--only``,
``--subspace``) and the teacher a fine-tune of the base on D's demonstrations
(``--teacher-lr``, ``--teacher-only``, ``--teacher-subspace``, ``--teacher-ema``).

The pool must be held out from every base compared on it.
"""

from __future__ import annotations

import argparse
import pathlib

from goalmisgen import provenance
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.retarget import PILOT_TAUS, ExchangeRates, RouteScorer, load_agent
from goalmisgen.retarget import (
    EMA,
    AllWeights,
    EarlyStopping,
    FineTunedTeacher,
    InSubspace,
    Learner,
    OneTensor,
    Optimiser,
    PrequentialCoding,
    RequentialCoding,
    RunSpec,
    diagnostics,
    measure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=str, help="Route-model run directory, or random:<run dir>.")
    parser.add_argument("--pool", type=pathlib.Path, required=True)
    parser.add_argument("--tau", type=float, required=True, help="Exchange rate of the goal, in steps.")
    parser.add_argument(
        "--study-taus",
        type=float,
        nargs="+",
        default=list(PILOT_TAUS),
        help="Every goal of the study: their tied levels leave the pool for every arm. Keep it fixed across a study.",
    )
    parser.add_argument("--n", type=int, default=16384, help="Length of the training stream D.")
    parser.add_argument("--n-valid", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--ordering", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=Optimiser.learning_rate)
    parser.add_argument("--patience", type=int, default=EarlyStopping.patience)
    parser.add_argument("--max-steps", type=int, default=EarlyStopping.max_steps)
    restrict = parser.add_mutually_exclusive_group()
    restrict.add_argument("--only", type=str, default=None, help="Train only this top-level parameter (e.g. goal_embedding).")
    restrict.add_argument(
        "--subspace",
        type=pathlib.Path,
        default=None,
        help="Train only the coefficients of this saved Subspace (make_subspace.py); give it its own --lr.",
    )
    parser.add_argument("--coding", choices=("prequential", "requential"), default="prequential")
    requential = parser.add_argument_group("requential coding")
    requential.add_argument("--calls", type=int, default=RequentialCoding.calls)
    requential.add_argument("--call-batch", type=int, default=RequentialCoding.batch, help="Routes per REC call.")
    requential.add_argument("--student-epochs", type=int, default=RequentialCoding.student_epochs)
    requential.add_argument("--teacher-lr", type=float, default=Optimiser.learning_rate)
    teacher = requential.add_mutually_exclusive_group()
    teacher.add_argument("--teacher-only", type=str, default=None, help="Fine-tune only this top-level tensor of the teacher.")
    teacher.add_argument("--teacher-subspace", type=pathlib.Path, default=None)
    requential.add_argument("--teacher-ema", action="store_true", help="Teach from a moving average of the teacher (EMA()).")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def restriction(only: str | None, subspace: pathlib.Path | None):
    """What a learner trains, from its command-line restriction."""
    if subspace is not None:
        return InSubspace.load(subspace)
    return AllWeights() if only is None else OneTensor(only)


def coding_of(args: argparse.Namespace):
    if args.coding == "prequential":
        return PrequentialCoding()
    teacher = FineTunedTeacher(
        optimiser=Optimiser(learning_rate=args.teacher_lr),
        trains=restriction(args.teacher_only, args.teacher_subspace),
        ema=EMA() if args.teacher_ema else None,
    )
    return RequentialCoding(teacher=teacher, calls=args.calls, batch=args.call_batch, student_epochs=args.student_epochs)


def main() -> None:
    args = parse_args()
    print(provenance.header())
    agent = load_agent(args.base)
    family = ExchangeRates()
    trains = restriction(args.only, args.subspace)
    if args.subspace is not None:
        print(f"learner restricted to subspace {trains.subspace.name!r} (k = {trains.subspace.k}) from {args.subspace}\n")
    spec = RunSpec(
        base=args.base,
        pool=str(args.pool),
        goal=args.tau,
        study_goals=tuple(sorted({*args.study_taus, family.null})),
        n=args.n,
        n_valid=args.n_valid,
        n_test=args.n_test,
        ordering=args.ordering,
        seed=args.seed,
        learner=Learner(
            optimiser=Optimiser(learning_rate=args.lr),
            stopping=EarlyStopping(patience=args.patience, max_steps=args.max_steps),
            trains=trains,
        ),
        coding=coding_of(args),
    )
    pool = DemoSet.load(args.pool, mmap=False, hide_values=agent.hide_values)
    print(f"base {args.base}")
    record = measure(
        spec,
        family,
        RouteScorer(agent.model),
        agent.params,
        pool,
        provenance={"commit": provenance.commit(), **agent.provenance},
    )
    result = record.result
    print(
        f"\n{result.kind}: {result.total_bits:,.1f} bits ({result.overhead_bits:,.1f} overhead)   "
        f"L(0) {result.test_bits[0]:.3f}   L(n) {result.test_bits[-1]:.3f}"
    )
    record.save(args.out)
    print(f"-> {args.out}")
    findings = diagnostics.run_checks(record)
    print(f"\nchecks: {diagnostics.worst(findings).value}\n{diagnostics.render(findings)}")


if __name__ == "__main__":
    main()
