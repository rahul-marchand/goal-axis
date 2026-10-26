"""The directions a subspace retargeting learner may move along, from one base's value-axis sweep.

    uv run python scripts/studies/retarget/make_subspace.py /workspace/data/offline/runs/bcnv11.s1 \
        --out /workspace/data/offline/subspaces/bcnv11.s1 [--table results/retarget/subspaces.bcnv11.s1.txt]

Each direction set is saved as a :class:`goalmisgen.retarget.Subspace`
(orthonormal rows over the flat parameter vector) for ``retarget_cost.py
--subspace``. All come from the arms of one sweep (default ``o1@1k``, the sweep
the published route-model axis and the where-axis study use, and the one whose
values the retargeting goals move: arm offset ``o`` on colour 1's value is the
expert at ``tau = 10 - 20 o``):

``axis``           the Experiment 2 value axis as published: the slope of
                   ``diff = drift + offset * axis`` over every non-null arm
                   (``analysis.weights.fit_axis_and_drift``, as ``027`` fits it)
``axis-heldout/``  one file per goal rate ``tau``: the same fit without the arms
                   whose own rate lies within ``--holdout`` of ``tau``, so a
                   retargeting run cannot be reading its target's own arm back
``axis-drift``     the axis and the fit's intercept (the drift every arm shares),
                   a 2-D control for whether the axis needs drift to fit routes
``pca<k>``         the top ``k`` principal directions of the arms' diffs, centred
                   across arms (so the shared drift is not the first component)
``random<k>``      ``k`` iid Gaussian directions, the control for "any ``k``-D
                   learner is cheap"

Every set is orthonormalised, so a coefficient is a distance in parameter
space. For ``axis`` a coefficient ``c`` is an offset of ``c / |axis|`` on
colour 1's value, i.e. ``tau`` moves by ``-20 c / |axis|`` steps; the table
reports ``|axis|`` so a fitted coefficient can be read as a rate.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from goalmisgen.analysis.weights import cosine, fit_axis_and_drift
from goalmisgen.offline.axis import expected_indifference, load_base, load_sweep
from goalmisgen.provenance import header
from goalmisgen.retarget import Subspace

TAUS = [0.5, 2.5, 6.5, 8.5, 10, 12.5, 14.5, 18.5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--sweep", default="o1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--taus", type=float, nargs="+", default=TAUS, help="Goal rates to build held-out axes for.")
    parser.add_argument("--holdout", type=float, default=2.0, help="Drop arms whose own rate is within this of the goal's.")
    parser.add_argument("--pca", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--random", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--table", type=pathlib.Path, default=None)
    args = parser.parse_args()

    lines = [header(), ""]

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    base = load_base(args.run)
    objective = int(args.sweep[-1])
    offsets, diffs = load_sweep(base, args.sweep, args.steps, fit_window=np.inf, minimum=3)
    rates = np.array([expected_indifference(base.values, objective, float(o)) for o in offsets])
    common = {"base": str(args.run), "sweep": args.sweep, "steps": args.steps}
    say(f"base {args.run.name} @ step {base.step:,}; {base.flat.size:,} parameters")
    say(
        f"sweep {args.sweep}@{args.steps}: {len(offsets)} arms (null arm excluded), offsets {offsets.min():+.2f}..{offsets.max():+.2f}"
    )
    say(f"  arm rates tau = (v0 - v1) / 0.05: {', '.join(f'{t:g}' for t in rates)}")

    axis, drift = fit_axis_and_drift(offsets, diffs)
    norm = float(np.linalg.norm(axis))
    say(
        f"\naxis: |axis| {norm:.4f} per unit value, |drift| {np.linalg.norm(drift):.4f}, cos(axis, drift) {cosine(axis, drift):+.3f}"
    )
    say(
        f"  a unit coefficient along it is an offset of {1 / norm:.4f} on colour {objective}'s value, "
        f"tau {expected_indifference(base.values, objective, 1 / norm) - expected_indifference(base.values, objective, 0):+.3f} steps"
    )
    saved = {}

    def save(name: str, vectors, **meta) -> Subspace:
        subspace = Subspace.spanning(np.asarray(vectors), name=name, meta={**common, **meta})
        path = args.out / f"{name}.npz"
        subspace.save(path)
        saved[name] = subspace
        return subspace

    save(
        "axis",
        axis[None],
        kind="axis",
        offsets=offsets.tolist(),
        axis_norm=norm,
        tau_per_unit=float(
            expected_indifference(base.values, objective, 1 / norm) - expected_indifference(base.values, objective, 0)
        ),
    )
    save("axis-drift", np.stack([axis, drift]), kind="axis+drift", offsets=offsets.tolist(), axis_norm=norm)

    say(f"\nheld-out axes (arms with |tau_arm - tau| <= {args.holdout:g} left out)")
    say(f"  {'tau':>5} {'arms':>5} {'dropped':<38} {'|axis|':>7} {'cos(full)':>10}")
    for tau in args.taus:
        keep = np.abs(rates - tau) > args.holdout + 1e-9
        held_axis, _ = fit_axis_and_drift(offsets[keep], diffs[keep])
        held = Subspace.spanning(
            held_axis[None],
            name="axis-heldout",
            meta={
                **common,
                "kind": "axis held out",
                "tau": tau,
                "holdout": args.holdout,
                "offsets": offsets[keep].tolist(),
                "axis_norm": float(np.linalg.norm(held_axis)),
            },
        )
        held.save(args.out / "axis-heldout" / f"tau{tau:g}.npz")
        dropped = ", ".join(f"{t:g}" for t in rates[~keep]) or "-"
        say(f"  {tau:>5g} {keep.sum():>5} {dropped:<38} {np.linalg.norm(held_axis):>7.3f} {cosine(held_axis, axis):>+10.3f}")

    centred = diffs - diffs.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    share = singular**2 / (singular**2).sum()
    say(f"\nPCA of the centred arm diffs: variance share {', '.join(f'{s:.1%}' for s in share[: max(args.pca) + 1])}")
    for k in args.pca:
        sub = save(f"pca{k}", vt[:k], kind="pca", k=k, share=share[:k].tolist(), offsets=offsets.tolist())
        captured = float(np.linalg.norm(sub.directions.astype(np.float64) @ (axis / norm)))
        say(f"  pca{k}: share {share[:k].sum():.1%}; |projection of the unit axis onto it| {captured:.3f}")
    say(
        "  per component: cos with the axis "
        + ", ".join(f"PC{i + 1} {cosine(vt[i], axis):+.3f}" for i in range(max(args.pca)))
    )
    say(
        "  per component: correlation of the arms' scores with offset "
        + ", ".join(f"PC{i + 1} {np.corrcoef(centred @ vt[i], offsets)[0, 1]:+.3f}" for i in range(max(args.pca)))
    )

    rng = np.random.default_rng(args.seed)
    say(f"\nrandom (iid Gaussian, seed {args.seed})")
    for k in args.random:
        sub = save(f"random{k}", rng.standard_normal((k, base.flat.size)), kind="random", k=k, seed=args.seed)
        captured = float(np.linalg.norm(sub.directions.astype(np.float64) @ (axis / norm)))
        say(f"  random{k}: |projection of the unit axis onto it| {captured:.4f}")

    say(f"\n-> {args.out}: {', '.join(sorted(saved))}, axis-heldout/tau<tau>.npz")
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
