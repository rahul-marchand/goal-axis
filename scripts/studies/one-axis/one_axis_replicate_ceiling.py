"""G.2: replicate-based reliability of the fitted value axis, against the split-half one.

    PYTHONPATH=<snapshot> JAX_PLATFORMS=cpu python one_axis_replicate_ceiling.py \
        --rl novalue11g1.s1234=cp_149995520 resnet11nvg1.s1234=cp_120176640 vit11nvg1.s9012=cp_249999360

For each base, every objective whose ``o`` sweep has fine-tune-seed replicates
(``.r2``, ``.r3``, ...) at ``--arm-steps`` is fitted once per replicate set,
exactly as ``scripts/studies/one-axis/one_axis_null.py`` fits the original grid
(its ``fit``: offsets != 0, common component absorbed by an intercept, null arm
excluded). The replicate reliability of axis_k is the mean cosine between
independent fits of it. The split-half (Spearman-Brown) reliability of the
original fit, the raw cos(axis_0, axis_1) and the old ceiling are recomputed
with the same functions so the rows can be laid next to
``results/one-axis/one-axis-null.txt``.

The replicate ceiling sqrt(r0 * r1) needs both objectives replicated. Where only
one is, it is reported as not computable rather than filled in.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
from pathlib import Path

import numpy as np

from goalmisgen import provenance
from goalmisgen.analysis.weights import cosine
from goalmisgen.volume import iter_arms

HERE = Path(__file__).resolve()
NULL_SCRIPT = Path("scripts/studies/one-axis/one_axis_null.py")


def import_null():
    spec = importlib.util.spec_from_file_location("one_axis_null", NULL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replicates_of(arms: Path, objective: int, steps: int) -> list[int]:
    found = set()
    for k in range(2, 10):
        if any(True for _ in iter_arms(arms, f"o{objective}", steps=steps, replicate=k)):
            found.add(k)
    return sorted(found)


def load_sweep(null, base, root: Path, objective: int, arm_steps: int, replicate: int):
    from goalmisgen.axis import diff_from
    from goalmisgen.volume import discover_arms

    selected = discover_arms(
        root / "arms", objective, null.BASE_VALUES[objective], steps=arm_steps, at=-1, family="o", replicate=replicate
    )
    return {
        round(v - null.BASE_VALUES[objective], 10): diff_from(base, p, subtract_in="float64")
        for v, p in sorted(selected.items())
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", nargs="+", required=True)
    ap.add_argument("--arm-steps", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(provenance.header())
    print()

    null = import_null()
    from goalmisgen.axis import checkpoint_eval_config, load_checkpoint

    rows = []
    for spec in args.rl:
        agent, _, checkpoint = spec.partition("=")
        root = null.RUNS / agent
        config = checkpoint_eval_config(
            root / "local-files" / checkpoint, null.LEVELS, split="test", num_envs=8, seed=0,
            values=(null.BASE_VALUES[0], null.BASE_VALUES[1]),
        )
        base = load_checkpoint(root / "local-files" / checkpoint, config)

        fits = {}  # (objective, replicate) -> fit dict
        reps = {}
        for objective in (0, 1):
            reps[objective] = [1] + replicates_of(root / "arms", objective, args.arm_steps)
            for k in reps[objective]:
                diffs = load_sweep(null, base, root, objective, args.arm_steps, k)
                fits[(objective, k)] = null.fit(diffs, args.seed)

        print(f"=== {agent} ({checkpoint}, {base.flat.size:,} parameters) ===")
        for objective in (0, 1):
            for k in reps[objective]:
                f = fits[(objective, k)]
                print(f"  o{objective} r{k}: {len(f['offsets'])} arms {f['offsets'].min():+.2f}..{f['offsets'].max():+.2f}"
                      f"  |axis| {np.linalg.norm(f['axis']):.4g}  |drift| {np.linalg.norm(f['drift']):.4g}"
                      f"  split-half r (SB) {f['reliability']:+.3f}")
        rep_r = {}
        for objective in (0, 1):
            ks = reps[objective]
            if len(ks) < 2:
                rep_r[objective] = None
                print(f"  o{objective}: no replicates at {args.arm_steps} steps; replicate reliability not computable")
                continue
            pair = {(a, b): cosine(fits[(objective, a)]["axis"], fits[(objective, b)]["axis"]) for a, b in itertools.combinations(ks, 2)}
            rep_r[objective] = float(np.mean(list(pair.values())))
            print(f"  o{objective} replicate cosines: " + "  ".join(f"r{a}~r{b} {c:+.3f}" for (a, b), c in pair.items())
                  + f"   mean {rep_r[objective]:+.3f}")
            dcos = {(a, b): cosine(fits[(objective, a)]["drift"], fits[(objective, b)]["drift"]) for a, b in itertools.combinations(ks, 2)}
            print(f"  o{objective} drift cosines:     " + "  ".join(f"r{a}~r{b} {c:+.3f}" for (a, b), c in dcos.items()))

        a0 = fits[(0, 1)]["axis"]
        observed = cosine(a0, fits[(1, 1)]["axis"])
        cross = {k: cosine(a0, fits[(1, k)]["axis"]) for k in reps[1]}
        r0_sb, r1_sb = fits[(0, 1)]["reliability"], fits[(1, 1)]["reliability"]
        old_ceiling = float(np.sqrt(r0_sb * r1_sb))
        print(f"  cos(axis_0, axis_1) original {observed:+.3f};  with each o1 replicate: "
              + "  ".join(f"r{k} {c:+.3f}" for k, c in cross.items()))
        print(f"  split-half ceiling sqrt(r0 r1) {old_ceiling:+.3f}   cos/ceiling {observed / old_ceiling:+.2f}")
        if rep_r[0] is not None and rep_r[1] is not None:
            ceiling = float(np.sqrt(rep_r[0] * rep_r[1]))
            print(f"  replicate ceiling sqrt(r0 r1) {ceiling:+.3f}   cos/ceiling {observed / ceiling:+.2f}")
        else:
            ceiling = None
            print("  replicate ceiling sqrt(r0 r1): NOT COMPUTABLE (only "
                  + ", ".join(f"o{k}" for k in (0, 1) if rep_r[k] is not None) + " is replicated)")
        print()
        rows.append(dict(agent=agent, cos=observed, cross=cross, r0_sb=r0_sb, r1_sb=r1_sb, old=old_ceiling,
                         r0_rep=rep_r[0], r1_rep=rep_r[1], ceiling=ceiling))

    fmt = lambda x: "   n/a" if x is None else f"{x:+.3f}"
    print("=== summary ===")
    print(f"{'base':<22} {'cos(a0,a1)':>10} {'cos/a1 reps':>16} {'r0 SB':>7} {'r1 SB':>7} {'old ceil':>8} "
          f"{'cos/old':>7} {'r0 rep':>7} {'r1 rep':>7} {'r1rep/r1SB':>10} {'rep ceil':>8}")
    for r in rows:
        reps_cos = "/".join(f"{c:+.2f}" for k, c in r["cross"].items() if k != 1)
        ratio = None if r["r1_rep"] is None else r["r1_rep"] / r["r1_sb"]
        print(f"{r['agent']:<22} {r['cos']:>+10.3f} {reps_cos:>16} {r['r0_sb']:>+7.3f} {r['r1_sb']:>+7.3f} {r['old']:>+8.3f} "
              f"{r['cos'] / r['old']:>+7.2f} {fmt(r['r0_rep']):>7} {fmt(r['r1_rep']):>7} "
              f"{('   n/a' if ratio is None else f'{ratio:.2f}'):>10} {fmt(r['ceiling']):>8}")


if __name__ == "__main__":
    main()
