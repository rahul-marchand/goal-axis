"""Experiment 4 of gain-versus-factorised, one table: which module families can move the threshold, at what cost.

    uv run python scripts/studies/gain-vs-factorised/e4_summary.py figures/data/gain_vs_factorised/e4 \
        --bases novalue11g1.s1234 novalue11g1.s9012 novalue11g1.s3456e > results/gain-vs-factorised/e4-summary.txt

Reads what ``e4_chain.sh`` wrote: ``<base>.<family>.json`` from ``cheapest_edit.py`` and
``<base>.collateral.json`` from ``edit_collateral.py``. Per family and base:

* **reach**: least-squares slope of held-out threshold on the written dose (steps
  per step; 1 = the edit does what it was trained to, 0 = it moves nothing),
  mean over search seeds, beside the fine-tuned axis written the same way;
* **|u| / |v|** per step of threshold; the search is Adam with a small L2 and no
  norm ball, so this is the size of what it found, not a minimum;
* **cos(Ju, Jv)**, the function-space cosine with the fitted axis;
* collateral: navigation with only one objective left and choice accuracy away
  from the threshold, at the largest dose, against the base.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = ("readout", "late", "input", "all")


def reach(evals) -> float:
    shifts = np.array([e["shift"] for e in evals])
    taus = np.array([e["tau"] for e in evals])
    return float(np.polyfit(shifts, taus, 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bases", nargs="+", required=True)
    args = parser.parse_args()

    for base in args.bases:
        print(f"=== {base} ===")
        print(
            f"  {'family':<9}{'reach':>7}{'|u|/|v|':>9}{'cos Ju,Jv':>11}{'cos u,v':>9}{'seed cos':>10}  module shares of |u|^2"
        )
        axis_reach = None
        for family in FAMILIES:
            path = args.directory / f"{base}.{family}.json"
            if not path.exists():
                print(f"  {family:<9}  (missing)")
                continue
            d = json.loads(path.read_text())
            axis = d["axis"]
            if axis_reach is None:
                axis_reach = reach(axis["eval"])
                print(
                    f"  {'axis v':<9}{axis_reach:>7.2f}{1.0:>9.2f}{1.0:>11.2f}{1.0:>9.2f}{axis['reliability']:>10.2f}  (reliability)"
                )
            runs = d["runs"]
            r = np.mean([reach(run["eval"]) for run in runs])
            norm = np.mean([run["norm"] for run in runs]) / axis["norm_per_step"]
            fcos = axis["function_cos"]["mean"]["v"]
            wcos = axis["weight_cos"]["mean"]
            seed_cos = np.mean(d["seed_cosines"]) if d["seed_cosines"] else float("nan")
            shares = ", ".join(f"{m['module']} {m['u_share']:.0%}" for m in axis["modules"] if m["u_share"] >= 0.02)
            print(f"  {family:<9}{r:>7.2f}{norm:>9.2f}{fcos:>+11.2f}{wcos:>+9.2f}{seed_cos:>+10.2f}  {shares}")

        path = args.directory / f"{base}.collateral.json"
        if not path.exists():
            print("  collateral: (missing)\n")
            continue
        c = json.loads(path.read_text())

        def row(label, x, target=None):
            alone = lambda k, m: x[f"alone_{k}"][m]  # noqa: E731
            t = f"{target:>7.2f}" if target is not None else "      -"
            print(
                f"  {label:<18}{t}{x['tau']:>7.2f}{x['away_own']['accuracy']:>8.1%}"
                f"{alone('rich', 'success'):>7.1%}{alone('rich', 'shortest'):>7.1%}{alone('poor', 'success'):>7.1%}{alone('poor', 'shortest'):>7.1%}"
            )

        print(f"\n  {'collateral':<18}{'target':>7}{'tau':>7}{'away':>8}{'R ok':>7}{'R opt':>7}{'P ok':>7}{'P opt':>7}")
        row("base", c["base"])
        for name, entry in c["edits"].items():
            for dose in entry["doses"]:
                if abs(dose["shift"]) == max(abs(s) for s in c["shifts"]):
                    row(f"{name} {dose['shift']:+.0f}", dose, dose["target"])
        print(
            f"\n  {'arm':<18}{'offset':>7}{'tau':>7}{'away':>8}{'R ok':>7}{'R opt':>7}{'P ok':>7}{'P opt':>7}   cos(diff,v)  |frozen diff|"
        )
        for name, arm in sorted(c["arms"].items(), key=lambda kv: (kv[1]["offset"], kv[0])):
            alone = lambda k, m: arm[f"alone_{k}"][m]  # noqa: E731
            print(
                f"  {name:<18}{arm['offset']:>+7.2f}{arm['tau']:>7.2f}{arm['away_own']['accuracy']:>8.1%}"
                f"{alone('rich', 'success'):>7.1%}{alone('rich', 'shortest'):>7.1%}{alone('poor', 'success'):>7.1%}{alone('poor', 'shortest'):>7.1%}"
                f"{arm['weight_cos_v']:>+12.2f}{arm['frozen_norm']:>13.2g}"
            )
        for family, fit in c.get("fitted", {}).items():
            if family == "frozen_vs_free":
                print(f"  cos(frozen axis, free axis) {fit['cos']:+.2f}, on movable coordinates {fit['cos_movable']:+.2f}")
            else:
                print(
                    f"  fitted {family} axis: slope {fit['slope_steps_per_unit']:+.2f} steps/unit, cos with v {fit['weight_cos_v']:+.2f}"
                )
        print(
            "  function space vs v per step: "
            + ", ".join(
                f"{n} {v:+.2f} (x{c['function_norm_ratio_v'][n]:.2f})" for n, v in c["function_cos_v"].items() if n != "v"
            )
        )
        print()


if __name__ == "__main__":
    main()
