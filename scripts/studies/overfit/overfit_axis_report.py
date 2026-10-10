"""The overfitting study in three tables: what each base learned, and what its arms wrote.

    uv run python scripts/studies/overfit/overfit_axis_report.py /workspace/data/offline/results \
        --runs bcnv11.s1 bcnv11.s1.n1k bcov11.n1k.s1

Reads what ``scripts/pods/overfit_axis_pod.sh analysis`` leaves behind -- ``027``'s
JSON on the held-out levels, the same on the base's own training levels, and
``overfit_gap.py`` -- and puts the conditions side by side, because the only
thing any of these numbers means is in comparison with the others:

``bcnv11.s1``       the published campaign: a base trained on 900k
                    demonstrations, arms on 50k fresh ones
``bcnv11.s1.n<N>``  the same base, arms on N levels: the control for how much
                    data an arm needs
``bcov11.n<N>.s1``  a base trained on those N levels and nothing else, arms on
                    the same N: a memoriser, fine-tuned

A missing file is reported as a blank row rather than an error, so the report
can be run while the chain is still filling the directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Directory holding overfit.*.json.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run names, in the order to show them.")
    parser.add_argument("--sweep", default="o0")
    return parser.parse_args()


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    args = parse_args()

    axis = {run: load(args.results / f"overfit.value_axis.{run}.{args.sweep}.json") for run in args.runs}
    memory = {run: load(args.results / f"overfit.value_axis.{run}.{args.sweep}.mem.json") for run in args.runs}
    gaps = {run: load(args.results / f"overfit.gap.{run}.json") for run in args.runs}

    print("=== how much of each base's competence is memory ===")
    print(
        f"  {'run':<20}{'levels':>9}{'own acc':>10}{'held acc':>10}{'own opt.':>10}{'held opt.':>11}{'own indiff':>12}{'held indiff':>13}"
    )
    for run in args.runs:
        gap = gaps.get(run)
        if not gap:
            print(f"  {run:<20}{'-':>9}")
            continue
        own, held = gap["sets"].get("own"), gap["sets"].get("holdout")
        print(
            f"  {run:<20}{own['levels']:>9,}{own['token_accuracy']:>10.3f}{held['token_accuracy']:>10.3f}"
            f"{own['chose_optimal']:>10.3f}{held['chose_optimal']:>11.3f}{own['indifference']:>12.2f}{held['indifference']:>13.2f}"
        )
    print("  own = the levels the base trained on. A memoriser is near 1.0 there and poor beside it.")

    print("\n=== is the sweep an axis at all? (weights only; the same numbers either way) ===")
    print(f"  {'run':<20}{'arms':>6}{'cos same':>10}{'cos opp.':>10}{'implied slope':>15}{'reliability':>13}{'mean R2':>9}")
    for run in args.runs:
        data = axis.get(run)
        if not data:
            print(f"  {run:<20}{'-':>6}")
            continue
        r2 = np.mean([v["r2"] for v in data["loo"].values()])
        print(
            f"  {run:<20}{len(data['offsets']):>6}{data['cos_same_residual']:>+10.3f}{data['cos_opposite_residual']:>+10.3f}"
            f"{data['implied_slope']:>15.3f}{data['reliability']:>13.3f}{r2:>9.3f}"
        )
    print("  cos same > 0 and cos opposite < 0 with the drift removed = the arms moved along one line.")
    print("  implied slope 1 = an axis fitted without an arm reads that arm's offset back at scale.")

    print("\n=== does writing the axis move behaviour? ===")
    print(
        f"  {'run':<20}{'scored on':<11}{'reached':>9}{'base indiff':>13}{'arms':>8}{'written':>9}"
        f"{'+drift':>9}{'expert':>8}{'|w-arm|':>9}{'random':>9}"
    )
    for run in args.runs:
        for label, source in (("held-out", axis), ("memorised", memory)):
            data = source.get(run)
            if not data or not data.get("behaviour", {}).get("arms"):
                continue
            b = data["behaviour"]
            randoms = [v["indifference"] for k, v in b.get("controls", {}).items() if k.startswith("random")]
            noise = np.mean([abs(r - b["base"]["indifference"]) for r in randoms]) if randoms else float("nan")
            # The write's own slope, and the same write with the common component
            # added back. They come apart: a direction can read the offset back in
            # weight space and still leave behaviour where it found it unless the
            # drift goes with it.
            offsets = np.array([float(o) for o in b["arms"]])
            with_drift = np.array([b["arms"][o]["written_with_drift"]["indifference"] for o in b["arms"]])
            ok = np.isfinite(with_drift)
            slope_drift = float(np.polyfit(offsets[ok], with_drift[ok], 1)[0]) if ok.sum() > 2 else float("nan")
            print(
                f"  {run:<20}{label:<11}{b['base']['reached']:>9.3f}{b['base']['indifference']:>13.2f}"
                f"{b['slope_arms']:>8.1f}{b['slope_written']:>9.1f}{slope_drift:>9.1f}"
                f"{b['slope_expert']:>8.1f}{b['loo_error_mean_abs']:>9.2f}{noise:>9.2f}"
            )
    print("  arms/written/+drift/expert are steps of detour per unit of value. `reached` is the base's:")
    print("  an exchange rate fitted from few completed episodes is worth little whatever it says.")
    print("  random = mean |indifference - base| under directions of the same length as 0.45*axis and 0.2*axis;")
    print("  it is the scale at which 'the write did something' stops meaning anything.")


if __name__ == "__main__":
    main()
