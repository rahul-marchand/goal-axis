"""One table over every cheapest-edit run: how far each edit moved the threshold, how cleanly, and whether it is v.

    uv run python scripts/cheapest_edit_report.py figures/data/cheapest_edit/*.json [--out results/cheapest-edit-report.txt]

Per run (one row per seed, plus the fine-tuned axis written through the same
rollout as a reference row per base):

``gain``      held-out threshold shift per trained step, a least-squares slope
              through the trained doses; 1.00 is a perfect edit
``fit``       RMS error, in steps, between achieved and target threshold at the
              trained doses
``extrap``    the same over held-out doses where the agent stays competent
``floor``     the lowest threshold reached with reach >= 95% and on-route rate
              within 3 points of the base -- below zero is the regime no
              fine-tuned axis reached
``ceil``      the highest, same competence rule
``|d|``       edit length per step of threshold, and as a multiple of ``|v|``
``cos v``     weight-space cosine with the axis; ``ceil`` beside it is
              ``sqrt(r_u r_v)``, with ``r_u`` the seed-to-seed cosine
``J cos``     function-space cosine with the axis over all route steps; ``J fork`` at forks only
``in conv``   cosine inside the input layer
``split``     contingent control: half the above-minus-below threshold difference per trained step
              (1.00 = the contingent target met; a uniform threshold edit reads 0)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def competent(row: dict, base: dict) -> bool:
    return row["reached"] >= 0.95 and row["on_route"] >= base["on_route"] - 0.03


def summarise(rows: list[dict], base: dict, trained: set[float]) -> dict:
    tau0 = base["tau"]
    train = [r for r in rows if r["shift"] in trained]
    held = [r for r in rows if r["shift"] not in trained and competent(r, base)]
    x = np.array([r["shift"] for r in train])
    y = np.array([r["tau"] - tau0 for r in train])
    gain = float(x @ y / (x @ x)) if len(x) else float("nan")
    fit = float(np.sqrt(np.mean((y - x) ** 2))) if len(x) else float("nan")
    extrap = float(np.sqrt(np.mean([(r["tau"] - tau0 - r["shift"]) ** 2 for r in held]))) if held else float("nan")
    ok = [r for r in rows if competent(r, base)] + [base]
    # Contingent control: the threshold should move up by the shift where the richer goal is above and down where
    # it is below, so half the difference between the two readouts should track the shift.
    split = np.array([(r["tau_rich_above"] - r["tau_rich_below"]) / 2 - (base["tau_rich_above"] - base["tau_rich_below"]) / 2 for r in train])
    split_gain = float(x @ split / (x @ x)) if len(x) else float("nan")
    return {
        "split": split_gain,
        "gain": gain,
        "fit": fit,
        "extrap": extrap,
        "floor": min(r["tau"] for r in ok),
        "ceil": max(r["tau"] for r in ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lines = []
    header = (
        f"{'run':<44}{'seed':>5}{'gain':>7}{'fit':>6}{'extrap':>7}{'floor':>7}{'ceil':>7}{'|d|':>8}{'x|v|':>6}"
        f"{'cos v':>7}{'ceil':>6}{'J cos':>7}{'J fork':>7}{'in conv':>8}{'split':>7}"
    )
    by_base: dict[str, list] = {}
    for path in sorted(args.paths):
        data = json.loads(path.read_text())
        base_name = Path(data["run"]).name
        by_base.setdefault(base_name, []).append((path, data))

    for base_name, runs in by_base.items():
        lines.append(f"\n=== {base_name} ===\n{header}")
        reference_done = False
        for path, data in runs:
            base = data["base_eval"]
            base["shift"] = 0.0
            trained = set(data["shifts"])
            axis = data.get("axis", {})
            v_norm = axis.get("norm_per_step", float("nan"))
            if axis and not reference_done:
                s = summarise(axis["eval"], base, trained)
                lines.append(
                    f"{'v (fine-tuned axis, per step)':<44}{'':>5}{s['gain']:>7.2f}{s['fit']:>6.2f}{s['extrap']:>7.2f}"
                    f"{s['floor']:>7.2f}{s['ceil']:>7.2f}{v_norm:>8.3f}{1.0:>6.2f}"
                )
                reference_done = True
            seed_cos = data.get("seed_cosines") or []
            r_u = float(np.mean(seed_cos)) if seed_cos else float("nan")
            ceiling = float(np.sqrt(max(r_u, 0) * axis["reliability"])) if axis and seed_cos else float("nan")
            modules = {m["module"]: m["cos"] for m in axis.get("modules", [])}
            input_cos = modules.get("input conv", modules.get("patch embed", float("nan")))
            label = path.stem
            for run in data["runs"]:
                s = summarise(run["eval"], base, trained)
                name = f"seed{run['seed']}"
                wcos = axis.get("weight_cos", {}).get(name, float("nan"))
                fcos = axis.get("function_cos", {}).get(name, {}).get("v", float("nan"))
                fork_cos = axis.get("function_cos_fork", {}).get(name, {}).get("v", float("nan"))
                lines.append(
                    f"{label[:43]:<44}{run['seed']:>5}{s['gain']:>7.2f}{s['fit']:>6.2f}{s['extrap']:>7.2f}{s['floor']:>7.2f}"
                    f"{s['ceil']:>7.2f}{run['norm']:>8.3f}{run['norm'] / v_norm:>6.2f}{wcos:>+7.2f}{ceiling:>6.2f}{fcos:>+7.2f}{fork_cos:>+7.2f}"
                    f"{input_cos:>+8.2f}{s['split']:>+7.2f}"
                )
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
