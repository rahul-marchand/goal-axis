"""Assemble E14.md from the per-base readout-fit JSON, so no agent has to.

Run after each base finishes; it rewrites the whole report from whatever JSON
exists, so a killed chain still leaves a readable file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from goalmisgen.paths import figures_data_dir, repo_root

RESULTS = repo_root() / "results" / "gain-vs-factorised"

BASES = ["novalue11g1.s1234", "novalue11g1.s9012", "novalue11g1.s3456e"]

PREAMBLE = """# E14: is the decision a projection of the two fields at the agent's cell?

E12 compared the two objectives' fields by their **length** at the agent's own
cell -- `ln|A_rich| - ln|A_poor|` -- and found it tracks the level's distance gap
and crosses zero near the measured threshold. A length discards which way a
field points, so this fits the **direction** instead:

    x = A_rich[agent cell] - A_poor[agent cell]
    s = w . x

`w` is fitted on a training split of levels to predict the model's own choice in
a closed-loop rollout; every number below is on held-out levels.

**Read the no-intercept `fitted` row.** Its decision boundary is
`w . A_rich = w . A_poor` -- the two fields project equally -- so the gap at
which it crosses zero is a prediction to be checked against the behavioural
threshold, not a fitted parameter. `fitted +bias` is an upper bound on what any
linear readout at this one cell could do, and its crossing is not a prediction.

**References in every table.** `norm` is E12's length readout, unfitted.
`random xN` is N random unit directions, the floor. `shuffled fit` is the same
fit on permuted labels: it must not produce a crossing near the threshold.

**Caveats.** Levels where a field does not reach the agent's cell are dropped
(the count is the `n` column), as are rollouts that timed out. The fields come
from erasure on teacher-forced positions, so the `neither` baseline is off
distribution exactly as in E12. One base per section; agreement across the three
is the check.
"""


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else figures_data_dir() / "gain-vs-factorised" / "e14"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "E14.md"
    tables = Path(sys.argv[3]) if len(sys.argv) > 3 else RESULTS

    parts = [PREAMBLE]
    for base in BASES:
        parts.append(f"\n## {base}\n")
        payload = root / f"{base}.json"
        table = tables / f"e14-{base}.txt"
        if not payload.exists():
            parts.append(
                "Not finished; see the .err file.\n" if not table.exists() else "```\n" + table.read_text()[-4000:] + "\n```\n"
            )
            continue
        data = json.loads(payload.read_text())
        parts.append(
            f"Levels {data['levels']}, train fraction {data['train_frac']}, "
            f"measured threshold **{data['tau']:.2f}** "
            f"[{data['tau_low']:.2f}, {data['tau_high']:.2f}], reached {data['reached']:.1%}.\n"
        )
        readouts = [r for r in data["rows"] if r["analysis"] == "readout"]
        controls = {(r["site"], r["readout"]): r for r in data["rows"] if r["analysis"] == "control"}
        parts.append("\n| site | readout | AUC | corr with gap | crossing | n |")
        parts.append("|---|---|---|---|---|---|")
        for row in readouts:
            parts.append(
                f"| {row['site']} | {row['readout']} | {row['auc']:.3f} | {row['corr']:+.3f} "
                f"| {row['crossing']:+.2f} | {row['n']} |"
            )
        parts.append("\n### Controls\n")
        parts.append(
            "| site | random: mean AUC | random: best AUC | random: median crossing | " "shuffled AUC | shuffled crossing |"
        )
        parts.append("|---|---|---|---|---|---|")
        sites = sorted(
            {s for s, _ in controls},
            key=lambda s: [r["site"] for r in readouts].index(s) if s in [r["site"] for r in readouts] else 99,
        )
        for site in sites:
            rand = controls.get((site, "random"), {})
            shuffled = controls.get((site, "shuffled"), {})
            parts.append(
                f"| {site} | {rand.get('auc_mean', float('nan')):.3f} | "
                f"{rand.get('auc_best', float('nan')):.3f} | "
                f"{rand.get('crossing_median', float('nan')):+.2f} | "
                f"{shuffled.get('auc', float('nan')):.3f} | "
                f"{shuffled.get('crossing', float('nan')):+.2f} |"
            )
        direction = [r for r in data["rows"] if r["analysis"] == "direction"]
        if direction:
            parts.append("\n### What the direction is\n")
            parts.append("| site | cos with the axis's own change at that cell | cos with the mean rich field |")
            parts.append("|---|---|---|")
            for row in direction:
                parts.append(f"| {row['site']} | {row['cos_axis_change']:+.3f} | {row['cos_mean_rich']:+.3f} |")
        transfer = [r for r in data["rows"] if r["analysis"] == "transfer"]
        if transfer:
            parts.append("\n### Transfer: fitted at one site (row), used at another (column), held-out AUC\n")
            names = transfer[0]["used_at"]
            parts.append("| fitted at | " + " | ".join(names) + " |")
            parts.append("|---" * (len(names) + 1) + "|")
            for row in transfer:
                parts.append(f"| {row['fitted_at']} | " + " | ".join(f"{v:.3f}" for v in row["auc"]) + " |")
        parts.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
