"""Pull the value-axis numbers out of the results files into figures/data.

    uv run python figures/extract_value_axis.py

``make_figures.py`` reads JSON and never has a number typed into it, so that a
re-measurement cannot leave a stale annotation behind on a plot. The experiment
scripts print tables rather than emitting JSON, so this parses those tables once
and writes the result where the figures can find it.

The tables are read with :mod:`goalmisgen.analysis.tables`, the one parser of
014's format. Parsing formatted text is fragile, which is why every extraction
here asserts on the number of rows it found. A changed table format fails loudly
instead of silently producing a figure with three points on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run with plain python3 as well as uv

from goalmisgen.analysis import tables  # noqa: E402

RESULTS = ROOT / "results"
OUT = Path(__file__).parent / "data"


def rows(found: list[tuple[float, ...]], expected: int, what: str) -> list[tuple[float, ...]]:
    if len(found) != expected:
        raise SystemExit(f"expected {expected} {what}, parsed {len(found)} — has the table format changed?")
    return found


def labelled(text: str, label: str) -> list[tuple[float, ...]]:
    return [(row.point,) for row in tables.rows(text) if row.label == label]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    full = (RESULTS / "value-axis-full.txt").read_text()
    held = (RESULTS / "value-axis-heldout.txt").read_text()

    # The arms themselves, and the base they were all fine-tuned from.
    trained = rows([(r.value, r.point) for r in tables.value_rows(full, ("fine-tuned",))], 7, "fine-tuned arms")
    base = rows(labelled(full, tables.BASE_LABEL), 1, "base measurement")[0][0]

    # Written from an axis fitted *without* the arm being predicted. This is the
    # claim; the in-sample version cannot separate an axis from a lookup of the
    # arms it was built from.
    heldout = rows([loo[:3] for loo in tables.loo_rows(held)], 6, "held-out writes")

    # Values outside the fitted grid. These come from the full-data axis, since
    # there is no arm at 0.20 or 1.10 to hold out.
    beyond = rows([(r.value, r.point) for r in tables.value_rows(full, ("written (unseen)",))], 2, "extrapolations")

    # Directions of the same length, chosen at random. The control that makes
    # the rest mean anything.
    random = rows(
        [(float(m.group("magnitude")), r.point) for r in tables.rows(full) if (m := tables.RANDOM_LABEL.match(r.label))],
        2,
        "random controls",
    )

    # The axis written far outside the fitted grid, in both directions. Reach is
    # carried through because it is the whole caveat: past a certain write the
    # agent stops solving the maze, and a threshold measured on a policy that
    # fails two thirds of its episodes is not a preference.
    def written(path: Path) -> list[tuple[float, ...]]:
        found = tables.value_rows(path.read_text(), ("written (unseen)", "written (grid)"))
        return [(r.value, r.point, r.low, r.high, r.reach) for r in found]

    ood_rows = rows(written(RESULTS / "value-axis-ood.txt"), 17, "out-of-grid writes")
    ood0_rows = rows(written(RESULTS / "value-axis-ood-colour0.txt"), 18, "colour-0 out-of-grid writes")

    payload = {
        "base_value": 0.5,
        "step_penalty": 0.05,
        "other_objective": 1.0,
        # cfg.loss.gamma of the agent every arm was fine-tuned from. The optimal
        # threshold is what the agent is trained to maximise, which is discounted
        # return, so the undiscounted (value gap)/penalty overstates it. This
        # script serves the original 0.995-era seven-point grid only; the g1
        # (undiscounted) families flow through extract_wide_axis.py and the
        # fig-3 parser, which carry their own discount.
        "discount": 0.995,
        "base_exchange_rate": base,
        "trained": [{"value": v, "steps": s} for v, s in trained],
        "written_heldout": [{"value": v, "trained": t, "written": w} for v, t, w in heldout],
        "written_beyond_grid": [{"value": v, "steps": s} for v, s in beyond],
        "random_controls": [{"magnitude": m, "steps": s} for m, s in random],
        # Both sweeps, keyed by the value written for the objective each moves.
        # "held" is what the *other* objective is worth throughout, which is what
        # turns a written value into a gap — and, because discounting depends on
        # the absolute values rather than only their difference, is also what
        # gives each sweep its own optimal curve.
        "written_ood_colour0": [
            {"value": v, "steps": st, "low": lo, "high": hi, "reached": r / 100} for v, st, lo, hi, r in ood0_rows
        ],
        "written_ood": [{"value": v, "steps": st, "low": lo, "high": hi, "reached": r / 100} for v, st, lo, hi, r in ood_rows],
    }
    (OUT / "value_axis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT / 'value_axis.json'}")
    print(
        f"  base {base} steps, {len(trained)} arms, {len(heldout)} held-out writes, "
        f"{len(beyond)} beyond the grid, {len(random)} controls"
    )


if __name__ == "__main__":
    main()
