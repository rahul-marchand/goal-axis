"""Tables for the far-dose decodes: does the written axis follow its matched gain past the window?

    uv run python scripts/axis_far_dose_report.py DATA_DIR [DATA_DIR ...] \
        --txt results/gain-vs-factorised/axis-far-dose.txt --json figures/data/axis_far_dose.json

Reads ``scripts/axis_far_dose.py`` output (one directory per base). Per base:

* **threshold** per full decode -- the pooled crossing (a two-parameter logistic in
  gap over every ``d_near``, no shared surface) with a paired percentile bootstrap
  over levels, plus reached and on-route rates;
* **log law** -- ``tau = a - b ln g`` fitted on colour-1 brightness decodes that
  still reach >= ``--competent`` of objectives, and ``k`` in ``g = 1 + k dv``
  from the colour-1 gains matched to the axis inside the fit window; the axis's
  predicted threshold ``a - b ln(1 + k dv)`` against what it did, and the first
  offset where the prediction leaves the interval by more than ``--tolerance``;
* **matched gains far out** -- the colour-1 gain each write actually matches,
  against ``1 + k dv``;
* **parts and random writes** at the part offsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.far_dose import first_departure, fit_linear_gain, fit_log_law  # noqa: E402
from goalmisgen.analysis.separability import Decisions, banded_crossings, bootstrap, paired, tabulate  # noqa: E402

POOLED = [(1, 80)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--competent", type=float, default=0.95)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--txt", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def load(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    full = manifest["episodes"]
    rows = [r for r in manifest["conditions"].values() if r["episodes"] == full]
    decisions = []
    for row in rows:
        z = np.load(directory / f"{row['name']}.npz")
        dec = Decisions.from_arrays(z["d_rich"], z["d_poor"], z["reached"], z["reached_fid"], z["colour_of_rich"])
        usable = dec.usable & ~z["value_tie"]
        decisions.append(Decisions(dec.d_rich, dec.d_poor, dec.took_rich & usable, usable))
    return manifest, rows, decisions


def crossings(table, n: int) -> np.ndarray:
    return np.array([banded_crossings(table, i, POOLED, min_n=50.0)[0] for i in range(n)])


def report_base(directory: Path, args, say) -> dict:
    manifest, rows, decisions = load(directory)
    n = len(rows)
    point = crossings(tabulate(decisions), n)
    draws = bootstrap(decisions, lambda t: crossings(t, n), resamples=args.resamples, paired=paired(decisions))
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    for i, row in enumerate(rows):
        row.update(cross=float(point[i]), low=float(low[i]), high=float(high[i]))
    by = {r["name"]: r for r in rows}
    base = by["base"]
    window = manifest["fit_window"]

    say("=" * 100)
    say(f"{Path(manifest['run']).name}   {manifest['checkpoint']}")
    say(f"levels {manifest['levels']} [{manifest['split']}], {manifest['episodes']:,} episodes per condition, paired: {paired(decisions)}")
    say(f"|axis| {manifest['axis_norm']:.3g}; input-conv gain per unit offset colour 0 {manifest['betas']['0']:+.3f}, "
        f"colour 1 {manifest['betas']['1']:+.3f}; fit window |dv| <= {window}")

    def line(label, r, extra=""):
        say(f"  {label:<26}{r['cross']:>+7.2f} [{r['low']:+6.2f},{r['high']:+6.2f}]{r['reached']:>8.1%}{r['on_route']:>8.1%}  {extra}")

    header = f"  {'':<26}{'tau':>7}  {'95% interval':^15}{'reached':>8}{'route':>8}"

    # --- log law -------------------------------------------------------------------------------
    bright = [r for r in rows if r["kind"] == "base" or r["kind"] == "bright" or (r["kind"] == "matched" and r["family"] == "c1")]
    gain_of = lambda r: 1.0 if r["kind"] == "base" else float(r["gains"]["1"])  # noqa: E731
    competent = [r for r in bright if r["reached"] >= args.competent and np.isfinite(r["cross"])]
    law = fit_log_law([gain_of(r) for r in competent], [r["cross"] for r in competent])
    inside = [r for r in rows if r["kind"] == "matched" and r["family"] == "c1" and abs(r["offset"]) <= window + 1e-9]
    k = fit_linear_gain([r["offset"] for r in inside], [np.exp(r["log_gain"]) for r in inside])
    say(f"\n--- log law: tau = {law.a:.2f} - {law.b:.2f} ln g on {len(competent)} colour-1 brightness decodes reaching >= {args.competent:.0%} "
        f"(rms {law.rms:.2f} steps); inside the window g = 1 + {k:.2f} dv ---")

    axis = sorted([base] + [r for r in rows if r["kind"] == "axis"], key=lambda r: r.get("offset", 0.0))
    dvs = np.array([r.get("offset", 0.0) for r in axis])
    predicted = law.tau(1 + k * dvs)
    matched_c1 = {r["matches"]: r for r in rows if r["kind"] == "matched" and r["family"] == "c1"}
    matched_two = {r["matches"]: r for r in rows if r["kind"] == "matched" and r["family"] == "two"}
    say(header + f"{'log-law':>9}{'obs-pred':>9}{'g match':>9}{'1+k dv':>8}{'two-plane s':>13}")
    for r, pred, dv in zip(axis, predicted, dvs):
        m1, m2 = matched_c1.get(r["name"]), matched_two.get(r["name"])
        g = f"{np.exp(m1['log_gain']):>9.2f}" if m1 else f"{'-':>9}"
        s = f"{m2['log_gain']:>13.2f}" if m2 else f"{'-':>13}"
        line(f"axis {dv:+.2f}", r, f"{pred:>+7.2f}{r['cross'] - pred:>+9.2f}{g}{1 + k * dv:>8.2f}{s}")
    departure = first_departure(dvs, [r["cross"] for r in axis], [r["low"] for r in axis], [r["high"] for r in axis], predicted, args.tolerance)
    reach_drop = next((float(dv) for r, dv in zip(axis, dvs) if r["reached"] < args.competent), None)
    say(f"  first offset leaving the log law by > {args.tolerance} steps: {departure}; first offset reaching < {args.competent:.0%}: {reach_drop}")

    say("\n--- colour-1 brightness ---")
    say(header + f"{'log-law':>9}")
    for r in sorted([r for r in rows if r["kind"] == "bright"], key=gain_of):
        line(f"colour 1 x{gain_of(r):g}", r, f"{law.tau(gain_of(r)):>+7.2f}")

    say("\n--- brightness matched to each axis write (same threshold) ---")
    say(header + "  gains")
    for r in sorted([r for r in rows if r["kind"] == "matched"], key=lambda r: (r["family"], r["offset"])):
        gains = ", ".join(f"c{c} x{g:.2f}" for c, g in sorted(r["gains"].items()))
        line(f"{r['family']} = axis {r['offset']:+.2f}", r, f"target {r['target_tau']:+.2f}   {gains}")

    say("\n--- parts of the axis ---")
    say(header)
    for part in ("gain in weights", "gain as brightness", "input conv", "input conv rescaled", "axis minus gain"):
        for r in sorted([r for r in rows if r["kind"] == "part" and r["part"] == part], key=lambda r: r["offset"]):
            line(f"{part} {r['offset']:+.2f}", r)
        say("")

    say("--- random writes (|dtau| from base; isotropic at |axis| dv, per-module at the axis's module norms) ---")
    say(f"  {'':<26}" + "".join(f"{'dv ' + format(dv, '+.1f'):>22}" for dv in sorted({r['offset'] for r in rows if r['kind'] == 'random'})))
    random_rows = [r for r in rows if r["kind"] == "random"]
    for family in ("isotropic", "per-module"):
        for draw in sorted({r["draw"] for r in random_rows if r["family"] == family}):
            cells = sorted([r for r in random_rows if r["family"] == family and r["draw"] == draw], key=lambda r: r["offset"])
            say(f"  {family + ' ' + str(draw):<26}" + "".join(
                f"{abs(r['cross'] - base['cross']):>8.2f}{r['reached']:>7.1%}{r['on_route']:>7.1%}" for r in cells))
    say("  (each cell: |dtau|, reached, route)")
    return {"run": manifest["run"], "law": {"a": law.a, "b": law.b, "rms": law.rms}, "k": k, "departure": departure,
            "reach_drop": reach_drop, "rows": rows}


def main() -> None:
    args = parse_args()
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    say("Past the fitted window, does the written o1 axis keep acting like the colour gain it matches?")
    say("tau is the pooled crossing (logistic in gap over every d_near), paired bootstrap over levels.\n")
    out = [report_base(d, args, say) for d in args.data]
    if args.txt:
        args.txt.parent.mkdir(parents=True, exist_ok=True)
        args.txt.write_text("\n".join(lines) + "\n")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
