"""Is the axis's effect on choice a pure shift, and does it bend the contours the way brightness does?

    uv run python scripts/studies/gain-vs-factorised/separability_report.py DATA_DIR [DATA_DIR ...] \
        --txt results/gain-vs-factorised/separability.txt --json figures/data/separability.json

Reads the decodes written by ``scripts/studies/gain-vs-factorised/separability_decode.py`` (one directory
per base) and reports, per base:

* **field shape** -- the ratio model ``logit = c + psi(d_rich) - psi(d_poor)``
  on the base, free psi against linear psi (an exponential field), with how far
  the free psi strays in steps, and how much worse the ratio model is than an
  unconstrained surface h(d_rich, d_poor);
* **global separability** -- deviance of ``a(condition) + h`` against models
  with condition x d_near and condition x gap terms (cells are clustered by
  level, so the chi-square p-values are optimistic; the bootstrap is the
  honest interval);
* **per condition** -- threshold at the median d_near, contour tilt dtau/dd_near,
  width q75-q25, from the full model; and banded nonparametric crossings;
* **matched pairs** -- each axis write against the colour-1 and colour-0
  brightness at the same threshold: tilt and width differences, and the banded
  contour difference;
* **family trends** -- tilt and width regressed on threshold across the axis
  writes and across the brightness conditions; the difference in slope is the
  one-number answer to "do contours move with threshold the same way under both".

Intervals are percentile bootstraps over levels, paired across conditions when
the decodes are paired (checked).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from goalmisgen.analysis.separability import (
    Decisions,
    Surface,
    banded_crossings,
    bootstrap,
    field_shape,
    fit_model,
    paired,
    tabulate,
)

BANDS = [(1, 2), (3, 4), (5, 6), (7, 9), (10, 14)]
NEARS = np.arange(2, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--txt", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def load(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [r for r in manifest["conditions"].values() if r["kind"] in ("base", "axis", "matched", "below")]
    order = {"base": 0, "axis": 1, "matched": 2, "below": 3}
    rows.sort(key=lambda r: (order[r["kind"]], r["name"]))
    decisions = []
    for row in rows:
        z = np.load(directory / f"{row['name']}.npz")
        dec = Decisions.from_arrays(z["d_rich"], z["d_poor"], z["reached"], z["reached_fid"], z["colour_of_rich"])
        usable = dec.usable & ~z["value_tie"]
        decisions.append(Decisions(dec.d_rich, dec.d_poor, dec.took_rich & usable, usable))
    return manifest, rows, decisions


def summarise(table, surface, rows):
    """Everything bootstrapped, as one flat vector plus the index to read it."""
    fit = fit_model(table, "full", surface=surface)
    near_median = surface.near_centre
    out, keys = [], []
    tau, tilt, width = {}, {}, {}
    for i, row in enumerate(rows):
        contour = fit.contour(i, NEARS)
        ok = np.isfinite(contour)
        tilt[i] = float(np.polyfit(NEARS[ok], contour[ok], 1)[0]) if ok.sum() >= 3 else np.nan
        tau[i] = float(fit.contour(i, [near_median])[0])
        width[i] = fit.width(i, near_median)
        near_coef, gap_coef = fit.interaction(i)
        bands = banded_crossings(table, i, BANDS)
        for name, value in [
            ("tau", tau[i]),
            ("tilt", tilt[i]),
            ("width", width[i]),
            ("near_coef", near_coef),
            ("gap_coef", gap_coef),
        ]:
            keys.append((row["name"], name))
            out.append(value)
        for b, value in zip(BANDS, bands):
            keys.append((row["name"], f"band{b[0]}-{b[1]}"))
            out.append(value)
    by_name = {row["name"]: i for i, row in enumerate(rows)}
    for i, row in enumerate(rows):
        if row["kind"] != "matched":
            continue
        a = by_name[row["matches"]]
        for name, source in [("d_tilt", tilt), ("d_width", width), ("d_tau", tau)]:
            keys.append((row["name"], name))
            out.append(source[i] - source[a])
    families = {
        "axis": [i for i, r in enumerate(rows) if r["kind"] in ("base", "axis")],
        "bright": [i for i, r in enumerate(rows) if r["kind"] in ("base", "matched", "below")],
        "bright_in_range": [i for i, r in enumerate(rows) if r["kind"] in ("base", "matched")],
    }
    slopes = {}
    for family, members in families.items():
        for quantity, source in [("tilt", tilt), ("width", width)]:
            x = np.array([tau[i] for i in members])
            y = np.array([source[i] for i in members])
            ok = np.isfinite(x) & np.isfinite(y)
            slope = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else np.nan
            slopes[(family, quantity)] = slope
            keys.append((f"family:{family}", f"{quantity}_per_tau"))
            out.append(slope)
    for quantity in ("tilt", "width"):
        keys.append(("family:axis-bright_in_range", f"{quantity}_per_tau"))
        out.append(slopes[("axis", quantity)] - slopes[("bright_in_range", quantity)])
    return np.array(out, float), keys


def report_base(directory: Path, resamples: int, say) -> dict:
    manifest, rows, decisions = load(directory)
    base_name = Path(manifest["run"]).name
    is_paired = paired(decisions)
    table = tabulate(decisions)
    surface = Surface.for_table(table)
    say(f"\n{'=' * 100}\n{base_name}   checkpoint {manifest['checkpoint']}")
    say(
        f"levels {manifest['levels']} [{manifest['split']}], {manifest['episodes']:,} episodes per condition, "
        f"{len(rows)} conditions, paired across conditions: {is_paired}"
    )
    say(
        f"surface: gap knots {surface.gap_knots[0]:g}..{surface.gap_knots[-1]:g}, d_near knots "
        f"{surface.near_knots[0]:g}..{surface.near_knots[-1]:g}, median-ish d_near {surface.near_centre:.2f}"
    )

    # --- field shape --------------------------------------------------------
    shape = field_shape(table, condition=0)
    lr = shape.deviance_linear - shape.deviance_free
    say("\n--- base field shape, ratio model logit = c + psi(d_rich) - psi(d_poor) ---")
    say(
        f"  linear psi (exponential field): slope {shape.slope:+.3f} log-odds/step, crossing at gap {-shape.intercept / shape.slope:.2f} steps"
    )
    say(
        f"  free vs linear psi: deviance +{lr:.1f} on {shape.df} df (p {stats.chi2.sf(lr, shape.df):.2g});"
        f" largest departure {shape.max_departure_steps:.2f} steps of distance"
    )
    aic = shape.ratio_vs_surface_deviance - 2 * shape.ratio_vs_surface_df
    say(
        f"  ratio model (free psi) vs the h surface (not nested): deviance +{shape.ratio_vs_surface_deviance:.1f} with "
        f"{shape.ratio_vs_surface_df} fewer parameters; AIC(ratio) - AIC(surface) = {aic:+.1f}"
    )
    linear_steps = -(shape.psi - shape.psi[np.isfinite(shape.psi)][0]) / (-shape.slope) if shape.slope else shape.psi
    say("  psi in steps (free psi / |linear slope|; a straight line of slope 1 is exponential):")
    say(
        "    "
        + "  ".join(
            f"d{d}:{v:5.1f}" for d, v, c in zip(shape.distances, linear_steps, shape.counts) if np.isfinite(v) and c >= 200
        )
    )

    # --- global -----------------------------------------------------------------
    fits = {m: fit_model(table, m, surface=surface) for m in ("separable", "near", "width", "full")}
    c = len(rows) - 1
    say("\n--- global separability, all conditions (base reference) ---")
    for m in ("near", "width", "full"):
        d = fits["separable"].deviance - fits[m].deviance
        df = c * (2 if m == "full" else 1)
        say(
            f"  separable -> {m:<6} deviance -{d:8.1f} on {df:>3} df  (chi2 p {stats.chi2.sf(d, df):.2g}; clustered, optimistic)"
        )
    say(f"  residual deviance of full model {fits['full'].deviance:.0f} on {len(table.n) - fits['full'].n_params} df")

    # --- bootstrap ----------------------------------------------------------------
    point, keys = summarise(table, surface, rows)
    boots = bootstrap(decisions, lambda t: summarise(t, surface, rows)[0], resamples=resamples, paired=is_paired)
    low, high = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    index = {k: j for j, k in enumerate(keys)}

    def ci(name, quantity):
        j = index[(name, quantity)]
        return point[j], low[j], high[j]

    def fmt(name, quantity, digits=2):
        p, lo, hi = ci(name, quantity)
        return f"{p:+.{digits}f} [{lo:+.{digits}f},{hi:+.{digits}f}]"

    say("\n--- per condition (full model at the median d_near; tilt = dtau/dd_near over d_near 2..12) ---")
    say(f"  {'condition':<18}{'gain':>7}{'reach':>7}{'route':>7}  {'tau':^22}{'tilt':^22}{'width':^22}{'x near coef':^22}")
    for row in rows:
        gain = next(iter(row["gains"].values())) if row.get("gains") else 1.0
        say(
            f"  {row['name']:<18}{gain:>7.3g}{row['reached']:>7.1%}{row['on_route']:>7.1%}  "
            f"{fmt(row['name'], 'tau'):^22}{fmt(row['name'], 'tilt', 3):^22}{fmt(row['name'], 'width'):^22}{fmt(row['name'], 'near_coef', 3):^22}"
        )
    say("\n  banded nonparametric crossings (steps), d_near bands " + " ".join(f"{a}-{b}" for a, b in BANDS))
    for row in rows:
        say(f"  {row['name']:<18}" + "".join(f"{ci(row['name'], f'band{a}-{b}')[0]:>8.2f}" for a, b in BANDS))

    say("\n--- matched pairs: brightness minus the axis write it matches (same threshold) ---")
    say(f"  {'brightness':<18}{'matches':<12}{'d tau':^22}{'d tilt':^24}{'d width':^22}")
    for row in rows:
        if row["kind"] == "matched":
            say(
                f"  {row['name']:<18}{row['matches']:<12}{fmt(row['name'], 'd_tau'):^22}{fmt(row['name'], 'd_tilt', 3):^24}{fmt(row['name'], 'd_width'):^22}"
            )

    say("\n--- family trends: how contour tilt and width move per step of threshold ---")
    for family in ("axis", "bright_in_range", "bright", "axis-bright_in_range"):
        say(
            f"  {family:<22} tilt/tau {fmt('family:' + family, 'tilt_per_tau', 3):<26} width/tau {fmt('family:' + family, 'width_per_tau', 3)}"
        )

    result = {
        "base": base_name,
        "paired": is_paired,
        "conditions": rows,
        "field_shape": {
            "slope": shape.slope,
            "intercept": shape.intercept,
            "lr_free_vs_linear": lr,
            "df": shape.df,
            "max_departure_steps": shape.max_departure_steps,
            "ratio_vs_surface_deviance": shape.ratio_vs_surface_deviance,
            "ratio_vs_surface_df": shape.ratio_vs_surface_df,
            "psi": shape.psi.tolist(),
            "counts": shape.counts.tolist(),
        },
        "global": {m: {"deviance": f.deviance, "n_params": f.n_params} for m, f in fits.items()},
        "stats": [
            {"condition": k[0], "quantity": k[1], "point": float(point[j]), "low": float(low[j]), "high": float(high[j])}
            for j, k in enumerate(keys)
        ],
        "bands": BANDS,
    }
    return result


def main() -> None:
    args = parse_args()
    lines: list[str] = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    say(__doc__.split("\n\n")[0])
    results = [report_base(d, args.resamples, say) for d in args.data]
    say(f"\n{'=' * 100}\ncross-base agreement")
    for quantity in ("tilt_per_tau", "width_per_tau"):
        for family in ("axis", "bright_in_range", "axis-bright_in_range"):
            cells = []
            for r in results:
                s = next(x for x in r["stats"] if x["condition"] == f"family:{family}" and x["quantity"] == quantity)
                cells.append(f"{s['point']:+.3f} [{s['low']:+.3f},{s['high']:+.3f}]")
            say(f"  {quantity:<14}{family:<22}" + "   ".join(cells))
    if args.txt:
        args.txt.parent.mkdir(parents=True, exist_ok=True)
        args.txt.write_text("\n".join(lines) + "\n")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
