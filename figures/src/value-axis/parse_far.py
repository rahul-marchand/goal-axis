#!/usr/bin/env python3
"""Turn the far-past-parity measures into the appendix figure's ``farpanels``.

    python3 figures/src/value-axis/parse_far.py --results results --bc figures/data/bc

Same four panels as ``parse_g1.py`` builds for fig 3, carried as far out as
anything was measured. Fig 3 stops at ``PLOT_LIMIT`` because that is where the
story is decided; this file exists to show what happens after it, so nothing is
windowed here and the reach floor is recorded rather than silently applied.

Per sweep, per base, three series in offset units:

* ``trained``: the fitted-window arms from the ``o`` tables plus every
  colour-keyed ``k`` arm, however far out it was trained.
* ``written``: the base agent with the axis written at ``dv``. Read from the
  ``-far`` tables (``014 --extrapolate`` re-run with a long list) where they
  exist, else from the ordinary ``o`` tables, so the parser degrades to fig 3's
  reach while the far measures are still in flight.
* ``exhausted``: the first ``|dv|`` at which a written agent stopped finishing
  episodes (reach below ``REACH_FLOOR``). Rows past it are dropped from the
  series -- a threshold read off a handful of survivors is noise -- and the
  builder draws the cut instead.

Written points below the reach floor are the write breaking the agent, which
is a different failure from the write saturating; the appendix needs both
visible, so the floor is a field, not a filter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # run with plain python3 as well as uv

from goalmisgen.analysis import tables  # noqa: E402

# The figure's own choices -- which bases, the even grid, the reach floor -- are
# parse_g1's, so that fig 3 and its appendix cannot disagree about them.
sys.path.insert(0, str(HERE))
from parse_g1 import BASE_VALUE, EVEN_GRID, FAMILIES, REACH_FLOOR  # noqa: E402


def read_rows_keep_reach(path: Path) -> tuple[dict[float, dict], dict[float, dict], dict | None]:
    """Like ``parse_g1.read_rows`` but nothing is dropped; ``reach`` travels with each row."""
    if not path.is_file():
        return {}, {}, None
    text = path.read_text()
    trained: dict[float, dict] = {}
    written: dict[float, dict] = {}
    for row in tables.value_rows(text):
        target = trained if row.kind == "fine-tuned" else written
        target[row.value] = {"value": row.value, "steps": row.point, "lo": row.low, "hi": row.high, "reach": row.reach}
    found = tables.base_row(text)
    base = None if found is None else {"steps": found.point, "lo": found.low, "hi": found.high, "reach": found.reach}
    return trained, written, base


def offset_rows(rows: dict[float, dict], base_value: float) -> list[dict]:
    return [{"dv": round(v - base_value, 2), **{k: r[k] for k in ("steps", "lo", "hi", "reach")}} for v, r in rows.items()]


def apply_floor(rows: list[dict]) -> tuple[list[dict], float | None]:
    """Drop rows past the first reach failure, in order of distance from the base."""
    rows = sorted(rows, key=lambda r: abs(r["dv"]))
    kept: list[dict] = []
    for row in rows:
        if row.get("reach", 100.0) < REACH_FLOOR:
            return sorted(kept, key=lambda r: r["dv"]), abs(row["dv"])
        kept.append(row)
    return sorted(kept, key=lambda r: r["dv"]), None


def rl_series(results: Path, run: str, sweep: str) -> dict | None:
    crossing_family = {"o0": "k0", "o1": "k1"}[sweep]
    value = BASE_VALUE[sweep]
    trained, written, base = read_rows_keep_reach(results / f"g1-{run}-{sweep}.txt")
    if base is None:
        return None
    far_path = results / f"g1-{run}-{sweep}-far.txt"
    _, far_written, far_base = read_rows_keep_reach(far_path)
    source = "far" if far_base is not None else "o"
    if far_base is not None:
        # the far table is a superset re-measure of the same axis; prefer it row for row
        written = {**written, **far_written}
        base = far_base
    crossing, _, _ = read_rows_keep_reach(results / f"g1-{run}-{crossing_family}.txt")
    trained.update(crossing)

    out_trained = offset_rows(trained, value)
    out_written = offset_rows(written, value)
    out_written.append({"dv": 0.0, **base})
    # the crossing side is the only side anything extends on; keep the fitted
    # window on the other side so the panel still shows the axis interpolating
    out_trained, trained_cut = apply_floor(out_trained)
    out_written, written_cut = apply_floor(out_written)
    return {
        "name": run.split(".", 1)[1],
        "trained": out_trained,
        "written": out_written,
        "exhausted": {"trained": trained_cut, "written": written_cut},
        "written_source": source,
    }


def bc_series(bc: Path, results: Path, seed: int, sweep: str) -> dict | None:
    crossing_family = {"o0": "k0", "o1": "k1"}[sweep]
    path = bc / f"value_axis.bcnv11.s{seed}.{sweep}.json"
    if not path.is_file():
        return None
    behaviour = json.loads(path.read_text())["behaviour"]
    trained, written = [], []
    for key, row in behaviour["arms"].items():
        dv = round(float(key), 2)
        if not any(abs(dv - g) < 1e-9 for g in EVEN_GRID):
            continue
        trained.append({"dv": dv, "steps": row["arm"]["indifference"], "reach": 100.0 * row["arm"].get("reached", 1.0)})
        written.append({"dv": dv, "steps": row["written"]["indifference"], "reach": 100.0 * row["written"].get("reached", 1.0)})
    trained.append({"dv": 0.0, "steps": behaviour["null"]["indifference"], "reach": 100.0 * behaviour["null"].get("reached", 1.0)})
    written.append({"dv": 0.0, "steps": behaviour["base"]["indifference"], "reach": 100.0 * behaviour["base"].get("reached", 1.0)})

    crossing, _, _ = read_rows_keep_reach(results / f"g1-bcnv11.s{seed}-{crossing_family}.txt")
    trained.extend(offset_rows(crossing, BASE_VALUE[crossing_family]))

    sign = -1.0 if sweep == "o0" else 1.0
    source = "o"
    for name in ("extrap_axisonly_far.json", "extrap_axisonly.json"):
        sidecar = bc / name
        if not sidecar.is_file():
            continue
        rows = json.loads(sidecar.read_text()).get(f"s{seed}", {}).get(sweep, {})
        if not rows:
            continue
        source = "far" if name.endswith("_far.json") else "near"
        for key, row in rows.items():
            dv = round(float(key), 2)
            if dv * sign <= 0:
                continue
            written.append({"dv": dv, "steps": row["indifference"], "reach": 100.0 * row["reached"]})
        break
    trained, trained_cut = apply_floor(trained)
    written, written_cut = apply_floor(written)
    return {
        "name": f"s{seed}",
        "trained": trained,
        "written": written,
        "exhausted": {"trained": trained_cut, "written": written_cut},
        "written_source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bc", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data.json")
    args = parser.parse_args()

    panels = []
    for key, title, prefix, seeds in FAMILIES:
        sweeps: dict[str, list[dict]] = {"o0": [], "o1": []}
        for seed in seeds:
            for sweep in ("o0", "o1"):
                found = rl_series(args.results, f"{prefix}.{seed}", sweep)
                if found is None:
                    print(f"  {prefix}.{seed} {sweep}: no measure, skipped")
                    continue
                sweeps[sweep].append(found)
        panels.append({"key": key, "title": title, "sweeps": sweeps})
    sweeps = {"o0": [], "o1": []}
    for seed in (1, 2, 3):
        for sweep in ("o0", "o1"):
            found = bc_series(args.bc, args.results, seed, sweep)
            if found is not None:
                sweeps[sweep].append(found)
    panels.append({"key": "bc", "title": "BC transformer", "sweeps": sweeps})

    payload = json.loads(args.out.read_text()) if args.out.is_file() else {}
    payload["farpanels"] = panels
    args.out.write_text(json.dumps(payload, indent=1))
    for panel in panels:
        for sweep, series in panel["sweeps"].items():
            for s in series:
                t = [r["dv"] for r in s["trained"]]
                w = [r["dv"] for r in s["written"]]
                cut = s["exhausted"]
                print(
                    f"{panel['key']:7s} {sweep} {s['name']:7s} trained {min(t):+.1f}..{max(t):+.1f} "
                    f"({len(t)})  written[{s['written_source']}] {min(w):+.1f}..{max(w):+.1f} ({len(w)})"
                    + (f"  written reach fails at |dv|={cut['written']}" if cut["written"] else "")
                    + (f"  trained reach fails at |dv|={cut['trained']}" if cut["trained"] else "")
                )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
