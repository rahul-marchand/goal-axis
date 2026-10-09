#!/usr/bin/env python3
"""Turn the g1 (undiscounted) measure tables into the figure's ``xpanels``.

    python3 figures/src/value-axis/parse_g1.py --results <dir of g1-*.txt> --bc <dir of value_axis.bcnv11.*.json>

One panel per architecture, each carrying both sweeps. Within a sweep the
series are assembled from three sources, because one 014 run cannot produce
them all:

* ``o<n>`` tables give the fitted window: arms trained at +-0.1..0.4 of the
  base value, and the same offsets written from the axis fitted without them.
* ``x<n>`` tables give the arms trained *past* preference parity. They are a
  separate 014 run only because ``discover_arms`` hides held-out families from
  the fitted one; the axis those runs print is unbalanced by construction and
  is ignored here — only the arms' own behaviour is read.
* ``written (unseen)`` rows in the ``o`` tables are the same axis written at
  those past-parity offsets, which is what makes the saturation visible: the
  trained arms cross the preference flip, the written ones flatten.

Bases are listed explicitly rather than globbed: the g1 campaign also produced
seeds that never converged, and a directory listing cannot tell them apart.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROW = re.compile(
    r"^\s+v=(?P<value>-?\d\.\d\d)\s+(?P<kind>fine-tuned|written, held out|written \(grid\)|written \(unseen\))"
    r"\s+(?P<steps>-?\d+\.\d)\s+\[\s*(?P<lo>-?\d+\.\d),\s*(?P<hi>-?\d+\.\d)\]\s+(?P<reach>\d+\.\d)%",
    re.MULTILINE,
)
BASE = re.compile(r"^\s+base, untouched\s+(-?\d+\.\d)\s+\[\s*(-?\d+\.\d),\s*(-?\d+\.\d)\]\s+(\d+\.\d)%", re.MULTILINE)

# Which bases entered the figure, per architecture. Everything else the g1
# campaign trained either failed to converge within its budget or was a
# measurement of one that did not; see the campaign notes.
FAMILIES = [
    ("drc", "DRC", "novalue11g1", ["s1234", "s9012", "s3456e"]),
    ("resnet", "ResNet", "resnet11nvg1", ["s1234", "s2468", "s3456"]),
    ("vit", "ViT", "vit11nvg1", ["s9012"]),
]
BASE_VALUE = {"o0": 1.0, "x0": 1.0, "k0": 1.0, "o1": 0.5, "x1": 0.5, "k1": 0.5}
EVEN_GRID = tuple(round(0.1 * k, 1) for k in range(-4, 5))
"""The offsets every panel is drawn at, so the panels are visually comparable.

The RL sweeps were trained on exactly this grid. bcnv11 was swept on the older
leverage-optimised design -- a cluster at 0.38..0.45 plus sparse interior
points -- which contains this grid as a subset, so BC needs no retraining to
match: its extra arms still constrain its axis fit, they are simply not drawn.
"""
PLOT_LIMIT = 0.75
"""How far past the fitted window the figure runs.

Far enough for the trained arms to carry the threshold through zero and for the
written ones to be seen failing to follow, and no further: beyond this the
agents stop finishing episodes and a threshold read off the survivors is noise.
"""
DV_WINDOW = 0.45
"""The offsets the axis was fitted from. Everything outside is extrapolation.

The figure shades it, because a written point inside it is the axis
interpolating and a written point outside it is the axis extrapolated past
every arm it was fitted from -- which is exactly where it is expected to fail,
and does.
"""
REACH_FLOOR = 50.0
"""Rows below this are the agent failing to finish episodes, not a threshold.

A threshold read off a handful of finished episodes is noise with a confidence
interval drawn around it, and the far end of the extended grid is exactly where
an agent stops finishing. Dropping them is a statement about what the number
means, not about which points look better.
"""


def read_rows(path: Path) -> tuple[dict[float, dict], dict[float, dict], dict | None]:
    """``(trained, written, base)`` keyed by the value each row is measured at."""
    if not path.is_file():
        return {}, {}, None
    text = path.read_text()
    trained: dict[float, dict] = {}
    written: dict[float, dict] = {}
    for match in ROW.finditer(text):
        row = {
            "value": float(match.group("value")),
            "steps": float(match.group("steps")),
            "lo": float(match.group("lo")),
            "hi": float(match.group("hi")),
            "reach": float(match.group("reach")),
        }
        if row["reach"] < REACH_FLOOR:
            continue
        target = trained if match.group("kind") == "fine-tuned" else written
        target[row["value"]] = row
    found = BASE.search(text)
    base = None
    if found:
        base = {"steps": float(found.group(1)), "lo": float(found.group(2)),
                "hi": float(found.group(3)), "reach": float(found.group(4))}
    return trained, written, base


def series(results: Path, run: str, sweep: str) -> tuple[list[dict], list[dict]] | None:
    """One base's trained and written points for one sweep, in offset units."""
    held_out = {"o0": "k0", "o1": "k1"}[sweep]
    trained, written, base = read_rows(results / f"g1-{run}-{sweep}.txt")
    if base is None:
        return None
    extended, _, _ = read_rows(results / f"g1-{run}-{held_out}.txt")
    trained.update(extended)

    value = BASE_VALUE[sweep]
    out_trained = [{"dv": round(v - value, 2), **{k: r[k] for k in ("steps", "lo", "hi")}} for v, r in trained.items()]
    out_written = [{"dv": round(v - value, 2), **{k: r[k] for k in ("steps", "lo", "hi")}} for v, r in written.items()]
    # The unedited agent is the written series' own zero: nothing was written.
    out_written.append({"dv": 0.0, "steps": base["steps"], "lo": base["lo"], "hi": base["hi"]})
    order = lambda rows: sorted((r for r in rows if abs(r["dv"]) <= PLOT_LIMIT), key=lambda r: r["dv"])
    return order(out_trained), order(out_written)


def limit_series(results: Path, run: str, sweep: str) -> tuple[list[dict], list[dict]] | None:
    """The same sweep carried past parity, for the signed-threshold figure.

    Differs from :func:`series` in two ways. The extension arms come from the
    ``k`` family, whose levels keep colour ``i`` on objective ``i`` so that an
    arm past parity is asked to reverse a preference rather than to relabel one;
    and nothing is windowed, because the whole point is what happens outside the
    window the axis was fitted in.
    """
    crossing_family = {"o0": "k0", "o1": "k1"}[sweep]
    trained, written, base = read_rows(results / f"g1-{run}-{sweep}.txt")
    if base is None:
        return None
    crossing, _, _ = read_rows(results / f"g1-{run}-{crossing_family}.txt")
    trained.update(crossing)
    value = BASE_VALUE[sweep]
    out_trained = [{"dv": round(v - value, 2), **{k: r[k] for k in ("steps", "lo", "hi")}} for v, r in trained.items()]
    out_written = [{"dv": round(v - value, 2), **{k: r[k] for k in ("steps", "lo", "hi")}} for v, r in written.items()]
    out_written.append({"dv": 0.0, "steps": base["steps"], "lo": base["lo"], "hi": base["hi"]})
    order = lambda rows: sorted(rows, key=lambda r: r["dv"])
    return order(out_trained), order(out_written)


def bc_panel(bc: Path, results: Path) -> dict:
    """The imitation-trained transformer, whose measures are json, not tables.

    Its crossing region comes from two places the agents' does not. The trained
    arms are ``k``-family fine-tunes measured by ``scripts/bc_arm_thresholds.py``,
    which prints the same table the agents' measures do; the written points are
    ``027``'s extrapolation controls, run at the offsets the agents carry. Both
    are read only on the crossing side of each sweep -- lowering colour 0, or
    raising colour 1 -- because that is the side where the axis is asked for a
    reversal, and the only side the agents' panels extend on.
    """
    sweeps: dict[str, list[dict]] = {"o0": [], "o1": []}
    for sweep in ("o0", "o1"):
        crossing_family = {"o0": "k0", "o1": "k1"}[sweep]
        for seed in (1, 2, 3):
            path = bc / f"value_axis.bcnv11.s{seed}.{sweep}.json"
            if not path.is_file():
                continue
            behaviour = json.loads(path.read_text())["behaviour"]
            trained, written = [], []
            for key, row in behaviour["arms"].items():
                dv = round(float(key), 2)
                trained.append({"dv": dv, "steps": row["arm"]["indifference"]})
                # axis-only, matching the agents' write convention; 027 stores
                # the drift-included variant too, deliberately unused here
                written.append({"dv": dv, "steps": row["written"]["indifference"]})
            trained.append({"dv": 0.0, "steps": behaviour["null"]["indifference"]})
            written.append({"dv": 0.0, "steps": behaviour["base"]["indifference"]})
            keep = lambda rows: [r for r in rows if any(abs(r["dv"] - g) < 1e-9 for g in EVEN_GRID)]
            trained, written = keep(trained), keep(written)

            crossing, _, _ = read_rows(results / f"g1-bcnv11.s{seed}-{crossing_family}.txt")
            for value, row in crossing.items():
                dv = round(value - BASE_VALUE[crossing_family], 2)
                if abs(dv) <= PLOT_LIMIT:
                    trained.append({"dv": dv, **{k: row[k] for k in ("steps", "lo", "hi")}})
            # The value_axis JSONs on disk were measured when 027 stored only
            # the drift-included extrapolation (extrapolate_with_drift_*); the
            # axis-only re-measures live in a sidecar written by the
            # bc_extrap_probe pod job. Rows without an axis-only measure are
            # omitted rather than plotted in the wrong convention.
            sidecar = bc / "extrap_axisonly.json"
            axisonly = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            for key, row in axisonly.get(f"s{seed}", {}).get(sweep, {}).items():
                dv = round(float(key), 2)
                sign = -1.0 if sweep == "o0" else 1.0
                if dv * sign <= 0 or abs(dv) > PLOT_LIMIT or row["reached"] * 100 < REACH_FLOOR:
                    continue
                written.append({"dv": dv, "steps": row["indifference"]})

            sweeps[sweep].append({
                "name": f"s{seed}",
                "trained": sorted(trained, key=lambda r: r["dv"]),
                "written": sorted(written, key=lambda r: r["dv"]),
            })
    return {"key": "bc", "title": "BC transformer", "sweeps": sweeps}


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
                found = series(args.results, f"{prefix}.{seed}", sweep)
                if found is None:
                    print(f"  {prefix}.{seed} {sweep}: no measure yet, skipped")
                    continue
                trained, written = found
                sweeps[sweep].append({"name": seed, "trained": trained, "written": written})
        panels.append({"key": key, "title": title, "sweeps": sweeps})
    panels.append(bc_panel(args.bc, args.results))


    payload = json.loads(args.out.read_text()) if args.out.is_file() else {}
    payload["step_penalty"] = payload.get("step_penalty", 0.05)
    payload["discount"] = 1.0
    payload["xpanels"] = panels
    payload["fit_window"] = DV_WINDOW
    args.out.write_text(json.dumps(payload, indent=1))
    for panel in panels:
        counts = {s: len(v) for s, v in panel["sweeps"].items()}
        spans = [f"{r['dv']:+.1f}" for v in panel["sweeps"].values() for s in v for r in s["trained"]]
        print(f"{panel['key']:8s} bases {counts}  dv {min(spans, default='-')}..{max(spans, default='-')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
