#!/usr/bin/env python3
"""Extract the where-the-axis appendix figure's data.

    python3 figures/extract_where_axis.py

Per architecture and module, in forward order: the module's share of the
sweep-agreeing axis (minus the per-module inner product of the two sweeps'
fitted axes, normalised over the network, from figures/data/axis_concentration.json),
its share of the parameters, and three restricted writes in steps from
figures/data/restricted/<base>.o1.json (scripts/restricted_writes{,_bc}.py):
written alone, written alone rescaled to the axis length, and lost when left
out (the full write minus "everything except"). Each is the mean absolute
shift over the two signs of the +-0.30 write, then mean and range over bases.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "src" / "where-axis" / "data.json"
CONC = json.loads((ROOT / "data" / "axis_concentration.json").read_text())

FAMILIES = [("DRC", "novalue11g1"), ("ResNet", "resnet11nvg1"), ("ViT", "vit11nvg1"), ("BC transformer", "bcnv11")]


def module_of(path: str) -> str:
    import sys

    sys.path.insert(0, str(ROOT.parent))
    from goalmisgen.analysis.layers import module_of as _module_of

    return _module_of(path)


def share(run: dict, order: list[str]) -> tuple[list[float], list[float]]:
    leaves = CONC[re.sub(r"^.*/", "", run["run"])]["leaves"]
    sig = {m: 0.0 for m in order}
    n = {m: 0 for m in order}
    for path, leaf in leaves.items():
        group = module_of(path)
        if group in sig:
            sig[group] += -leaf["dot01"]
            n[group] += leaf["n"]
    total = sum(sig.values())
    n_all = sum(leaf["n"] for leaf in leaves.values())
    return [max(sig[m], 0.0) / total for m in order], [n[m] / n_all for m in order]


def writes(run: dict, order: list[str]) -> tuple[dict[str, list[float]], float, list[float]]:
    base = next(w["steps"] for w in run["writes"] if w["kind"] == "base")
    full = {w["offset"]: w["steps"] - base for w in run["writes"] if w["kind"] == "full"}
    sign = {o: float(np.sign(v)) for o, v in full.items()}
    acc: dict[tuple[str, str], list[float]] = {}
    stretch: dict[str, float] = {}
    for w in run["writes"]:
        if w["kind"] in ("alone", "rescaled", "without"):
            acc.setdefault((w["kind"], w["module"]), []).append((w["steps"] - base) * sign[w["offset"]])
            if w["kind"] == "rescaled":
                stretch[w["module"]] = w["scale"]
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    full_abs = float(np.mean([abs(v) for v in full.values()]))
    cols = {
        "alone": [mean.get(("alone", m), float("nan")) for m in order],
        "rescaled": [mean.get(("rescaled", m), float("nan")) for m in order],
        "lost": [full_abs - mean.get(("without", m), float("nan")) for m in order],
    }
    return cols, full_abs, [stretch.get(m, float("nan")) for m in order]


def summarise(rows: list[list[float]]) -> list[dict]:
    arr = np.array(rows, dtype=float)
    return [
        {"mean": float(np.nanmean(arr[:, i])), "lo": float(np.nanmin(arr[:, i])), "hi": float(np.nanmax(arr[:, i]))}
        for i in range(arr.shape[1])
    ]


def main() -> None:
    panels = []
    for title, family in FAMILIES:
        runs = [json.loads(pathlib.Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / "data" / "restricted" / f"{family}.*.o1.json")))]
        if not runs:
            panels.append({"title": title, "family": family, "pending": True})
            continue
        order = [m for m in runs[0]["modules"] if m != "other"]
        shares = [share(r, order)[0] for r in runs]
        params = share(runs[0], order)[1]
        per_run = [writes(r, order) for r in runs]
        panels.append(
            {
                "title": title,
                "family": family,
                "bases": [re.sub(r"^.*/", "", r["run"]) for r in runs],
                "modules": order,
                "params_share": params,
                "axis_share": summarise(shares),
                "alone": summarise([p[0]["alone"] for p in per_run]),
                "rescaled": summarise([p[0]["rescaled"] for p in per_run]),
                "lost": summarise([p[0]["lost"] for p in per_run]),
                "full_write": float(np.mean([p[1] for p in per_run])),
                "stretch": per_run[0][2],
                "offsets": runs[0]["write_offsets"],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"panels": panels, "max_stretch": 8.0}, indent=1) + "\n")
    print(f"wrote {OUT}: " + ", ".join(f"{p['title']} {len(p.get('bases', []))} bases" for p in panels))


if __name__ == "__main__":
    main()
