#!/usr/bin/env python3
"""Extract the dose-response figure's data.

    python3 figures/extract_axis_dose.py

Per architecture and tensor: the threshold shift, in steps, from writing that
tensor's slice of the axis at each multiple of its own size (the ``--doses``
mode of scripts/measure/restricted_writes{,_bc}.py), sign-aligned with the full
write and averaged over the two signs, then averaged over bases with the
range kept. Reads figures/data/restricted/<base>.dose.json.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "src" / "axis-dose" / "data.json"
FAMILIES = [("DRC", "novalue11g1"), ("ResNet", "resnet11nvg1"), ("ViT", "vit11nvg1"), ("BC transformer", "bcnv11")]


def curves(run: dict) -> tuple[dict[str, list[float]], float]:
    base = next(w["steps"] for w in run["writes"] if w["kind"] == "base")
    full = {w["offset"]: w["steps"] - base for w in run["writes"] if w["kind"] == "full"}
    sign = {o: float(np.sign(v)) for o, v in full.items()}
    out = {}
    for m in run["modules"]:
        if m == "other":
            continue
        out[m] = [
            float(
                np.mean(
                    [
                        (w["steps"] - base) * sign[w["offset"]]
                        for w in run["writes"]
                        if w["kind"] == "dose" and w["module"] == m and w["dose"] == d
                    ]
                )
            )
            for d in run["doses"]
        ]
    return out, float(np.mean([abs(v) for v in full.values()]))


def main() -> None:
    panels = []
    for title, family in FAMILIES:
        runs = [
            json.loads(pathlib.Path(f).read_text())
            for f in sorted(glob.glob(str(ROOT / "data" / "restricted" / f"{family}.*.dose.json")))
        ]
        if not runs:
            panels.append({"title": title, "family": family, "pending": True})
            continue
        per = [curves(r) for r in runs]
        modules = [m for m in runs[0]["modules"] if m != "other"]
        tensors = {}
        for m in modules:
            arr = np.array([p[0][m] for p in per])
            tensors[m] = {"mean": arr.mean(axis=0).tolist(), "lo": arr.min(axis=0).tolist(), "hi": arr.max(axis=0).tolist()}
        panels.append(
            {
                "title": title,
                "family": family,
                "bases": [re.sub(r"^.*/", "", r["run"]) for r in runs],
                "doses": runs[0]["doses"],
                "modules": modules,
                "tensors": tensors,
                "full_write": float(np.mean([p[1] for p in per])),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"panels": panels}, indent=1) + "\n")
    print(f"wrote {OUT}: " + ", ".join(f"{p['title']} {len(p.get('bases', []))}" for p in panels))


if __name__ == "__main__":
    main()
