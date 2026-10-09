#!/usr/bin/env python3
"""Extract binned psychometric curves for the s-curve figure (fig 4).

    python3 figures/extract_scurves.py

Reads per-episode decode arrays (d_rich, d_poor, reached, reached_fid,
colour_of_rich) and writes figures/src/scurves/data.json: per architecture,
per write dose alpha, the fraction of reached episodes that took the 1.0 goal
in 2-step bins of the distance gap d_rich - d_poor.

Sources (bulk npz on the volume, mirrored under figures/data/), all gamma=1 era:
  DRC    alpha 0     figures/data/drc/g1/novalue11g1.{s1234,s9012,s3456e}.npz  (3 bases)
  DRC    alpha +-0.2 figures/data/drc/scurves-g1/scurve.novalue11g1.s1234.a*.npz
  ResNet alpha 0     figures/data/resnet/base/resnet11nvg1.{s1234,s2468,s3456}.npz
  ResNet alpha +-0.2 figures/data/resnet/scurves/scurve.resnet11nvg1.s1234.a*.npz
  ViT    all alphas  figures/data/vit/{base,scurves}/  (one base, s9012)
  BC     all alphas  figures/data/h1/grid/written.s{1,2,3}/o1{-020,+000,+020}.npz
Write curves come from one base; their band is the per-bin 95% Wilson interval.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "src" / "scurves" / "data.json"

BIN_EDGES = range(-7, 23, 2)  # 2-step bins over integer gaps: {-7,-6}, {-5,-4}, ...
MIN_PER_BIN = 25
Z95 = 1.959964


def binned(path: pathlib.Path) -> list[dict]:
    z = np.load(path)
    gap = z["d_rich"] - z["d_poor"]
    reached = z["reached"]
    took = (z["reached_fid"] == z["colour_of_rich"]) & reached
    rows = []
    for lo in BIN_EDGES:
        m = (gap >= lo) & (gap < lo + 2) & reached
        n = int(m.sum())
        if n >= MIN_PER_BIN:
            rows.append({"gap": lo + 0.5, "p": round(float(took[m].mean()), 4), "n": n})
    return rows


def wilson(p: float, n: int) -> tuple[float, float]:
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def logistic_fit(paths: list[pathlib.Path]) -> tuple[float, float]:
    """Unbinned MLE of P(took rich) = sigmoid((theta - gap)/s), pooled over paths."""
    counts: dict[int, list[int]] = {}
    for path in paths:
        z = np.load(path)
        gap = z["d_rich"] - z["d_poor"]
        reached = z["reached"]
        took = (z["reached_fid"] == z["colour_of_rich"]) & reached
        for g in np.unique(gap[reached]):
            m = reached & (gap == g)
            row = counts.setdefault(int(g), [0, 0])
            row[0] += int(took[m].sum())
            row[1] += int(m.sum())
    gaps = np.array(sorted(counts))
    k = np.array([counts[g][0] for g in gaps], dtype=float)
    n = np.array([counts[g][1] for g in gaps], dtype=float)
    thetas = np.arange(gaps.min(), gaps.max() + 0.01, 0.02)
    scales = np.arange(0.2, 8.0, 0.02)
    x = (thetas[:, None, None] - gaps[None, None, :]) / scales[None, :, None]
    logp = -np.logaddexp(0.0, -x)
    log1mp = -np.logaddexp(0.0, x)
    ll = (k * logp + (n - k) * log1mp).sum(axis=-1)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return round(float(thetas[i]), 2), round(float(scales[j]), 2)


def combine(base_rows: list[list[dict]]) -> list[dict]:
    """Mean and band across bases (band = range), or Wilson CI for one base."""
    if len(base_rows) == 1:
        out = []
        for r in base_rows[0]:
            lo, hi = wilson(r["p"], r["n"])
            out.append({**r, "lo": round(lo, 4), "hi": round(hi, 4)})
        return out
    shared = sorted(set.intersection(*(set(r["gap"] for r in rows) for rows in base_rows)))
    out = []
    for g in shared:
        ps = [next(r["p"] for r in rows if r["gap"] == g) for rows in base_rows]
        ns = [next(r["n"] for r in rows if r["gap"] == g) for rows in base_rows]
        out.append(
            {
                "gap": g,
                "p": round(sum(ps) / len(ps), 4),
                "lo": round(min(ps), 4),
                "hi": round(max(ps), 4),
                "n": sum(ns),
            }
        )
    return out


def curve(alpha: float, sources: dict[str, pathlib.Path]) -> dict:
    paths = list(sources.values())
    theta, scale = logistic_fit(paths)
    per_base = [logistic_fit([p]) for p in paths]
    return {
        "alpha": alpha,
        "sources": {name: str(p.relative_to(ROOT)) for name, p in sources.items()},
        "theta": theta,
        "scale": scale,
        "theta_per_base": [t for t, _ in per_base],
        "scale_per_base": [s for _, s in per_base],
        "band": "bases" if len(sources) > 1 else "ci95",
        "bins": combine([binned(p) for p in paths]),
    }


def main() -> None:
    drc_scurve = DATA / "drc" / "scurves-g1"
    bc_written = DATA / "h1" / "grid"
    panels = [
        {
            "key": "drc",
            "title": "DRC",
            "curves": [
                curve(-0.2, {"s1234": drc_scurve / "scurve.novalue11g1.s1234.a-0.20.npz"}),
                curve(
                    0.0,
                    {s: DATA / "drc" / "g1" / f"novalue11g1.{s}.npz" for s in ("s1234", "s9012", "s3456e")},
                ),
                curve(0.2, {"s1234": drc_scurve / "scurve.novalue11g1.s1234.a+0.20.npz"}),
            ],
        },
        {
            "key": "resnet",
            "title": "ResNet",
            "curves": [
                curve(-0.2, {"s1234": DATA / "resnet" / "scurves" / "scurve.resnet11nvg1.s1234.a-0.20.npz"}),
                curve(
                    0.0,
                    {s: DATA / "resnet" / "base" / f"resnet11nvg1.{s}.npz" for s in ("s1234", "s2468", "s3456")},
                ),
                curve(0.2, {"s1234": DATA / "resnet" / "scurves" / "scurve.resnet11nvg1.s1234.a+0.20.npz"}),
            ],
        },
        {
            "key": "vit",
            "title": "ViT",
            "curves": [
                curve(-0.2, {"s9012": DATA / "vit" / "scurves" / "scurve.vit11nvg1.s9012.a-0.20.npz"}),
                curve(0.0, {"s9012": DATA / "vit" / "base" / "vit11nvg1.s9012.npz"}),
                curve(0.2, {"s9012": DATA / "vit" / "scurves" / "scurve.vit11nvg1.s9012.a+0.20.npz"}),
            ],
        },
        {
            "key": "bc",
            "title": "BC transformer",
            "curves": [
                curve(a, {s: bc_written / f"written.{s}" / f"o1{tag}.npz" for s in ("s1", "s2", "s3")})
                for a, tag in ((-0.2, "-020"), (0.0, "+000"), (0.2, "+020"))
            ],
        },
    ]
    payload = {
        "step_penalty": 0.05,
        "base_values": {"o0": 1.0, "o1": 0.5},
        "swept_objective": "o1",
        "doses": [-0.2, 0.0, 0.2],
        "optimum_gap": 10.0,
        "panels": panels,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    for p in panels:
        if p.get("pending"):
            print(f"{p['title']:15s} pending")
            continue
        for c in p["curves"]:
            print(
                f"{p['title']:15s} alpha {c['alpha']:+.1f}  theta {c['theta']}  "
                f"scale {c['scale']}  per-base {c['theta_per_base']}"
            )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
