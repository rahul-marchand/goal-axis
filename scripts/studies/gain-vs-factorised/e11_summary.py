"""The route-model scaling table: where each model's value axis can be written.

    uv run python scripts/studies/gain-vs-factorised/e11_summary.py /workspace/data/figures-data/gain-vs-factorised \
        --models d128l4 d256l4 d512l4 d512l16

Reads the JSON that ``cheapest_edit.py`` writes for each family and prints one
row per (model, family): the reach -- the slope of the held-out threshold on the
dose it was trained for, one meaning the edit does what it was asked -- the
edit's size and parameter count, how much competence it costs at the extreme
doses, and how well it matches the fine-tuned axis in weight and function space.

A family that reaches one with a few thousand parameters at the head is a value
knob applied after the plan; a family that only reaches one when it may touch
the blocks is value fused into the representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--prefix", default="e11-")
    return parser.parse_args()


def reach(evaluations: list[dict]) -> float:
    """Slope of the held-out threshold on the requested shift."""
    shifts = np.array([e["shift"] for e in evaluations], dtype=float)
    taus = np.array([e["tau"] for e in evaluations], dtype=float)
    keep = np.isfinite(taus)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(shifts[keep], taus[keep], 1)[0])


def main() -> None:
    args = parse_args()
    rows = []
    for model in args.models:
        for path in sorted(args.directory.glob(f"{args.prefix}{model}.*.json")):
            tag = path.stem.split(".", 1)[1]
            payload = json.loads(path.read_text())
            runs = payload.get("runs", [])
            if not runs:
                continue
            reaches = [reach(run["eval"]) for run in runs]
            worst = min(
                (e["reached"] for run in runs for e in run["eval"] if np.isfinite(e.get("tau", np.nan))),
                default=float("nan"),
            )
            route = min(
                (e["on_route"] for run in runs for e in run["eval"] if np.isfinite(e.get("tau", np.nan))),
                default=float("nan"),
            )
            axis = payload.get("axis") or {}
            rows.append(
                {
                    "model": model,
                    "family": tag,
                    "parameters": runs[0]["edit_parameters"],
                    "norm": float(np.mean([run["norm"] for run in runs])),
                    "reach": float(np.mean(reaches)),
                    "reach_spread": float(np.max(reaches) - np.min(reaches)) if len(reaches) > 1 else 0.0,
                    "reached": worst,
                    "route": route,
                    "seed_cos": float(np.mean(payload.get("seed_cosines", [np.nan]))),
                    "cos_uv": axis.get("weight_cos", {}).get("mean", float("nan")),
                    "fn_cos": axis.get("function_cos", {}).get("mean", {}).get("v", float("nan")),
                    "fn_cos_fork": axis.get("function_cos_fork", {}).get("mean", {}).get("v", float("nan")),
                    "reliability": axis.get("reliability", float("nan")),
                    "tau_base": payload.get("base_eval", {}).get("tau", float("nan")),
                }
            )

    print(
        f"{'model':<9}{'family':<18}{'params':>11}{'|u|/step':>10}{'reach':>8}{'spread':>8}{'reached':>9}{'route':>8}{'seed cos':>10}{'cos(u,v)':>10}{'fn cos':>8}{'fork':>7}"
    )
    for row in rows:
        print(
            f"{row['model']:<9}{row['family']:<18}{row['parameters']:>11,}{row['norm']:>10.3f}"
            f"{row['reach']:>8.2f}{row['reach_spread']:>8.2f}{row['reached']:>9.1%}{row['route']:>8.1%}"
            f"{row['seed_cos']:>10.3f}{row['cos_uv']:>10.3f}{row['fn_cos']:>8.3f}{row['fn_cos_fork']:>7.3f}"
        )
    print("\n  reach = slope of held-out tau on the trained dose; 1 means the edit does what it was asked.")
    print("  reached / route are the worst over evaluated doses, so they show what the edit costs at the extremes.")


if __name__ == "__main__":
    main()
