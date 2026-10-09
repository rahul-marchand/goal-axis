"""Blend the two objectives' colours towards each other, in the observation only.

The full swap (``colour_swap.py``) is forced: colour is the only thing telling
the objectives apart, so exchanging the planes must invert the threshold. The
informative version is the path between. With

    object A -> (1 - alpha) * colour 0 + alpha * colour 1
    object B -> (1 - alpha) * colour 1 + alpha * colour 0

alpha = 0 is the base, alpha = 0.5 makes the two objects identical, alpha = 1 is
the swap. Both objectives move, so the symmetric point is the midpoint, and it
is the midpoint that separates two accounts of how value is keyed to colour:

* **graded** -- value rides on a continuous colour code, so tau falls smoothly,
  and at alpha = 0.5 the agent faces two identical objects, is indifferent, and
  still plays well (going to the nearer one is correct for identical objects);
* **categorical** -- the network classifies "which colour is this" and looks up
  a value, so tau stays near its base while the classification survives and
  jumps near the midpoint, where the classifier is confused: reach falls and the
  choice curve blows up.

The zero crossing at alpha = 0.5 is forced by symmetry under either account, so
the evidence is the SHAPE away from the midpoint, the COMPETENCE at it, and the
WIDTH of the choice curve there -- not the crossing.

Width is the logistic scale in steps, from an IRLS fit of the choice on the
distance gap; a wider curve means a less decisive agent at a given gap.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.provenance import header

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def blend_planes(observations: np.ndarray, alpha: float, channel_axis: int) -> np.ndarray:
    """Copy of ``observations`` with each colour plane mixed ``alpha`` towards the other."""
    out = np.array(observations, dtype=np.float32, copy=True)
    moved = np.moveaxis(out, channel_axis, -1)
    a, b = COLOUR
    p0, p1 = moved[..., a].copy(), moved[..., b].copy()
    moved[..., a] = (1 - alpha) * p0 + alpha * p1
    moved[..., b] = (1 - alpha) * p1 + alpha * p0
    return out


def logistic_width(gaps: np.ndarray, took_richer: np.ndarray) -> float:
    """Logistic scale in steps: IRLS fit of choice on gap, width = 1/|slope|."""
    x = np.stack([np.ones_like(gaps, dtype=np.float64), np.asarray(gaps, dtype=np.float64)], axis=1)
    y = np.asarray(took_richer, dtype=np.float64)
    beta = np.zeros(2)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-x @ beta))
        w = np.clip(p * (1 - p), 1e-6, None)
        try:
            step = np.linalg.solve((x * w[:, None]).T @ x, x.T @ (y - p))
        except np.linalg.LinAlgError:
            return float("nan")
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    slope = abs(beta[1])
    return float("inf") if slope < 1e-9 else float(1.0 / slope)


def main() -> None:
    args = parse_args()
    print(header())
    from goalmisgen.analysis.behaviour import on_shortest_route, value_distance_decisions
    from goalmisgen.axis import Greedy, load_base

    base = load_base(args.run, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed)
    checkpoint, values = base.checkpoint, base.values
    params = base.unravel(np.asarray(base.raveled, dtype=np.float32))
    print(f"base   {checkpoint}, values {values}\n")

    greedy = Greedy.of(base)
    out: dict = {"run": str(args.run), "checkpoint": str(checkpoint), "values": list(values), "rows": []}

    print(f"  {'alpha':>6}{'steps':>9}  {'95% interval':^15}{'width':>8}{'reached':>9}{'route':>9}")
    for alpha in args.alphas:
        found = greedy.threshold(params, args.episodes, args.seed, transform=partial(blend_planes, alpha=alpha, channel_axis=1))
        point, low, high, outcomes = found.point, found.low, found.high, found.outcomes
        gaps, took_richer, _ = value_distance_decisions(outcomes)
        width = logistic_width(gaps, took_richer)
        reached = found.reached
        route = on_shortest_route(outcomes)
        print(f"  {alpha:>6.2f}{point:>9.2f}  [{low:6.1f},{high:6.1f}]{width:>8.2f}{reached:>9.1%}{route:>9.1%}",
              flush=True)
        out["rows"].append({"alpha": float(alpha), "steps": float(point), "low": float(low), "high": float(high),
                            "width": float(width), "reached": float(reached), "on_route": float(route),
                            "decisions": int(len(gaps))})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
