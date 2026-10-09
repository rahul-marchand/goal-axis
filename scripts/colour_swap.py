"""Repaint the objectives in each other's colour, in the observation only.

Every edit in this study changes weights. This one changes nothing: it permutes
the two colour planes of the observation the network is fed, which is on
distribution -- every square is still a clean 1.0 in exactly one plane -- and
needs no fitted direction. If the value of an objective is carried by its colour
code, then presenting the rich objective in the poor colour should invert the
threshold.

The swap moves two things at once, because a colour's signature has a length and
a direction and the two colours differ in both (DRC s1234: |K0|/|K1| = 1.91,
31.5 degrees apart). So four conditions, not two:

* ``base``            -- untouched.
* ``swap``            -- planes exchanged: both direction and strength swapped.
* ``swap, matched``   -- planes exchanged, then each scaled by the other's
  signature length, so each objective keeps the strength it had and only its
  direction changes. The direction half of the swap.
* ``strength only``   -- no permutation, each plane scaled by the other's
  signature length. The magnitude half, with identity untouched -- a pure gain.

A network that keeps value in the direction of the colour code should flip on
``swap, matched`` and not on ``strength only``; one that keeps it in the length
should do the reverse. The DRC's fitted axis moves both (results/gain-vs-
factorised/colour-signature-arms.md), so the interesting number is how the flip
divides between them.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from goalmisgen.analysis.input_gain import input_kernel
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.provenance import header

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def permute_planes(
    observations: np.ndarray, channel_axis: int, gains: dict[int, float] | None = None, swap: bool = False
) -> np.ndarray:
    """Copy of ``observations`` with the two colour planes optionally exchanged, then scaled."""
    out = np.array(observations, dtype=np.float32, copy=True)
    moved = np.moveaxis(out, channel_axis, -1)
    if swap:
        a, b = COLOUR
        moved[..., [a, b]] = moved[..., [b, a]]
    for channel, gain in (gains or {}).items():
        moved[..., channel] *= gain
    return out


def main() -> None:
    args = parse_args()
    print(header())
    import jax

    from goalmisgen.analysis.behaviour import on_shortest_route
    from goalmisgen.axis import Greedy, load_base

    base = load_base(args.run, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed)
    checkpoint, values = base.checkpoint, base.values
    params = base.unravel(np.asarray(base.raveled, dtype=np.float32))

    kernel = input_kernel(base.params)
    whole = np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree_util.tree_leaves(base.params)])
    lengths = {c: float(np.linalg.norm(whole[kernel.plane(c)])) for c in COLOUR}
    ratio = lengths[COLOUR[0]] / lengths[COLOUR[1]]
    print(f"base   {checkpoint}, values {values}")
    print(
        f"input  {kernel.path} {kernel.shape}; |K0| {lengths[COLOUR[0]]:.3f}, |K1| {lengths[COLOUR[1]]:.3f}, "
        f"ratio {ratio:.3f}\n"
    )

    greedy = Greedy.of(base)
    out: dict = {
        "run": str(args.run),
        "checkpoint": str(checkpoint),
        "values": list(values),
        "lengths": {str(k): v for k, v in lengths.items()},
        "ratio": ratio,
        "rows": [],
    }

    def measure(label: str, swap: bool, gains: dict[int, float] | None = None) -> None:
        found = greedy.threshold(
            params, args.episodes, args.seed, transform=partial(permute_planes, channel_axis=1, gains=gains, swap=swap)
        )
        point, low, high, outcomes = found.point, found.low, found.high, found.outcomes
        reached = found.reached
        route = on_shortest_route(outcomes)
        print(f"  {label:>28}{point:>9.2f}  [{low:6.1f},{high:6.1f}]{reached:>9.1%}{route:>9.1%}", flush=True)
        out["rows"].append(
            {
                "label": label,
                "swap": swap,
                "steps": float(point),
                "low": float(low),
                "high": float(high),
                "reached": float(reached),
                "on_route": float(route),
                "gains": {str(k): v for k, v in (gains or {}).items()},
            }
        )

    print(f"  {'':>28}{'steps':>9}  {'95% interval':^15}{'reached':>9}{'route':>9}")
    measure("base", swap=False)
    measure("swap", swap=True)
    # After the swap each objective sits in the other plane; scaling by the other
    # signature's length hands it back its own strength, leaving only direction swapped.
    measure("swap, matched strength", swap=True, gains={COLOUR[0]: 1 / ratio, COLOUR[1]: ratio})
    measure("strength only", swap=False, gains={COLOUR[0]: 1 / ratio, COLOUR[1]: ratio})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
