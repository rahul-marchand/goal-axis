"""Is the value axis a brightness knob? Threshold against each colour plane's input gain.

    uv run python scripts/input_gain.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --json figures/data/input_gain/novalue11g1.s1234.json

The restricted writes put most of the DRC axis's effect in the input
convolution, almost all of it on the two colour planes, amplifying one plane's
kernel and attenuating the other's. A gain on a plane's kernel is exactly a
gain on that plane of the observation (:mod:`goalmisgen.analysis.input_gain`),
so the mechanism can be tested without a fitted direction at all:

``brightness``  the untouched network, each colour plane scaled on its own and
                both in opposition; the threshold read off held-out rollouts
``axis``        the o-sweep axis at +-offset, and four parts of it: the input
                layer's slice, the gain part of that slice (written into the
                weights, and separately as the brightness it equals -- the two
                must agree, which checks the plumbing), the slice minus its gain
                part, and the whole axis minus its gain part

If the gain part alone reproduces the input slice's shift, and the brightness
curve is a straight line through the base with the slope the gain coefficients
predict, the axis in the input layer *is* "how loudly each colour is seen". The
same curve on a base with no value axis is the control: any network whose choice
rises with a goal's input magnitude would show some version of it.

Bases without arms (``--no-axis``) get the brightness curves only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.input_gain import (
    gain_coefficients,
    gain_part,
    input_kernel,
    kernel_cosines,
)
from goalmisgen.analysis.layers import module_groups, restrict
from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.provenance import header

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)
"""Input channels of colour 0 (the 1.0 goal at the base values) and colour 1."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Agent directory holding BASE.json and arms/.")
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.3, 0.3])
    parser.add_argument("--gains", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0])
    parser.add_argument("--contrasts", type=float, nargs="+", default=[-0.5, -0.25, -0.1, 0.1, 0.25, 0.5])
    parser.add_argument("--no-axis", action="store_true", help="Brightness curves only; the base has no arms.")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    from goalmisgen.analysis.behaviour import on_shortest_route
    from goalmisgen.axis import Greedy, load_base, load_sweep

    base = load_base(args.run, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed)
    checkpoint, values, base_flat = base.checkpoint, base.values, base.flat
    kernel = input_kernel(base.params)
    print(f"base   {checkpoint}  ({base_flat.size:,} parameters), values {values}")
    print(f"input  {kernel.path} {kernel.shape}")

    out: dict = {
        "run": str(args.run),
        "checkpoint": str(checkpoint),
        "values": list(values),
        "kernel": kernel.path,
        "rows": [],
    }
    greedy = Greedy.of(base)

    def measure(flat: np.ndarray, label: str, gains: dict[int, float] | None = None, **tags) -> dict:
        found = greedy.threshold(base.params_of(flat), args.episodes, args.seed, gains=gains)
        point, low, high, reached = found.point, found.low, found.high, found.reached
        route = on_shortest_route(found.outcomes)
        print(f"  {label:>52}{point:>8.2f}  [{low:5.1f}, {high:5.1f}]{reached:>9.1%}{route:>9.1%}", flush=True)
        row = {
            "label": label,
            **found.as_row(),
            "on_route": float(route),
            "gains": {str(k): v for k, v in (gains or {}).items()},
            **tags,
        }
        out["rows"].append(row)
        return row

    columns = f"  {'':>52}{'steps':>8}  {'95% interval':^14}{'reached':>9}{'route':>9}"
    print(f"\n=== brightness: the untouched network, colour planes scaled at the input; {args.episodes} held-out levels ===\n")
    print(columns)
    measure(base_flat, "base, untouched", kind="base")
    for plane, channel in enumerate(COLOUR):
        print()
        for gain in args.gains:
            measure(base_flat, f"colour {plane} x{gain:g}", {channel: gain}, kind="brightness", plane=plane, gain=gain)
    print()
    for contrast in args.contrasts:
        gains = {COLOUR[0]: 1 + contrast, COLOUR[1]: 1 - contrast}
        measure(
            base_flat, f"colour 0 x{1 + contrast:g}, colour 1 x{1 - contrast:g}", gains, kind="contrast", contrast=contrast
        )

    if args.no_axis:
        finish(args, out)
        return

    offsets, diffs = load_sweep(base, args.sweep, arm_steps=args.arm_steps, fit_window=args.fit_window)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    groups, _ = module_groups(base.params)
    input_module = "input conv" if "input conv" in groups else "patch embed"
    input_slice = restrict(axis, groups, input_module)
    gain = gain_part(axis, base_flat, kernel, COLOUR)
    betas = gain_coefficients(axis, base_flat, kernel, COLOUR)
    cosines = kernel_cosines(axis, base_flat, kernel)
    out.update(
        sweep=args.sweep,
        offsets_fitted=offsets.tolist(),
        axis_norm=float(np.linalg.norm(axis)),
        input_module=input_module,
        input_share=float(input_slice @ input_slice / (axis @ axis)),
        gain_share_of_input=float(gain @ gain / (input_slice @ input_slice)),
        betas={str(k - FIRST_FEATURE_CHANNEL): v for k, v in betas.items()},
        kernel_cosines={str(k): v for k, v in cosines.items()},
    )
    print(f"\nsweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}")
    print(
        f"       {input_module} holds {100 * out['input_share']:.1f}% of ||axis||^2; its gain part {100 * out['gain_share_of_input']:.1f}% of that"
    )
    print("       cos(axis, base kernel) per input channel: " + ", ".join(f"{c}: {v:+.2f}" for c, v in cosines.items()))
    print(
        "       gain per unit offset: " + ", ".join(f"colour {k - FIRST_FEATURE_CHANNEL}: {v:+.3f}" for k, v in betas.items())
    )

    print(f"\n=== the {args.sweep} axis and its parts, written at the base values ===\n")
    print(columns)
    for offset in args.offsets:
        print()
        measure(base_flat + offset * axis, f"full axis @ {offset:+.2f}", kind="full", offset=offset)
        measure(base_flat + offset * input_slice, f"only {input_module} @ {offset:+.2f}", kind="input", offset=offset)
        measure(base_flat + offset * gain, f"gain part, in the weights @ {offset:+.2f}", kind="gain", offset=offset)
        equivalent = {channel: 1 + offset * beta for channel, beta in betas.items()}
        measure(base_flat, f"gain part, as brightness @ {offset:+.2f}", equivalent, kind="gain as brightness", offset=offset)
        measure(
            base_flat + offset * (input_slice - gain),
            f"{input_module} minus gain @ {offset:+.2f}",
            kind="input minus gain",
            offset=offset,
        )
        measure(
            base_flat + offset * (axis - gain), f"full axis minus gain @ {offset:+.2f}", kind="full minus gain", offset=offset
        )
    finish(args, out)


def finish(args: argparse.Namespace, out: dict) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
