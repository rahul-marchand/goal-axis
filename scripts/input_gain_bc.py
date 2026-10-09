"""Is the route model's value axis a brightness knob? The offline twin of ``input_gain.py``.

    uv run python scripts/input_gain_bc.py /workspace/data/offline/runs/bcnv11.s1 \
        --demos /workspace/data/offline/demos/test.rho100 --json figures/data/input_gain/bcnv11.s1.json

Same rows, same JSON shape; the first linear layer is ``cell_in`` and the
observation is scaled before the greedy decode rather than inside an environment
step. See ``scripts/input_gain.py`` for what each row asks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.behaviour import on_shortest_route
from goalmisgen.analysis.input_gain import (
    gain_coefficients,
    gain_part,
    input_kernel,
    kernel_cosines,
    scale_planes,
)
from goalmisgen.analysis.layers import module_groups, restrict
from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.offline.axis import load_base, load_sweep
from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.provenance import header

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.3, 0.3])
    parser.add_argument("--gains", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0])
    parser.add_argument("--contrasts", type=float, nargs="+", default=[-0.5, -0.25, -0.1, 0.1, 0.25, 0.5])
    parser.add_argument("--no-axis", action="store_true")
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    base = load_base(args.run)
    kernel = input_kernel(base.params)
    print(f"base   {args.run.name} @ step {base.step:,}  ({base.flat.size:,} parameters), values {base.values}")
    print(f"input  {kernel.path} {kernel.shape}")

    demos = DemoSet.load(args.demos, hide_values=base.hide_values)
    indices = np.arange(min(args.levels, len(demos)))
    observations = demos.observations(indices)
    out: dict = {"run": str(args.run), "checkpoint": f"step {base.step}", "values": list(base.values), "kernel": kernel.path, "rows": []}

    def measure(flat: np.ndarray, label: str, gains: dict[int, float] | None = None, **tags) -> dict:
        params = base.unravel(np.asarray(flat, dtype=np.float32))
        seen = scale_planes(observations, gains, channel_axis=-1) if gains else observations
        summary, _, outcomes = evaluate(base.model, params, demos, indices, observations=seen)
        point, reached = float(summary.indifference), float(summary.behaviour.reached_objective)
        route = on_shortest_route(outcomes)
        print(f"  {label:>52}{point:>8.2f}{reached:>9.1%}{route:>9.1%}", flush=True)
        row = {"label": label, "steps": point, "reached": reached, "on_route": route,
               "gains": {str(k): v for k, v in (gains or {}).items()}, **tags}
        out["rows"].append(row)
        return row

    columns = f"  {'':>52}{'steps':>8}{'reached':>9}{'route':>9}"
    print(f"\n=== brightness: the untouched model, colour planes scaled at the input; {len(indices)} held-out levels ===\n")
    print(columns)
    measure(base.flat, "base, untouched", kind="base")
    for plane, channel in enumerate(COLOUR):
        print()
        for gain in args.gains:
            measure(base.flat, f"colour {plane} x{gain:g}", {channel: gain}, kind="brightness", plane=plane, gain=gain)
    print()
    for contrast in args.contrasts:
        gains = {COLOUR[0]: 1 + contrast, COLOUR[1]: 1 - contrast}
        measure(base.flat, f"colour 0 x{1 + contrast:g}, colour 1 x{1 - contrast:g}", gains, kind="contrast", contrast=contrast)

    if args.no_axis:
        finish(args, out)
        return

    offsets, diffs = load_sweep(base, args.sweep, args.steps, fit_window=args.fit_window, minimum=3)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    groups, _ = module_groups(base.params)
    # "embed" also holds the position and action embeddings; the input layer proper is cell_in.
    input_slice = np.zeros_like(axis)
    rows = np.arange(kernel.start, kernel.start + int(np.prod(kernel.shape)))
    input_slice[rows] = axis[rows]
    gain = gain_part(axis, base.flat, kernel, COLOUR)
    betas = gain_coefficients(axis, base.flat, kernel, COLOUR)
    cosines = kernel_cosines(axis, base.flat, kernel)
    embed = restrict(axis, groups, "embed")
    out.update(
        sweep=args.sweep,
        offsets_fitted=offsets.tolist(),
        axis_norm=float(np.linalg.norm(axis)),
        input_module="cell_in kernel",
        input_share=float(input_slice @ input_slice / (axis @ axis)),
        embed_share=float(embed @ embed / (axis @ axis)),
        gain_share_of_input=float(gain @ gain / (input_slice @ input_slice)),
        betas={str(k - FIRST_FEATURE_CHANNEL): v for k, v in betas.items()},
        kernel_cosines={str(k): v for k, v in cosines.items()},
    )
    print(f"\nsweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}")
    print(f"       cell_in kernel holds {100 * out['input_share']:.2f}% of ||axis||^2; its gain part {100 * out['gain_share_of_input']:.1f}% of that")
    print("       cos(axis, base kernel) per input channel: " + ", ".join(f"{c}: {v:+.2f}" for c, v in cosines.items()))
    print("       gain per unit offset: " + ", ".join(f"colour {k - FIRST_FEATURE_CHANNEL}: {v:+.3f}" for k, v in betas.items()))

    print(f"\n=== the {args.sweep} axis and its parts, written at the base values ===\n")
    print(columns)
    for offset in args.offsets:
        print()
        measure(base.flat + offset * axis, f"full axis @ {offset:+.2f}", kind="full", offset=offset)
        measure(base.flat + offset * input_slice, f"only cell_in kernel @ {offset:+.2f}", kind="input", offset=offset)
        measure(base.flat + offset * gain, f"gain part, in the weights @ {offset:+.2f}", kind="gain", offset=offset)
        equivalent = {channel: 1 + offset * beta for channel, beta in betas.items()}
        measure(base.flat, f"gain part, as brightness @ {offset:+.2f}", equivalent, kind="gain as brightness", offset=offset)
        measure(base.flat + offset * (input_slice - gain), f"cell_in minus gain @ {offset:+.2f}", kind="input minus gain", offset=offset)
        measure(base.flat + offset * (axis - gain), f"full axis minus gain @ {offset:+.2f}", kind="full minus gain", offset=offset)
    finish(args, out)


def finish(args: argparse.Namespace, out: dict) -> None:
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
