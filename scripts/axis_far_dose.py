"""Decode the o1 axis past its fitted window, against matched brightness, its parts and random writes.

    uv run python scripts/axis_far_dose.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k \
        --out /workspace/data/figures-data/axis-far-dose/novalue11g1.s1234

The data for "why is the written axis slow and fragile below parity?"
(:mod:`goalmisgen.analysis.far_dose`; tables from ``scripts/axis_far_dose_report.py``).
Conditions, all on the held-out split:

``axis``       base + dv * axis over ``--axis-offsets`` (in and far past the window)
``bright``     colour 1 alone at ``--bright-gains``
``coarse``     short decodes of colour 1, colour 0, and both planes along the axis's
               own input-conv gain ratio (log gains ``s * beta_c / beta_1``), to
               interpolate matches from
``matched``    for every axis write, the colour-1 gain, colour-0 gain and two-plane
               ``s`` with the same threshold
``part``       at ``--part-offsets``: the gain part in the weights (the plumbing
               check, identical to brightness ``1 + dv beta_c``), the input conv
               alone raw and rescaled to the axis's length, and the axis minus
               its gain part
``random``     at ``--part-offsets``: isotropic directions of norm ``|axis| dv`` and
               directions norm-matched to the axis module by module

Full decodes share seed and environment count, so rows are paired across
conditions exactly as in ``scripts/separability_decode.py``, whose ``.npz`` format
this reuses. Every default is pinned here rather than read from another script,
so edits elsewhere cannot change a run mid-flight. An interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from goalmisgen.analysis.far_dose import interpolate_x, module_matched_direction, random_direction
from goalmisgen.analysis.input_gain import gain_coefficients, gain_part, input_kernel
from goalmisgen.analysis.layers import module_groups, restrict
from goalmisgen.analysis.weights import fit_axis_and_drift
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.provenance import header

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--axis-offsets", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--part-offsets", type=float, nargs="+", default=[0.3, 0.7, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--bright-gains", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--coarse-gains-1", type=float, nargs="+", default=[0.8, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0])
    parser.add_argument("--coarse-gains-0", type=float, nargs="+", default=[1.25, 0.8, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12])
    parser.add_argument(
        "--coarse-s",
        type=float,
        nargs="+",
        default=[-0.2, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5],
        help="Two-plane log gain of colour 1; colour 0 follows at beta_0 / beta_1 of it.",
    )
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument("--module-draws", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=16_384)
    parser.add_argument("--coarse-episodes", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes % args.num_envs or args.coarse_episodes % args.num_envs:
        sys.exit("episode counts must be multiples of --num-envs, or the per-slot level order breaks")
    print(header())
    print()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"conditions": {}}

    from goalmisgen.analysis import summarise
    from goalmisgen.analysis.behaviour import decision_arrays, indifference_point, on_shortest_route, value_distance_decisions
    from goalmisgen.axis import Greedy, load_base, load_sweep

    base = load_base(args.run, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed)
    checkpoint, values, base_flat = base.checkpoint, base.values, base.flat
    manifest.update(
        run=str(args.run),
        checkpoint=str(checkpoint),
        values=list(values),
        levels=args.levels,
        split=args.split,
        seed=args.seed,
        num_envs=args.num_envs,
        episodes=args.episodes,
        sweep=args.sweep,
        argv=sys.argv[1:],
    )
    print(f"base {checkpoint}, values {values}")
    greedy = Greedy.of(base)

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def decode(name: str, flat: np.ndarray, gains: dict[int, float] | None, episodes: int, **tags) -> dict:
        if name in manifest["conditions"] and (args.out / f"{name}.npz").exists():
            row = manifest["conditions"][name]
            print(f"  {name:>30}  (cached) tau {row['tau']:6.2f}", flush=True)
            return row
        start = time.perf_counter()
        outcomes = greedy.outcomes(base.params_of(flat), episodes, args.seed, gains)
        seconds = time.perf_counter() - start
        np.savez(args.out / f"{name}.npz", **decision_arrays(outcomes, steps=True, value_tie=True))
        gaps, took, _ = value_distance_decisions(outcomes)
        tau = indifference_point(gaps, took)
        row = {
            "name": name,
            "tau": float(tau),
            "episodes": episodes,
            "reached": float(summarise(outcomes).reached_objective),
            "on_route": float(on_shortest_route(outcomes)),
            "seconds": seconds,
            "gains": {str(k - FIRST_FEATURE_CHANNEL): g for k, g in (gains or {}).items()},
            **tags,
        }
        manifest["conditions"][name] = row
        save()
        print(
            f"  {name:>30}  tau {tau:6.2f}  reached {row['reached']:.1%}  route {row['on_route']:.1%}  {episodes / seconds:,.0f} eps/s",
            flush=True,
        )
        return row

    print("\n=== base ===")
    base_row = decode("base", base_flat, None, args.episodes, kind="base", offset=0.0)

    offsets, diffs = load_sweep(base, args.sweep, arm_steps=args.arm_steps, fit_window=args.fit_window)
    axis, _ = fit_axis_and_drift(offsets, diffs)
    del diffs
    kernel = input_kernel(base.params)
    groups, _ = module_groups(base.params)
    input_slice = restrict(axis, groups, "input conv")
    gain = gain_part(axis, base_flat, kernel, COLOUR)
    betas = gain_coefficients(axis, base_flat, kernel, COLOUR)
    beta0, beta1 = betas[COLOUR[0]], betas[COLOUR[1]]
    axis_norm = float(np.linalg.norm(axis))
    rescale = axis_norm / float(np.linalg.norm(input_slice))
    manifest.update(
        axis_norm=axis_norm,
        offsets_fitted=offsets.tolist(),
        betas={"0": beta0, "1": beta1},
        input_rescale=rescale,
        fit_window=args.fit_window,
    )
    save()
    print(
        f"|axis| {axis_norm:.3g}; gain per unit offset colour 0 {beta0:+.3f}, colour 1 {beta1:+.3f}; input conv rescale x{rescale:.2f}"
    )

    def two_plane(s: float) -> dict[int, float]:
        return {COLOUR[1]: float(np.exp(s)), COLOUR[0]: float(np.exp(s * beta0 / beta1))}

    print("\n=== axis writes ===")
    writes = [
        decode(f"axis{dv:+.2f}", base_flat + dv * axis, None, args.episodes, kind="axis", offset=dv)
        for dv in args.axis_offsets
    ]

    print("\n=== colour-1 brightness ===")
    for g in args.bright_gains:
        decode(f"bright-c1x{g:g}", base_flat, {COLOUR[1]: g}, args.episodes, kind="bright", plane=1, gain=g)

    print("\n=== coarse curves ===")
    coarse: dict[str, list[tuple[float, float]]] = {
        "c1": [(0.0, base_row["tau"])],
        "c0": [(0.0, base_row["tau"])],
        "two": [(0.0, base_row["tau"])],
    }
    for g in args.coarse_gains_1:
        coarse["c1"].append(
            (
                np.log(g),
                decode(f"coarse-c1x{g:g}", base_flat, {COLOUR[1]: g}, args.coarse_episodes, kind="coarse", plane=1, gain=g)[
                    "tau"
                ],
            )
        )
    for g in args.coarse_gains_0:
        coarse["c0"].append(
            (
                np.log(g),
                decode(f"coarse-c0x{g:g}", base_flat, {COLOUR[0]: g}, args.coarse_episodes, kind="coarse", plane=0, gain=g)[
                    "tau"
                ],
            )
        )
    for s in args.coarse_s:
        coarse["two"].append(
            (
                s,
                decode(f"coarse-two{s:+g}", base_flat, two_plane(s), args.coarse_episodes, kind="coarse", plane="two", s=s)[
                    "tau"
                ],
            )
        )

    print("\n=== brightness matched to each axis write ===")
    for write in writes:
        for family, make in (
            ("c1", lambda x: {COLOUR[1]: float(np.exp(x))}),
            ("c0", lambda x: {COLOUR[0]: float(np.exp(x))}),
            ("two", two_plane),
        ):
            x = interpolate_x(coarse[family], write["tau"])
            if x is None:
                print(f"  {write['name']}: {family} never reaches tau {write['tau']:.2f} on the coarse grid; skipped")
                continue
            decode(
                f"match{write['offset']:+.2f}-{family}",
                base_flat,
                make(x),
                args.episodes,
                kind="matched",
                family=family,
                log_gain=x,
                matches=write["name"],
                offset=write["offset"],
                target_tau=write["tau"],
            )

    print("\n=== parts ===")
    for dv in args.part_offsets:
        decode(f"gainw{dv:+.2f}", base_flat + dv * gain, None, args.episodes, kind="part", part="gain in weights", offset=dv)
        decode(
            f"gainb{dv:+.2f}",
            base_flat,
            {c: 1 + dv * b for c, b in betas.items()},
            args.episodes,
            kind="part",
            part="gain as brightness",
            offset=dv,
        )
        decode(f"input{dv:+.2f}", base_flat + dv * input_slice, None, args.episodes, kind="part", part="input conv", offset=dv)
        decode(
            f"inputx{dv:+.2f}",
            base_flat + dv * rescale * input_slice,
            None,
            args.episodes,
            kind="part",
            part="input conv rescaled",
            offset=dv,
        )
        decode(
            f"nogain{dv:+.2f}",
            base_flat + dv * (axis - gain),
            None,
            args.episodes,
            kind="part",
            part="axis minus gain",
            offset=dv,
        )

    print("\n=== random writes ===")
    for draw in range(args.random_draws):
        direction = random_direction(len(axis), axis_norm, np.random.default_rng(1000 + draw))
        for dv in args.part_offsets:
            decode(
                f"rand{draw}{dv:+.2f}",
                base_flat + dv * direction,
                None,
                args.episodes,
                kind="random",
                family="isotropic",
                draw=draw,
                offset=dv,
            )
    for draw in range(args.module_draws):
        direction = module_matched_direction(axis, groups, np.random.default_rng(2000 + draw))
        for dv in args.part_offsets:
            decode(
                f"modrand{draw}{dv:+.2f}",
                base_flat + dv * direction,
                None,
                args.episodes,
                kind="random",
                family="per-module",
                draw=draw,
                offset=dv,
            )
    print("\nFAR_DOSE_DONE")


if __name__ == "__main__":
    main()
