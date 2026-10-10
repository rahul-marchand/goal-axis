"""Per-episode behavioural decode of a DRC agent, in the BC scripts' schema.

    uv run python scripts/studies/utility-rule/decode_h1_drc.py /workspace/data/runs/novalue11.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@1M --episodes 50000 --out out.npz

The gate question for taking the utility-rule programme to the DRC: is its
within-threshold noise even comparable to the route model's? bcnv11's part-1
numbers are a crossing near 10.3 and a pooled width of 4.4-5.9 steps; if the
DRC's threshold is far sharper there is little xi to explain on that side.

Saves the same five arrays the BC decodes use (``d_rich``, ``d_poor``,
``colour_of_rich``, ``reached``, ``reached_fid``), with one structural
difference documented rather than hidden: rows are *episodes*, drawn from the
split with replacement by the env's sampler, not the enumerated levels of a
DemoSet. Cells (exact distance pairs) are unaffected; per-level identity is
not available on this path.

The run's BASE.json names the checkpoint and the values, as everywhere else.
Levels default to the training dataset's test split - the same underlying
levels bcnv11's holdout demos were built from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from goalmisgen.paths import level_dataset
from goalmisgen.provenance import header
from goalmisgen.volume import read_base_marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run directory with BASE.json, or a checkpoint directory itself.")
    parser.add_argument("--levels", type=str, default=str(level_dataset(n_levels=1_000_000)))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="Evaluate at these objective values instead of the checkpoint's own - an arm trained at "
        "shifted values is decoded at the base values, as everywhere else in the project.",
    )
    return parser.parse_args()


def resolve(run: Path) -> tuple[Path, tuple[float, ...]]:
    if (run / "BASE.json").exists():
        marker = read_base_marker(run)
        return marker.checkpoint, marker.values
    if (run / "cfg.json").exists():  # a checkpoint directory was passed directly
        cfg = json.loads((run / "cfg.json").read_text())
        cfg = cfg.get("cfg", cfg)
        return run, tuple(cfg["train_env"]["objective_values"])
    raise SystemExit(f"{run} has neither BASE.json nor cfg.json")


def crossing(rate_by_gap: dict[int, float], level: float) -> float:
    xs = sorted(rate_by_gap)
    ys = [rate_by_gap[x] for x in xs]
    for i in range(len(xs) - 1):
        if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
            return xs[i] + (ys[i] - level) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i])
    return float("nan")


def main() -> None:
    args = parse_args()
    print(header())
    print()

    checkpoint, values = resolve(args.run)
    if args.values is not None:
        values = tuple(args.values)
    print(f"checkpoint {checkpoint}")
    print(f"values     {values}")

    from goalmisgen.analysis import summarise
    from goalmisgen.analysis.behaviour import decision_arrays
    from goalmisgen.axis import Greedy, checkpoint_eval_config, load_checkpoint

    # Colours by value rank, as the literal this replaced had them, even for a
    # colour-keyed arm read directly: at the base values the two schemes paint
    # the same, but they draw differently from the level sampler's generator.
    config = checkpoint_eval_config(
        checkpoint, args.levels, split=args.split, num_envs=args.num_envs, seed=args.seed, values=values, colour_keyed=False
    )
    base = load_checkpoint(checkpoint, config)
    greedy = Greedy.of(base)

    start = time.perf_counter()
    outcomes = greedy.outcomes(base.params, args.episodes, args.seed)
    print(f"rolled out {len(outcomes):,} episodes in {time.perf_counter() - start:.0f}s  (update {base.update})")
    summary = summarise(outcomes)
    print(f"reached {summary.reached_objective:.1%}  chose optimal {summary.chose_optimal:.1%}\n")

    arrays = decision_arrays(outcomes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"saved {args.out}  ({len(outcomes):,} episode rows; levels sampled with replacement)\n")

    # --- the gate numbers, read off binned rates as everywhere else ----------
    ok = arrays["reached"] & (arrays["d_rich"] >= 0) & (arrays["d_poor"] >= 0)
    gap = (arrays["d_rich"] - arrays["d_poor"])[ok]
    took = (arrays["reached_fid"] == arrays["colour_of_rich"])[ok].astype(float)
    rate = {int(g): took[gap == g].mean() for g in np.unique(gap) if (gap == g).sum() >= 25}
    theta = crossing(rate, 0.5)
    q25, q75 = crossing(rate, 0.75), crossing(rate, 0.25)
    print("threshold, read from the pooled curve (bcnv11 part 1: crossing 10.9-11.1, width 4.4-5.9):")
    print(f"  crossing {theta:.2f}   width q75-q25 {q75 - q25:.2f} steps   (q25 {q25:.2f}, q75 {q75:.2f})")
    print("  take-richer rate by gap:")
    for g in sorted(rate):
        if 4 <= g <= 18:
            print(f"    gap {g:>3}: {rate[g]:.3f}  (n {(gap == g).sum():,})")
    if not np.isfinite(theta):
        print("  no crossing found in range; inspect the rate table")
        sys.exit(1)


if __name__ == "__main__":
    main()
