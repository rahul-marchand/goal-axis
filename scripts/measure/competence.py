"""Is an agent worth running an experiment around?

    uv run python scripts/measure/competence.py CHECKPOINT --levels DIR [--min-reached 95]

Exits non-zero when the agent fails the bar, so a long unattended script can
stop rather than spend hours sweeping around an agent whose numbers would mean
nothing.

The bar is deliberately about *reaching* rather than choosing well. An agent
that reaches an objective every time but chooses the wrong one has a goal worth
studying; an agent that wanders until the step limit has no goal to study, and
every downstream measurement — exchange rates, weight diffs — is then reading
noise. Choice quality is printed alongside for the log, not gated on.

Parse failures are not treated as failure. This measures the agent itself
rather than scraping another script's output, precisely so that "I could not
tell" cannot be confused with "the agent is bad".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from goalmisgen.analysis import summarise
from goalmisgen.axis import Greedy, checkpoint_eval_config, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-objectives", type=int, default=3, help="Must agree with --objective-values.")
    parser.add_argument("--objective-values", type=float, nargs="+", required=True)
    parser.add_argument("--size", type=int, default=None, help="Defaults to the checkpoint's own maze size.")
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-reached", type=float, default=95.0, help="Percent, below which this exits non-zero.")
    args = parser.parse_args()
    if len(args.objective_values) != args.n_objectives:
        parser.error(f"--objective-values has {len(args.objective_values)} entries but --n-objectives is {args.n_objectives}")
    return args


def main() -> None:
    args = parse_args()
    config = checkpoint_eval_config(
        args.checkpoint,
        args.levels,
        split=args.split,
        num_envs=args.num_envs,
        seed=args.seed,
        values=tuple(args.objective_values),
        size=args.size,
    )
    agent = load_checkpoint(args.checkpoint, config)
    update = agent.update
    summary = summarise(Greedy.of(agent).outcomes(agent.params, args.episodes, args.seed))
    print(f"checkpoint {args.checkpoint.name}  (update {update})")
    print(f"  reached        {summary.reached_objective:.1%}")
    print(f"  chose optimal  {summary.chose_optimal:.1%}")
    print(f"  followed f0    {summary.followed_feature_zero:.1%}")
    print(f"  ambiguous      {summary.ambiguous:.1%}")
    print(f"  return / steps {summary.mean_return:.3f} / {summary.mean_steps:.1f}")

    if summary.reached_objective * 100 < args.min_reached:
        print(f"\nBELOW BAR: reached {summary.reached_objective:.1%} against a required {args.min_reached:g}%")
        sys.exit(1)
    print(f"\nfit to sweep around: reached {summary.reached_objective:.1%}")


if __name__ == "__main__":
    main()
