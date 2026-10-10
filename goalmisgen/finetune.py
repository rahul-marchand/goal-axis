"""Fine-tune a saved agent onto new objective values: one arm of a value sweep.

``experiments/013_value_axis.py`` is the entry point and says why arms exist;
this is its machinery, so that ``scripts/train/frozen_arm.py`` and the tests reach it
by import rather than by loading a numbered experiment from its path.

Every arm must differ **only** in what the objectives pay: same seed, same
learning rate, same number of updates, and levels that are identical layouts
carrying different values. So the arm's configuration is the base run's own --
its network, its discount, whether it saw a value channel -- read from the
checkpoint (:mod:`goalmisgen.configs.saved`), with the values swapped and the
schedule flattened (:func:`finetune_config`). Resuming from the base needs two
counters reset first (:func:`reset_checkpoint`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path

import cleanba.cleanba_impala
import farconf
from cleanba.cleanba_impala import WandbWriter, train
from cleanba.config import Args

from goalmisgen import provenance
from goalmisgen.configs import saved
from goalmisgen.configs.presets import maze_drc33
from goalmisgen.configs.writers import CsvWriter


def parse_args(argv: list[str] | None = None, description: str | None = None) -> argparse.Namespace:
    """013's command line. ``argv`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="The trained agent to fine-tune from.")
    parser.add_argument("--value", type=float, default=None, help="What colour 1 is worth.")
    parser.add_argument(
        "--value-zero",
        type=float,
        default=1.0,
        help="What colour 0 is worth. Sweeping this instead separates a value from a "
        "threshold: only the gap between the two drives behaviour, so a knob that is "
        "really the gap moves the same way whichever colour is changed, while a pair "
        "of value slots does not.",
    )
    parser.add_argument(
        "--objective-values",
        type=float,
        nargs="+",
        default=None,
        help="All objectives' values at once, overriding --value and --value-zero. Needed "
        "beyond two objectives, where the choice no longer reduces to a single difference.",
    )
    parser.add_argument(
        "--colour-keyed",
        action="store_true",
        help="Pin colour i to objective i for this arm's levels instead of to whichever "
        "objective is worth most. Only matters when the arm's values cross parity: without "
        "it the level generator repaints the richer objective as colour 0, so the arm learns "
        "a relabelled preference rather than a reversed one.",
    )
    parser.add_argument("--levels", type=str, required=True, help="Dataset generated at this objective value.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Full run directory, not a parent of one.")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Held constant. Annealing would make the diffs at different values incomparable, "
        "since each arm would sit at a different point on its own schedule.",
    )
    parser.add_argument(
        "--anneal-to",
        type=float,
        default=None,
        help="Anneal the rate down to this instead of holding it constant. Off for arms, "
        "where a constant rate is what makes their diffs comparable; on when this is used "
        "to carry a base agent further rather than to move its values, since a long run "
        "wants the schedule it would have had.",
    )
    parser.add_argument("--checkpoints", type=int, default=4, help="Evenly spaced, the last at the final update.")
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Why this run is being launched, written to NOTE.md beside its checkpoints. "
        "The configuration is saved anyway; the reason is not, and it is the part that "
        "cannot be reconstructed later from what is on disk.",
    )
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.objective_values is None and args.value is None:
        parser.error("give either --value, or --objective-values for more than two objectives")
    if args.objective_values is not None and len(args.objective_values) < 2:
        parser.error(f"--objective-values needs one value per objective, got {args.objective_values}")
    return args


def finetune_config(args: argparse.Namespace) -> Args:
    """The base run's configuration, with the value swapped and the schedule flattened."""
    values = tuple(args.objective_values) if args.objective_values else (args.value_zero, args.value)
    config = maze_drc33(
        feature_value_correlation=1.0,
        min_size=args.size,
        max_size=args.size,
        n_objectives=len(values),
        objective_values=values,
        hide_values=saved.hides_values(args.checkpoint),
        total_timesteps=args.steps,
        level_dataset=args.levels,
        seed=args.seed,
    )
    if args.colour_keyed:
        config.train_env = dataclasses.replace(config.train_env, colour_keyed_features=True)
    # The base's own drawing of a level: an arm of a legend base is a legend arm.
    config.train_env = dataclasses.replace(config.train_env, observation=saved.observation(args.checkpoint))
    gamma = saved.discount(args.checkpoint)
    if gamma is not None:
        config.loss = dataclasses.replace(config.loss, gamma=gamma)
    net = saved.trained_net(args.checkpoint)
    if net is not None:
        config.net = net

    # A fine-tune is short enough that an annealed rate would spend most of it
    # near zero, and each arm would anneal over its own run rather than sharing
    # one schedule. A constant rate is what makes the arms comparable.
    config.learning_rate = args.lr
    config.final_learning_rate = args.anneal_to if args.anneal_to is not None else args.lr
    config.anneal_lr = args.anneal_to is not None

    # Evaluation exists to track misgeneralisation across correlations during a
    # long run. Here the measurement is the weight diff, and the behavioural
    # readout is 006 on the finished checkpoint, so evaluation would only cost
    # rollout time. Dropping the environments also drops the one upstream path
    # that assumes Sokoban.
    config.eval_envs = {}
    config.eval_at_steps = checkpoint_updates(config, args.checkpoints)
    config.save_model = True
    return config


def local_batch_size(config: Args) -> int:
    return int(config.local_num_envs * config.num_steps * config.num_actor_threads * len(config.actor_device_ids))


def checkpoint_updates(config: Args, count: int) -> frozenset[int]:
    """Updates at which to save, the last being the final one.

    Checkpoints are written when ``learner_policy_version`` lands in this set, so
    the final update has to be a member or the run finishes with nothing saved.
    The intermediate ones are what let the analysis ask whether the diff's
    *direction* is settled long before its magnitude is.
    """
    total = config.total_timesteps // local_batch_size(config)
    return frozenset(total * i // count for i in range(1, count + 1))


def reset_checkpoint(base: Path, destination: Path, config: Args) -> Path:
    """A copy of ``base`` that ``train`` will resume from as though it were new.

    Two counters in the checkpoint would otherwise end the run before it starts.
    ``update_step`` is where the loop begins and ``learner_policy_version`` is
    where it stops, and the base checkpoint carries both from the end of a
    140M-step run — so a short fine-tune would iterate over an empty range and
    exit having done nothing. Both are reset here.

    The optimiser is also rebuilt from *this* file's config rather than from the
    caller's, which is how the constant learning rate actually takes effect.

    The ``model`` file is copied byte for byte instead of being re-serialised:
    loading sums the ConvLSTM fence kernel along its input axis, so a checkpoint
    written back out after a load no longer has the shape ``init_params`` builds,
    and the next load fails on it.
    """
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / "model", destination / "model")

    config = dataclasses.replace(config, learner_policy_version=0)
    payload = {"cfg": farconf.to_dict(config, Args), "update_step": 1}
    (destination / "cfg.json").write_text(json.dumps(payload, indent=2))
    return destination


def run(args: argparse.Namespace) -> None:
    """Train one arm as ``args`` describes it, saving into ``args.run_dir``."""
    print(provenance.header())

    config = finetune_config(args)
    updates = config.total_timesteps // local_batch_size(config)

    print(f"\nbase            {args.checkpoint}")
    print(f"objective values{config.train_env.objective_values}")
    print(f"levels          {args.levels}")
    print(f"steps           {config.total_timesteps:,}  ({updates:,} updates)")
    print(f"learning rate   {config.learning_rate:g}  constant")
    print(f"saving at       {sorted(config.eval_at_steps)}")
    print(f"run dir         {args.run_dir}")

    # The task's own exchange rate, which the agent is not told and has to be
    # driven to by reward alone. Printed here because it is what 006 measures
    # afterwards, and having it on the run's own log makes the two comparable
    # without going back to the config.
    values = config.train_env.objective_values
    best, second = sorted(values, reverse=True)[:2]
    print(
        f"\ntop two objectives worth {best:g} and {second:g}, so the richer is worth "
        f"{(best - second) / config.train_env.step_penalty:.1f} extra steps\n"
    )

    config.load_path = reset_checkpoint(args.checkpoint, args.run_dir / "init", config)
    config.base_run_dir = args.run_dir
    if args.note:
        (args.run_dir / "NOTE.md").write_text(args.note.strip() + "\n")

    if os.environ.get("WANDB_MODE") in ("disabled", "offline") or not os.environ.get("WANDB_API_KEY"):
        print("W&B unavailable; logging metrics to CSV instead\n")
        writer = CsvWriter(config, args.run_dir)
    else:
        writer = WandbWriter(config)

    cleanba.cleanba_impala.MUST_STOP_PROGRAM = False
    try:
        train(config, writer=writer)
    finally:
        if isinstance(writer, CsvWriter):
            writer.flush()
    print("FINETUNE COMPLETE")
