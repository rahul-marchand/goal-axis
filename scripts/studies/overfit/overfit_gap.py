"""How much of a route model's competence is memory: its own data against held-out data.

    uv run python scripts/studies/overfit/overfit_gap.py <run> \
        own=/workspace/data/offline/demos/overfit/1.00-0.50.n1k.train.rho100 \
        test=/workspace/data/offline/demos/test.rho100 [--json out.json]

The value-axis experiments ask what a base model's weights *encode*; this asks
what the same base has *learned*, which is the premise the overfitting study
turns into a variable. Two numbers per set, from the same final checkpoint:

``token accuracy``  teacher-forced, the quantity the run was trained on. On the
                    training set it says how completely the demonstrations were
                    absorbed; a memoriser reaches 1.0 there and stays low here.
``chose_optimal``   the greedy route replayed under the environment's rules -
                    whether the model takes the objective the task prefers.

The gap between the two sets is the whole point, so both are measured on the
same number of levels with the same decoder. Note that ``own`` on a fully
memorised set is not a claim about the utility rule: reproducing a stored route
gets the trade right on exactly the levels that were stored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.offline.axis import load_base
from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import cross_entropy, targets_from_routes, token_accuracy
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run directory; its last checkpoint is read.")
    parser.add_argument("sets", nargs="+", help="name=path demonstration sets to score.")
    parser.add_argument("--levels", type=int, default=2048, help="Levels scored per set.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch of the teacher-forced pass.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def teacher_forced(base, demos: DemoSet, indices: np.ndarray, batch_size: int) -> tuple[float, float]:
    """Mean cross-entropy and token accuracy of the expert's moves."""
    eos = base.model.config.eos

    @jax.jit
    def scores(params, observations, actions, lengths):
        targets = targets_from_routes(actions, lengths, eos)
        logits, _ = base.model.apply(params, observations, actions)
        mask = (targets >= 0).sum()
        return cross_entropy(logits, targets) * mask, token_accuracy(logits, targets) * mask, mask

    loss, correct, tokens = 0.0, 0.0, 0.0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batch_loss, batch_accuracy, batch_tokens = scores(
            base.params,
            jnp.asarray(demos.observations(chunk)),
            jnp.asarray(demos.routes(chunk)),
            jnp.asarray(np.asarray(demos.lengths[chunk], dtype=np.int32)),
        )
        loss += float(batch_loss)
        correct += float(batch_accuracy)
        tokens += float(batch_tokens)
    return loss / max(tokens, 1), correct / max(tokens, 1)


def main() -> None:
    args = parse_args()
    print(header())
    print()
    base = load_base(args.run)
    print(
        f"base {args.run.name} @ step {base.step:,}; {base.flat.size:,} parameters; "
        f"values {base.values}; values {'hidden' if base.hide_values else 'shown'}"
    )

    rows = {}
    print(
        f"\n  {'set':<14}{'levels':>8}{'loss':>9}{'token acc':>11}{'optimal':>9}{'matched':>9}{'reached':>9}{'legal':>8}{'indiff.':>9}"
    )
    for item in args.sets:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got {item!r}")
        demos = DemoSet.load(path, hide_values=base.hide_values)
        indices = np.arange(min(args.levels, len(demos)))
        loss, accuracy = teacher_forced(base, demos, indices, args.batch_size)
        summary, _, _ = evaluate(base.model, base.params, demos, indices)
        rows[name] = {
            "path": str(path),
            "levels": int(len(indices)),
            "loss": loss,
            "token_accuracy": accuracy,
            **{k: float(v) for k, v in summary.as_row().items()},
        }
        b = summary.behaviour
        print(
            f"  {name:<14}{len(indices):>8}{loss:>9.4f}{accuracy:>11.4f}{b.chose_optimal:>9.3f}"
            f"{summary.matched_expert:>9.3f}{b.reached_objective:>9.3f}{summary.legal:>8.3f}{summary.indifference:>9.2f}"
        )

    if len(rows) >= 2:
        names = list(rows)
        first, last = rows[names[0]], rows[names[-1]]
        print(
            f"\n  gap {names[0]} - {names[-1]}: token accuracy "
            f"{first['token_accuracy'] - last['token_accuracy']:+.4f}, "
            f"chose_optimal {first['chose_optimal'] - last['chose_optimal']:+.4f}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"run": str(args.run), "step": base.step, "sets": rows}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
