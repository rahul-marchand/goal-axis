"""Fine-tune onto a new objective value, and keep the weight change.

    uv run python experiments/013_value_axis.py CHECKPOINT --value 0.7 \
        --levels /workspace/data/levels/values/1.00-0.70@500k \
        --run-dir /workspace/data/runs/novalue11.s1234/arms/v070

``novalue11`` never saw a value channel. Its objectives were worth the constants
(1.0, 0.5), bound to colour, every single episode. So there is no per-episode
value for a probe to correlate against, and the whole correlational toolkit is
unavailable here by construction — which is the point. An agent trained this way
had no reason to represent "1.0" and "0.5" at all: it could have compiled the
comparison down into a threshold on the difference in distances and thrown the
values away.

Fine-tuning manufactures the variance that training withheld. Hold colour 0 at
1.0, retrain onto a new value ``v`` for colour 1, and the resulting change in
parameters is one sample of the map from value to weights. This script takes one
such sample; sweeping ``v`` and comparing the samples is what asks the real
question, which is not *where* the change lands but what *shape* the map has:

* if the diffs at different ``v`` are collinear, and scale with ``v``, then the
  agent holds what an objective is worth in something that behaves like a slot,
  and a direction in weight space can be fitted and written to
* if they are unrelated directions, each fine-tune rebuilt a threshold from
  scratch and there is no value quantity in there to find

Sparsity is deliberately not the criterion. An L0 or L1 penalty returns a sparse
diff whichever is true, so sparsity would measure the regulariser. Collinearity
across a grid cannot be produced that way.

Every arm must differ **only** in what the objectives pay: same seed, same
learning rate, same number of updates, and levels that are identical layouts
carrying different values (``FixedValues`` consumes no randomness, so a dataset
sweep generated at one seed has one set of mazes).

Two things about resuming are not obvious, and both are handled by
``goalmisgen.finetune.reset_checkpoint`` rather than by touching the training stack.
The machinery lives in :mod:`goalmisgen.finetune`, so ``scripts/train/frozen_arm.py``
and the tests import it instead of loading this file by its path.
"""

from __future__ import annotations

from goalmisgen.finetune import parse_args, run


def main() -> None:
    run(parse_args(description=__doc__))


if __name__ == "__main__":
    main()
