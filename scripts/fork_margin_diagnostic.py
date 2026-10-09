"""What does a direction that moves the threshold do to the fork margins? A check on the edit objective.

    uv run python scripts/fork_margin_diagnostic.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k --deltas /workspace/data/cheapest_edit/TAG.npz

The cheapest-edit search stands in for "moves the threshold by shift" with a
teacher-forced loss on the logit margin at each level's fork. The fine-tuned
axis ``v`` moves the rollout threshold exactly as asked, so whatever it does to
those margins is what a correct objective must accept. For each direction in the
npz and each shift this prints, over the same levels:

* the mean margin change in steps of gap, and its spread across levels
  (a uniform threshold shift would be ``shift`` with no spread);
* how that change depends on the gap and on the base margin (saturation);
* how often the sign of the teacher-forced margin agrees with the choice the
  agent actually makes in a closed-loop rollout under the same parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goalmisgen.provenance import header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--shifts", type=float, nargs="+", default=[-4, -2, 2, 4])
    args = parser.parse_args()
    print(header())

    import jax
    import jax.numpy as jnp

    from goalmisgen.analysis.closed_loop import logistic_fit
    from goalmisgen.edits.backends import rl_backend
    from goalmisgen.edits.core import fork_margins

    backend = rl_backend(args.run, args.levels, n_edit=args.n, n_eval=16)
    flat = backend.base_flat
    targets = backend.edit_targets
    arrays = backend.edit_arrays
    margin_of = jax.jit(lambda f, b: fork_margins(backend.logits_fn(f, b), b["toward_rich"], b["toward_poor"], b["role"]))

    def margins(params):
        return np.concatenate([
            np.asarray(margin_of(jnp.asarray(params, dtype=jnp.float32), {k: jnp.asarray(v[i : i + 64]) for k, v in arrays.items()}))
            for i in range(0, len(targets), 64)
        ])

    def choices(params):
        reached, _ = backend.evaluate(params, targets)
        return reached >= 0, reached == targets.rich_feature

    gaps = targets.gap.astype(float)
    m0 = margins(flat)
    b, a = np.polyfit(gaps, m0, 1)
    slope = abs(b)
    done, rich = choices(flat)
    base_rich = rich.copy()
    print(f"\nbase: margin {a:+.3f} {b:+.3f} x gap; behavioural threshold {logistic_fit(gaps[done], rich[done].astype(float))[0]:.2f}; "
          f"sign agrees with rollout {np.mean((m0 > 0)[done] == rich[done]):.1%}")

    deltas = np.load(args.deltas)
    for name in deltas.files:
        direction = deltas[name].astype(np.float64)
        print(f"\n=== {name}  |d| {np.linalg.norm(direction):.4g} per step ===")
        print(f"  {'shift':>6}{'tau':>7}{'mean dm':>9}{'sd dm':>8}{'corr gap':>10}{'corr m0':>9}{'sign~rollout':>14}{'sign~shifted base':>19}"
              f"{'|m0|<1 dm':>11}{'|m0|>3 dm':>11}{'flips':>7}{'fork too':>9}{'to target':>10}")
        for shift in args.shifts:
            m = margins(flat + shift * direction)
            dm = (m - m0) / slope
            done, rich = choices(flat + shift * direction)
            tau = logistic_fit(gaps[done], rich[done].astype(float))[0]
            near, far = np.abs(m0) < 1, np.abs(m0) > 3
            flipped = done & (rich != base_rich)
            fork_flip = ((m > 0) != (m0 > 0))[flipped].mean() if flipped.any() else float("nan")
            # does the flip go the way the shift asks? later threshold -> more rich choices
            toward = (rich[flipped] == (shift > 0)).mean() if flipped.any() else float("nan")
            print(
                f"  {shift:>+6.1f}{tau:>7.2f}{dm.mean():>+9.2f}{dm.std():>8.2f}{np.corrcoef(dm, gaps)[0, 1]:>+10.2f}"
                f"{np.corrcoef(dm, m0)[0, 1]:>+9.2f}{np.mean((m > 0)[done] == rich[done]):>14.1%}"
                f"{np.mean((m > 0) == (m0 + slope * shift > 0)):>19.1%}{dm[near].mean():>+11.2f}{dm[far].mean():>+11.2f}"
                f"{int(flipped.sum()):>7}{fork_flip:>9.1%}{toward:>10.1%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
