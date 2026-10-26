"""Does the goal-vector agent really have its dial? Decode it at every goal vector.

    uv run python scripts/studies/retarget/goal_vector_check.py /workspace/data/offline/runs/bcgv11.s1 \
        --demos /workspace/data/offline/demos/overfit/holdout.rho100 --json <out.json>

The positive control is only a control if each goal vector really sets the
agent's exchange rate. For each trained goal k, the vector of goal k is copied
into the base goal's slot (so every decoder, which uses the base slot, runs the
agent at goal k), the agent is decoded greedily on held-out levels, and its
indifference point is read, which should sit at that goal's tau. Then the same
for points between neighbouring vectors, the interpolation the held-out arms
rely on. Also reports the vectors' principal components, to show whether the
25 goals lie on one curve.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from goalmisgen.offline.decode import evaluate
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.fast_decode import greedy_decode_cached
from goalmisgen.offline.goals import with_base_goal_vector
from goalmisgen.offline.train import list_checkpoints, load_checkpoint, load_run_config
from goalmisgen.provenance import header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--demos", type=pathlib.Path, required=True)
    parser.add_argument("--levels", type=int, default=2048)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args()
    print(header())

    model, params = load_checkpoint(list_checkpoints(args.run)[-1][1])
    goals = load_run_config(args.run)["demos"]["goals"]
    taus = np.array([g["tau"] for g in goals])
    demos = DemoSet.load(args.demos, hide_values=True)
    indices = np.arange(min(args.levels, len(demos)))
    table = np.asarray(params["params"]["goal_embedding"])

    def exchange_rate(vector):
        summary, _, _ = evaluate(
            model, with_base_goal_vector(params, model, vector), demos, indices, decoder=greedy_decode_cached
        )
        return float(summary.indifference), float(summary.behaviour.reached_objective)

    print(f"{'goal':>5} {'tau':>6} {'decoded':>8} {'reached':>8}")
    trained = []
    for k, tau in enumerate(taus):
        rate, reached = exchange_rate(table[k])
        trained.append({"goal": k, "tau": float(tau), "decoded": rate, "reached": reached})
        print(f"{k:>5} {tau:>6.2f} {rate:>8.2f} {reached:>8.3f}", flush=True)

    print("\nmidpoints between neighbouring goals (by tau)")
    order = np.argsort(taus)
    between = []
    for a, b in zip(order, order[1:]):
        rate, reached = exchange_rate((table[a] + table[b]) / 2)
        between.append({"taus": [float(taus[a]), float(taus[b])], "decoded": rate, "reached": reached})
        print(f"  {taus[a]:>5.2f}..{taus[b]:<5.2f} -> {rate:6.2f}  reached {reached:.3f}", flush=True)

    centred = table - table.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    explained = singular**2 / (singular**2).sum()
    projection = centred @ vt[:2].T
    print(f"\nvariance explained by the first three components: {', '.join(f'{e:.1%}' for e in explained[:3])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "run": str(args.run),
                    "trained": trained,
                    "midpoints": between,
                    "explained": explained.tolist(),
                    "projection": projection.tolist(),
                    "taus": taus.tolist(),
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
