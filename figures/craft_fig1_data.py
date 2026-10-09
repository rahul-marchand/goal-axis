"""Data for the crafting task figure: one held-out field, the trained policy's own rollout on it, and the cost gaps.

    uv run python figures/craft_fig1_data.py RUN_DIR DEMOS --out figures/data/craftax/craft/fig1.npz
        [--checkpoint STEP] [--gap-range 8 11] [--candidates 128]

Chooses a field the expert sends to iron by a margin below the threshold (the
iron chain costs ``gap`` more actions than the coal chain, ``gap`` inside
``--gap-range``), on which the policy's closed-loop rollout reaches iron in the
expert's number of actions; among those, the one of median length. Records, for
that field, the tiles, both expert plans from the start, the policy's actions,
and the state (map, position, facing, inventory) and observation at every step,
so the figure is drawn from what the engine actually did. Also writes the cost
gap of every field in the demonstration set for the threshold histogram.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.craftax import closed_loop
from goalmisgen.craftax.craft import CraftDemoSet
from goalmisgen.offline.demonstrations import load_demonstrations
from goalmisgen.offline.train import list_checkpoints, load_checkpoint

IRON, COAL = 0, 1  # feature ids; kinds (IRON, COAL)


def objective_index(demos: CraftDemoSet, feature: int) -> np.ndarray:
    return (np.asarray(demos.feature_ids) == feature).argmax(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("demos", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--gap-range", type=int, nargs=2, default=(8, 11))
    parser.add_argument("--candidates", type=int, default=128)
    args = parser.parse_args()

    demos = load_demonstrations(args.demos, hide_values=True)
    task = demos.task
    rows = np.arange(len(demos))
    distances = np.asarray(demos.distances)
    cost_iron = distances[rows, objective_index(demos, IRON)]
    cost_coal = distances[rows, objective_index(demos, COAL)]
    gap = (cost_iron - cost_coal).astype(int)
    target_iron = np.asarray(demos.target) == objective_index(demos, IRON)
    values = sorted(demos.meta["values"], reverse=True)
    step_penalty, step_limit = float(demos.meta["step_penalty"]), int(demos.meta["step_limit"])
    threshold = (values[0] - values[1]) / step_penalty

    lo, hi = args.gap_range
    candidates = rows[target_iron & (gap >= lo) & (gap <= hi)][: args.candidates]
    print(f"{len(candidates)} candidate fields with gap in [{lo}, {hi}] sent to iron; threshold {threshold:.0f}")

    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.checkpoint is None else next(c for c in checkpoints if c[0] == args.checkpoint)
    model, params = load_checkpoint(directory)
    print(f"checkpoint step {step}")

    record: list = []
    decoded = closed_loop.rollout(model, params, demos, candidates, batch_size=len(candidates), record=record)
    outcomes = demos.replay_many(candidates, decoded.actions, [bool(e) for e in decoded.emitted_eos])
    good = [
        k
        for k, (i, o) in enumerate(zip(candidates, outcomes))
        if o["reached_objective"] and o.get("reached_feature_id") == IRON and int(decoded.lengths[k]) == int(cost_iron[i])
    ]
    print(f"{len(good)} of {len(candidates)} rollouts reach iron in the expert's {'{}'} actions")
    if not good:
        raise SystemExit("no clean rollout among the candidates; widen --gap-range or --candidates")
    good.sort(key=lambda k: int(decoded.lengths[k]))
    k = good[len(good) // 2]
    index = int(candidates[k])
    field = demos.level(index)
    length = int(decoded.lengths[k])
    plans = task.plans(field)
    print(f"field {index}: iron plan {cost_iron[index]}, coal plan {cost_coal[index]}, gap {gap[index]}, rollout {length}")

    # Per-step state of the chosen row: record[t] holds arrays over the chunk.
    maps = np.stack([entry[1][k] for entry in record[: length + 1]])
    positions = np.stack([entry[2][k] for entry in record[: length + 1]])
    facings = np.stack([entry[3][k] for entry in record[: length + 1]])
    inventories = np.stack([entry[4][k] for entry in record[: length + 1]])
    feature_values = demos.feature_values([index])[0]
    observations = task.observe_batch(
        maps, positions, np.repeat(feature_values[None], len(maps), 0), demos.hide_values, facings, inventories
    )

    np.savez_compressed(
        args.out,
        tiles=task.tiles(field),
        agent_start=np.asarray(field.agent_start),
        start_direction=np.asarray(task.start_direction),
        objective_positions=np.asarray([o.position for o in field.objectives]),
        objective_features=np.asarray([o.feature_id for o in field.objectives]),
        objective_values=np.asarray([o.value for o in field.objectives]),
        plan_iron=np.asarray(plans[objective_index(demos, IRON)[index]].actions),
        plan_coal=np.asarray(plans[objective_index(demos, COAL)[index]].actions),
        actions=np.asarray(decoded.actions[k, :length]),
        maps=maps,
        positions=positions,
        facings=facings,
        inventories=inventories,
        observations=observations,
        gap=gap,
        target_iron=target_iron,
    )
    meta = {
        "run": str(args.run),
        "checkpoint": int(step),
        "demos": str(args.demos),
        "field": index,
        "kinds": list(task.kinds),
        "values": values,
        "step_penalty": step_penalty,
        "step_limit": step_limit,
        "threshold": threshold,
        "cost_iron": int(cost_iron[index]),
        "cost_coal": int(cost_coal[index]),
        "rollout_length": length,
        "n_fields": int(len(demos)),
        "expert_takes_iron": float(target_iron.mean()),
        "channels": ["walls", "agent", "trees", "stone", "table", "iron", "coal"]
        + ["facing up", "facing down", "facing left", "facing right", "wood", "stone held", "wood pickaxe", "stone pickaxe"],
        "n_candidates": int(len(candidates)),
        "n_clean": len(good),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
