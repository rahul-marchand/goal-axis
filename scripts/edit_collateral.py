"""Edits and frozen arms side by side: threshold reached, what else broke, and how much like the value axis.

    uv run python scripts/edit_collateral.py /workspace/data/runs/novalue11g1.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@500k \
        --deltas readout=figures/data/cheapest_edit/s1234.readout.npz input=... all=... \
        --axis-from all --arms frozen/o1-030=/workspace/data/runs/novalue11g1.s1234/arms-frozen/o1-030@400k \
        --arms free/o1-030=/workspace/data/runs/novalue11g1.s1234/arms/o1-030@400k ... \
        --json figures/data/gain_vs_factorised/e4/s1234.collateral.json

Experiment 4 of the gain-versus-factorised programme asks whether a value
change *can* live downstream of the input layer (an edit confined to the
readout, or an arm with the input layer frozen) and whether that version is
harmless. Every condition is measured on the same held-out levels through the
same JAX rollout:

* **threshold** at each dose, against its target ``tau_base + shift``;
* **navigation alone**: success, shortest-route rate and excess steps with only
  the richer, or only the poorer, objective left in each level
  (:mod:`goalmisgen.edits.collateral`);
* **choices away from the threshold**: accuracy on levels at least ``--margin``
  steps from the condition's own measured threshold;
* **likeness to the axis**: relative size of the edit per module,
  ``cos(u, v)`` in weight space (for frozen arms, also restricted to the
  coordinates they could move) and ``cos(J u, J v)`` in function space.

Searched deltas come from ``cheapest_edit.py --save-deltas``; the mean over
seeds is the edit, and ``v_step`` (the fine-tuned axis per step of threshold) is
read from the ``--axis-from`` file. Arms named ``frozen/...`` and ``free/...``
with matching suffixes are also fitted as axes and compared with each other.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from goalmisgen.provenance import header


def pairs(text: str) -> tuple[str, str]:
    name, _, path = text.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--n-eval", type=int, default=2048)
    parser.add_argument("--deltas", type=pairs, nargs="*", default=[])
    parser.add_argument("--axis-from", type=str, required=True, help="Name of the --deltas file holding v_step.")
    parser.add_argument("--arms", type=pairs, nargs="*", default=[], action="extend")
    parser.add_argument("--shifts", type=float, nargs="+", default=[-4.0, -2.0, 2.0, 4.0])
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--base-value", type=float, default=0.5, help="Base value of the swept objective (o1).")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()
    started = time.time()

    import jax.numpy as jnp

    from goalmisgen.analysis import closed_loop
    from goalmisgen.analysis.layers import module_groups
    from goalmisgen.analysis.route_targets import route_targets
    from goalmisgen.analysis.weights import cosine, fit_axis_and_drift
    from goalmisgen.axis import diff_from, load_base, route_levels
    from goalmisgen.edits import compare
    from goalmisgen.edits import rl as rl_edit
    from goalmisgen.edits.collateral import decisions_away_from, navigation, relative_change, single_objective

    # One environment and no seed: built only to load the weights.
    agent = load_base(args.run, args.levels, split="test", num_envs=1, seed=None)
    policy, unravel = agent.policy, agent.unravel
    base = agent.flat
    groups, order = module_groups(agent.params)
    levels = route_levels(agent.config, "test", args.n_eval)
    targets = route_targets(levels, size=11)
    singles = {keep: single_objective(levels, keep, size=11) for keep in ("rich", "poor")}
    rollout = closed_loop.make_rollout_fn(policy, unravel)
    logits_fn = rl_edit.make_logits_fn(policy, unravel)
    ones = jnp.ones(4)

    def run(flat, static, start):
        reached, steps = [], []
        for s in range(0, len(static), 1024):
            r, t = rollout(
                jnp.asarray(flat, dtype=jnp.float32), jnp.asarray(static[s : s + 1024]), jnp.asarray(start[s : s + 1024]), ones
            )
            reached.append(np.asarray(r))
            steps.append(np.asarray(t))
        return np.concatenate(reached), np.concatenate(steps)

    def measure(flat, target_tau=None) -> dict:
        reached, steps = run(flat, targets.static, targets.positions[:, 0, 0])
        r = closed_loop.read_outcomes(targets, reached, steps)
        row = {**r.as_row(), "away_own": decisions_away_from(targets.gap, targets.rich_feature, reached, r.tau, args.margin)}
        if target_tau is not None:
            row["target"] = target_tau
            row["away_target"] = decisions_away_from(targets.gap, targets.rich_feature, reached, target_tau, args.margin)
        for keep, single in singles.items():
            row[f"alone_{keep}"] = navigation(single, *run(flat, single.static, single.start))
        return row

    def show(label, row):
        target = f"{row['target']:>7.2f}" if "target" in row else "      -"
        print(
            f"  {label:<22}{target}{row['tau']:>7.2f} [{row['low']:5.1f},{row['high']:5.1f}]{row['reached']:>7.1%}{row['on_route']:>7.1%}"
            f"{row['away_own']['accuracy']:>8.1%}"
            f"{row['alone_rich']['success']:>7.1%}{row['alone_rich']['shortest']:>7.1%}"
            f"{row['alone_poor']['success']:>7.1%}{row['alone_poor']['shortest']:>7.1%}",
            flush=True,
        )

    columns = f"  {'condition':<22}{'target':>7}{'tau':>7} {'95% interval':<13}{'reach':>7}{'route':>7}{'away':>8}{'R ok':>7}{'R opt':>7}{'P ok':>7}{'P opt':>7}"
    legend = (
        f"  away = accuracy on levels >= {args.margin:g} steps from the condition's own threshold; "
        "R/P = only the richer/poorer objective left: success and shortest-route rate"
    )

    out: dict = {
        "run": str(args.run),
        "levels": args.levels,
        "n_eval": len(targets),
        "margin": args.margin,
        "shifts": args.shifts,
    }
    print("=== base ===")
    print(legend)
    print(columns)
    base_row = measure(base)
    show("base", base_row)
    out["base"] = base_row
    tau_base = base_row["tau"]

    # --- searched edits -----------------------------------------------------------------
    files = {name: np.load(path) for name, path in args.deltas}
    v_step = np.asarray(files[args.axis_from]["v_step"], dtype=np.float64)
    directions = {"v": v_step}
    for name, npz in files.items():
        seeds = [k for k in npz.files if k.startswith("seed")]
        directions[name] = np.mean([np.asarray(npz[k], dtype=np.float64) for k in seeds], axis=0)

    rows = np.random.default_rng(0).choice(len(targets), size=min(128, len(targets)), replace=False)
    arrays = {
        k: np.asarray(getattr(targets, k)[rows])
        for k in ("static", "positions", "toward_rich", "toward_poor", "role", "gap", "rich_above")
    }

    out["edits"] = {}
    print("\n=== searched edits, written at each dose (steps of threshold) ===")
    print(columns)
    for name, delta in directions.items():
        entry = {
            "norm_per_step": float(np.linalg.norm(delta)),
            "relative_per_step": relative_change(delta, base, groups, order),
            "weight_cos_v": float(cosine(delta, v_step)),
            "doses": [],
        }
        for shift in args.shifts:
            row = measure(base + shift * delta, tau_base + shift)
            row["shift"] = shift
            entry["doses"].append(row)
            show(f"{name} {shift:+.0f}", row)
        out["edits"][name] = entry

    # --- arms -------------------------------------------------------------------------------
    arm_diffs, arm_offsets = {}, {}
    out["arms"] = {}
    if args.arms:
        print("\n=== arms (the finished checkpoint, as trained) ===")
        print(columns)
    for name, path in args.arms:
        arm = Path(path)
        checkpoints = sorted(arm.glob("local-files/cp_*"), key=lambda p: int(re.sub(r"\D", "", p.name)))
        if not checkpoints:
            print(f"  {name}: no checkpoint in {arm}, skipped")
            continue
        diff = diff_from(agent, checkpoints[-1], subtract_in="float64")
        m = re.search(r"o1([+-])(\d{3})", arm.name)
        offset = (1 if m.group(1) == "+" else -1) * int(m.group(2)) / 100
        arm_diffs[name], arm_offsets[name] = diff, offset
        frozen_json = arm / "FROZEN.json"
        frozen = json.loads(frozen_json.read_text())["frozen"] if frozen_json.exists() else []
        movable = np.ones(len(base), dtype=bool)
        for module in frozen:
            movable[groups[module]] = False
        row = measure(base + diff)
        row.update(
            {
                "offset": offset,
                "checkpoint": str(checkpoints[-1]),
                "frozen": frozen,
                "frozen_norm": float(np.linalg.norm(diff[~movable])),
                "norm": float(np.linalg.norm(diff)),
                "relative": relative_change(diff, base, groups, order),
                "weight_cos_v": float(cosine(diff, v_step)),
                "weight_cos_v_movable": float(cosine(diff[movable], v_step[movable])),
                "v_share_in_frozen": float(v_step[~movable] @ v_step[~movable] / (v_step @ v_step)),
            }
        )
        show(name, row)
        out["arms"][name] = row
        directions[f"arm:{name}"] = diff

    # --- fitted axes from matched arm families --------------------------------------------
    out["fitted"] = {}
    for family in ("frozen", "free"):
        names = [n for n in arm_diffs if n.startswith(family + "/")]
        if len(names) < 3:
            continue
        offsets = np.array([arm_offsets[n] for n in names])
        axis, _ = fit_axis_and_drift(offsets, np.stack([arm_diffs[n] for n in names]))
        taus = np.array([out["arms"][n]["tau"] for n in names])
        slope = float(np.polyfit(offsets, taus, 1)[0])
        directions[f"axis:{family}"] = axis / slope if abs(slope) > 1e-9 else axis
        out["fitted"][family] = {
            "arms": names,
            "offsets": offsets.tolist(),
            "taus": taus.tolist(),
            "slope_steps_per_unit": slope,
            "weight_cos_v": float(cosine(axis, v_step)),
            "modules": compare.per_module(axis / np.linalg.norm(axis), v_step, groups, order),
        }
        print(
            f"\nfitted {family} axis from {len(names)} arms: threshold slope {slope:+.2f} steps per unit value, cos(axis, v) {out['fitted'][family]['weight_cos_v']:+.3f}"
        )
    if {"frozen", "free"} <= set(out["fitted"]):
        fa, ra = directions["axis:frozen"], directions["axis:free"]
        frozen = out["arms"][out["fitted"]["frozen"]["arms"][0]]["frozen"]
        movable = np.ones(len(base), dtype=bool)
        for module in frozen:
            movable[groups[module]] = False
        out["fitted"]["frozen_vs_free"] = {
            "cos": float(cosine(fa, ra)),
            "cos_movable": float(cosine(fa[movable], ra[movable])),
        }
        print(
            f"cos(frozen axis, free axis) {out['fitted']['frozen_vs_free']['cos']:+.3f}; on the coordinates both could move {out['fitted']['frozen_vs_free']['cos_movable']:+.3f}"
        )

    # --- function space ---------------------------------------------------------------------
    tangents = compare.function_tangents(logits_fn, base, arrays, directions)
    function = compare.cosine_matrix(tangents)
    out["function_cos_v"] = {n: function[n]["v"] for n in directions}
    out["function_norm_ratio_v"] = {n: float(np.linalg.norm(tangents[n]) / np.linalg.norm(tangents["v"])) for n in directions}
    print("\nfunction space (first-order logit change, centred), against v per step:")
    print(f"  {'direction':<26}{'cos(Ju,Jv)':>11}{'|Ju|/|Jv|':>11}")
    for n in directions:
        print(f"  {n:<26}{out['function_cos_v'][n]:>+11.3f}{out['function_norm_ratio_v'][n]:>11.3f}")

    out["seconds"] = time.time() - started
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
