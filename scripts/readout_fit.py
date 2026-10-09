"""Is the decision a projection of the two fields at the agent's own cell?

``field_decomp.py`` compares the two objectives' fields by their length at the
agent's cell: ``ln|A_rich| - ln|A_poor|`` tracks the level's distance gap and
crosses zero near the measured threshold. A length is a crude stand-in for
whatever the network reads, so this script fits the direction instead. Per site,

    x = A_rich[agent cell] - A_poor[agent cell]        (K,) per level
    s = w . x

with ``w`` fitted on a training split to predict **the model's own choice** in a
closed-loop rollout, and every number reported on held-out levels. The
no-intercept fit is the one that matters: its boundary is ``w . A_rich = w .
A_poor``, so the gap at which it crosses zero is a prediction to be compared
with the behavioural threshold, not a free parameter. The with-intercept fit is
an upper bound on any linear readout at this cell.

Judged against four references: the norm readout, random unit directions, a
direction fitted on shuffled labels, and directions fitted at other sites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.field_decomp import (
    BOTH,
    decompose,
    erasure_statics,
    make_grid_fn,
)
from goalmisgen.analysis.gain_alignment import sites_of
from goalmisgen.analysis.readout_fit import (
    agent_vectors,
    auc,
    evaluate_readout,
    logistic_direction,
    norm_readout,
    random_directions,
    unit,
)
from goalmisgen.provenance import header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-levels", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=4, help="Teacher-forced steps read, from the start.")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--n-random", type=int, default=64)
    parser.add_argument("--offset", type=float, default=0.30, help="Axis dose, for the direction's cosine with it.")
    parser.add_argument("--sweep", type=str, default="o1")
    parser.add_argument("--arm-steps", type=int, default=400_000)
    parser.add_argument("--fit-window", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=512,
                        help="Levels per forward pass; the whole set at once exhausts the GPU at 4096.")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())

    from goalmisgen.analysis.closed_loop import make_rollout_fn, read_outcomes
    from goalmisgen.analysis.route_targets import route_targets, usable
    from goalmisgen.analysis.weights import fit_axis_and_drift
    from goalmisgen.axis import load_base, load_sweep

    base = load_base(args.run, args.levels, split=args.split, num_envs=2, seed=args.seed)
    checkpoint, values, config, policy, unravel = base.checkpoint, base.values, base.config, base.policy, base.unravel
    base_flat = np.asarray(base.raveled, dtype=np.float32)

    sampler = config.sampler()
    rng = np.random.default_rng(args.seed)
    levels = []
    while len(levels) < args.n_levels:
        level = sampler.sample(rng)
        if usable(level):
            levels.append(level)
    targets = route_targets(levels, size=11)
    print(f"base   {checkpoint}, values {values}")
    print(f"levels {len(targets)} from {args.levels} [{args.split}], {args.n_steps} teacher-forced steps")

    offsets, diffs = None, None
    axis = None
    try:
        offsets, diffs = load_sweep(base, args.sweep, arm_steps=args.arm_steps, fit_window=args.fit_window)
        fitted, _ = fit_axis_and_drift(offsets, diffs)
        axis = np.asarray(fitted, dtype=np.float32)
        del diffs
        print(f"sweep  {args.sweep}: {len(offsets)} arms; |axis| {np.linalg.norm(axis):.3g}")
    except Exception as exc:  # the cosine is a nicety; the fit is the experiment
        print(f"sweep  unavailable ({type(exc).__name__}: {exc}); skipping the axis cosine")

    # --- the model's own choice, closed loop, on the same levels ------------------
    statics = erasure_statics(targets)
    start = np.asarray(targets.positions[:, 0, 0], dtype=np.int32)
    rollout = make_rollout_fn(policy, unravel)
    ones = np.ones(statics.shape[-1], dtype=np.float32)
    reached, steps = rollout(base_flat, statics[BOTH], start, ones)
    reached, steps = np.asarray(reached), np.asarray(steps)
    outcome = read_outcomes(targets, reached, steps)
    took_rich = (reached == np.asarray(targets.rich_feature)).astype(float)
    finished = reached >= 0
    print(f"choice rollouts: tau {outcome.tau:.2f} [{outcome.low:.2f}, {outcome.high:.2f}], "
          f"reached {outcome.reached:.1%}, took rich {outcome.took_rich:.1%}\n")

    # --- the fields, on teacher-forced positions ---------------------------------
    sites = sites_of(policy.cfg)
    spatial = tuple(i for i, s in enumerate(sites) if s.group not in ("dense", "logits"))
    grids = make_grid_fn(policy, unravel, n_steps=args.n_steps, site_index=spatial)
    positions = np.asarray(targets.positions[:, 0], dtype=np.int32)

    def capture(flat: np.ndarray) -> list[np.ndarray]:
        """The last step's grids per site, in level chunks so the device holds one at a time."""
        parts: list[list[np.ndarray]] = []
        for begin in range(0, len(targets), args.chunk):
            window = slice(begin, begin + args.chunk)
            out = grids(flat, ones, statics[:, window], positions[window])
            parts.append([np.asarray(site[-1], dtype=np.float32) for site in out])
        return [np.concatenate([part[k] for part in parts], axis=1) for k in range(len(spatial))]

    captured = {"base": capture(base_flat)}
    if axis is not None:
        captured["axis +"] = capture(base_flat + args.offset * axis)
    print(f"  read {len(spatial)} sites at step {args.n_steps - 1}\n")
    order = rng.permutation(len(targets))
    n_train = int(args.train_frac * len(targets))
    train_rows, test_rows = order[:n_train], order[n_train:]

    report: dict = {"run": str(args.run), "checkpoint": str(checkpoint), "values": list(values),
                    "levels": len(targets), "n_steps": args.n_steps, "train_frac": args.train_frac,
                    "tau": float(outcome.tau), "tau_low": float(outcome.low), "tau_high": float(outcome.high),
                    "reached": float(outcome.reached), "sites": [sites[i].name for i in spatial], "rows": []}

    gap = np.asarray(targets.gap, dtype=np.float64)
    vectors, directions, usable_rows = {}, {}, {}
    for k, index in enumerate(spatial):
        a_rich, a_poor, _, _ = decompose(captured["base"][k])
        x_rich = agent_vectors(a_rich, start)
        x_poor = agent_vectors(a_poor, start)
        reaches = (np.linalg.norm(x_rich, axis=1) > 0) & (np.linalg.norm(x_poor, axis=1) > 0)
        vectors[index] = (x_rich, x_poor)
        usable_rows[index] = reaches & finished

    print("=== fitted direction vs the norm, held out ===")
    print("    crossing is the gap at which the readout is zero; the measured threshold is "
          f"{outcome.tau:.2f}\n")
    print(f"  {'site':>18} {'readout':>14} {'AUC':>6} {'corr gap':>9} {'crossing':>9} {'n':>6}")
    for index in spatial:
        x_rich, x_poor = vectors[index]
        good = usable_rows[index]
        train = np.array([r for r in train_rows if good[r]], dtype=int)
        test = np.array([r for r in test_rows if good[r]], dtype=int)
        if len(train) < 50 or len(test) < 50:
            print(f"  {sites[index].name:>18}   fields reach the agent on {int(good.sum())} levels; skipped")
            continue
        x = x_rich - x_poor
        norm_score = norm_readout(x_rich, x_poor)

        w, _ = logistic_direction(x[train], took_rich[train], ridge=args.ridge, intercept=False)
        w_b, bias = logistic_direction(x[train], took_rich[train], ridge=args.ridge, intercept=True)
        directions[index] = w

        candidates = {
            "fitted": x[test] @ w,
            "fitted +bias": x[test] @ w_b + bias,
            "norm": norm_score[test],
        }
        for name, score in candidates.items():
            row = evaluate_readout(score, took_rich[test], gap[test])
            print(f"  {sites[index].name:>18} {name:>14} {row['auc']:6.3f} {row['corr']:+9.3f} "
                  f"{row['crossing']:+9.2f} {row['n']:6d}")
            report["rows"].append({"analysis": "readout", "site": sites[index].name, "readout": name, **row})

        # controls: random directions, and a fit on shuffled labels
        floor = random_directions(args.n_random, x.shape[1], seed=args.seed)
        aucs = np.array([auc(x[test] @ v, took_rich[test]) for v in floor])
        crossings = np.array([evaluate_readout(x[test] @ v, took_rich[test], gap[test])["crossing"] for v in floor])
        shuffled = rng.permutation(took_rich[train])
        w_shuffled, _ = logistic_direction(x[train], shuffled, ridge=args.ridge, intercept=False)
        shuffled_row = evaluate_readout(x[test] @ w_shuffled, took_rich[test], gap[test])
        print(f"  {sites[index].name:>18} {'random x' + str(args.n_random):>14} "
              f"{np.nanmean(aucs):6.3f} {'':>9} {np.nanmedian(crossings):+9.2f} "
              f"  (best AUC {np.nanmax(np.maximum(aucs, 1 - aucs)):.3f})")
        print(f"  {sites[index].name:>18} {'shuffled fit':>14} {shuffled_row['auc']:6.3f} "
              f"{shuffled_row['corr']:+9.3f} {shuffled_row['crossing']:+9.2f} {shuffled_row['n']:6d}")
        report["rows"].append({"analysis": "control", "site": sites[index].name, "readout": "random",
                               "auc_mean": float(np.nanmean(aucs)),
                               "auc_best": float(np.nanmax(np.maximum(aucs, 1 - aucs))),
                               "crossing_median": float(np.nanmedian(crossings)),
                               "crossing_iqr": float(np.nanpercentile(crossings, 75) - np.nanpercentile(crossings, 25))})
        report["rows"].append({"analysis": "control", "site": sites[index].name, "readout": "shuffled",
                               **shuffled_row})

    # --- does one direction serve the whole recurrence? --------------------------
    if len(directions) >= 2:
        print("\n=== transfer: a direction fitted at one site, used at another (held-out AUC) ===\n")
        names = [sites[i].name for i in directions]
        print("  " + " " * 18 + "".join(f"{n[-12:]:>14}" for n in names))
        for fitted_index in directions:
            cells = []
            for used_index in directions:
                x_rich, x_poor = vectors[used_index]
                good = usable_rows[used_index]
                test = np.array([r for r in test_rows if good[r]], dtype=int)
                score = (x_rich - x_poor)[test] @ directions[fitted_index]
                cells.append(auc(score, took_rich[test]))
            print(f"  {sites[fitted_index].name:>18}" + "".join(f"{c:14.3f}" for c in cells))
            report["rows"].append({"analysis": "transfer", "fitted_at": sites[fitted_index].name,
                                   "used_at": names, "auc": [float(c) for c in cells]})

    # --- what is the direction? --------------------------------------------------
    if axis is not None and "axis +" in captured:
        print("\n=== the fitted direction against the axis's own change at that cell ===\n")
        print(f"  {'site':>18} {'cos(w, axis change)':>22} {'cos(w, mean A_rich)':>22}")
        for k, index in enumerate(spatial):
            if index not in directions:
                continue
            change = captured["axis +"][k][BOTH] - captured["base"][k][BOTH]
            change_at_cell = unit(agent_vectors(change, start)[usable_rows[index]].mean(0))
            a_rich = decompose(captured["base"][k])[0]
            mean_rich = unit(agent_vectors(a_rich, start)[usable_rows[index]].mean(0))
            w = directions[index]
            print(f"  {sites[index].name:>18} {float(w @ change_at_cell):22.3f} {float(w @ mean_rich):22.3f}")
            report["rows"].append({"analysis": "direction", "site": sites[index].name,
                                   "cos_axis_change": float(w @ change_at_cell),
                                   "cos_mean_rich": float(w @ mean_rich)})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    print("READOUT_FIT_DONE")


if __name__ == "__main__":
    main()
