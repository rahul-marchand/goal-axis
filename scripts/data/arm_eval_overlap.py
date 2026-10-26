"""IN.2: how many RL arm-evaluation levels also sit in some g1 arm's training split?

Eval set: the split 014 and arm_thresholds score on (``--split test`` of
``levels/values/1.00-0.50@500k``, as their g1 result headers record). Arm pools:
every ``level_dataset`` named in a g1 arm's ``cfg.json``, at the split it names.

Two notions of overlap:
  level   identical walls, size, agent start and objective positions
  layout  identical walls and size only (the maze)

Splits are what ``MazeConfig.sampler`` would use: the stored ``split_*.npy`` if
the pool has them, else ``dataset.split_indices`` at the config's holdout sizes.

    PYTHONPATH=<snapshot> python arm_eval_overlap.py --eval <pool> --eval-split test --arms-glob '<runs>/<base>/arms/*'
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from goalmisgen import provenance
from goalmisgen.envs.dataset import LevelDataset, split_indices


def find(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                yield v
            yield from find(v, key)


def arm_pools(arm_globs: list[str]) -> dict[tuple[str, str, int, int], list[str]]:
    pools = defaultdict(list)
    for pattern in arm_globs:
        for arm in sorted(glob.glob(pattern)):
            cfgs = sorted((Path(arm) / "local-files").glob("cp_*/cfg.json"))[::-1] + [
                p for p in (Path(arm) / "init" / "cfg.json", Path(arm) / "cfg.json") if p.is_file()
            ]
            if not cfgs:
                continue
            cfg = json.loads(cfgs[0].read_text())
            ds = next(find(cfg, "level_dataset"), None)
            if ds is None:
                continue
            split = next(find(cfg, "dataset_split"), "train")
            valid = next(find(cfg, "dataset_valid_levels"), 50_000)
            test = next(find(cfg, "dataset_test_levels"), 50_000)
            pools[(ds, split, valid, test)].append(arm)
    return pools


def load(path: str, split: str, valid: int, test: int):
    d = LevelDataset.load(path, expected_fingerprint=None, mmap=True)
    splits = d.stored_splits or split_indices(len(d), valid=valid, test=test)
    idx = np.asarray(splits[split])
    walls = np.asarray(d.walls_packed)
    sizes = np.asarray(d.sizes).reshape(len(walls), -1)
    agent = np.asarray(d.agent).reshape(len(walls), -1)
    pos = np.asarray(d.positions).reshape(len(walls), -1)
    layout = np.concatenate([walls.astype(np.int64), sizes.astype(np.int64)], axis=1)
    level = np.concatenate([layout, agent.astype(np.int64), pos.astype(np.int64)], axis=1)
    return dict(n=len(walls), idx=idx, stored=bool(d.stored_splits), layout=layout, level=level, splits=splits)


def keys(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--arms-glob", nargs="+", required=True)
    ap.add_argument("--also", nargs="*", default=[], help="pool=split pairs reported separately (e.g. a base's training pool); not in the arm union")
    args = ap.parse_args()
    print(provenance.header())
    print()

    ev = load(args.eval, args.eval_split, 50_000, 50_000)
    ev_level = keys(ev["level"][ev["idx"]])
    ev_layout = keys(ev["layout"][ev["idx"]])
    print(f"eval   {args.eval}  split={args.eval_split}  ({len(ev['idx']):,} of {ev['n']:,} levels; "
          f"splits {'stored' if ev['stored'] else 'recomputed index-wise'}; "
          f"{len(np.unique(ev_layout)):,} distinct layouts)")

    pools = arm_pools(args.arms_glob)
    print(f"arm pools: {len(pools)} (pool, split) pairs over {sum(len(v) for v in pools.values())} arms with a cfg.json\n")

    any_level = np.zeros(len(ev_level), bool)
    any_layout = np.zeros(len(ev_layout), bool)
    nested_note = []
    print(f"{'pool':<44} {'split':<6} {'arms':>4} {'n_split':>8} {'stored':>6} {'nested':>6} "
          f"{'level-overlap':>13} {'layout-overlap':>14}")
    for (path, split, valid, test), arms in sorted(pools.items()):
        p = load(path, split, valid, test)
        # Nesting: is this pool's content the first n levels of the eval pool?
        m = min(p["n"], ev["n"])
        nested = bool(np.array_equal(p["level"][:m], ev["level"][:m]))
        nested_note.append(nested)
        tr_level = np.unique(keys(p["level"][p["idx"]]))
        tr_layout = np.unique(keys(p["layout"][p["idx"]]))
        hit_level = np.isin(ev_level, tr_level)
        hit_layout = np.isin(ev_layout, tr_layout)
        any_level |= hit_level
        any_layout |= hit_layout
        name = Path(path).name
        print(f"{name:<44} {split:<6} {len(arms):>4} {len(p['idx']):>8,} {str(p['stored']):>6} {str(nested):>6} "
              f"{hit_level.mean():>13.4f} {hit_layout.mean():>14.4f}")

    for spec in args.also:
        path, _, split = spec.partition("=")
        p = load(path, split or "train", 50_000, 50_000)
        m = min(p["n"], ev["n"])
        nested = bool(np.array_equal(p["level"][:m], ev["level"][:m]))
        hit_level = np.isin(ev_level, np.unique(keys(p["level"][p["idx"]])))
        hit_layout = np.isin(ev_layout, np.unique(keys(p["layout"][p["idx"]])))
        print(f"[not an arm pool] {Path(path).name:<27} {split:<6} {'-':>4} {len(p['idx']):>8,} {str(p['stored']):>6} {str(nested):>6} "
              f"{hit_level.mean():>13.4f} {hit_layout.mean():>14.4f}")

    print()
    print(f"pools whose content is a prefix of the eval pool: {sum(nested_note)} of {len(nested_note)}")
    print(f"eval levels present (identical level) in ANY arm train split:  {int(any_level.sum()):,} / {len(any_level):,} = {any_level.mean():.4f}")
    print(f"eval levels whose LAYOUT is in ANY arm train split:            {int(any_layout.sum()):,} / {len(any_layout):,} = {any_layout.mean():.4f}")


if __name__ == "__main__":
    main()
