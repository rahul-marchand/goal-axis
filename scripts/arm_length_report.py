"""The fine-tune-length appendix, assembled from the ladder's result tables.

    uv run python scripts/arm_length_report.py --agents novalue11g1.s1234 novalue11g1.s9012 novalue11g1.s3456e \
        --out figures/data/arm_length_ladder.json

Reads ``results/len-<agent>-*`` -- the ``014`` tables at every checkpoint of the
400k and 3M grids, their ``--leave-one-out`` twins, the colour-keyed ``k`` arm
thresholds, and the weight-space ladder JSON -- and prints one row per budget
per agent: Table 1's shares, the axis reliability and cosine, what the arms
learned (slope against the task's 20 steps per unit), how well the axis writes
them, how far it extrapolates, and whether the reversed-preference arms crossed
zero. Rows are keyed by the steps the arms had actually trained for, read from
their checkpoint names, so the 3M grid read at its first checkpoint sits next to
the 400k grid read at its last and the two can be checked against each other.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from goalmisgen.analysis.arm_length import slope, write_error
from goalmisgen.analysis.tables import MeasureTable, parse_measure_table

TASK_SLOPE = 20.0  # steps per unit value at a 0.05 step penalty
_NAME = re.compile(r"^len-(?P<agent>.+)-(?P<sweep>[ok][01])@(?P<budget>\d+[kM])-at(?P<index>\d+)(?P<loo>-loo)?\.txt$")
EXTRAPOLATED = {"o1": 1.4, "o0": 0.1}  # the furthest unseen write in each sweep
REVERSED = {"k1": 1.2, "k0": 0.3}  # the k arm asking for the strongest reversal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_tables(results: Path, agent: str) -> dict[tuple[str, str, int, bool], MeasureTable]:
    found = {}
    for path in sorted(results.glob(f"len-{agent}-*.txt")):
        match = _NAME.match(path.name)
        if match is None:
            continue
        table = parse_measure_table(path.read_text())
        # arm_thresholds tables name no checkpoints, so the k families carry no
        # read_at of their own; they are keyed by index and take the budget's.
        if table.read_at is None and match.group("sweep").startswith("o"):
            print(f"  {path.name}: no arm checkpoints found, skipped", file=sys.stderr)
            continue
        key = (match.group("sweep"), match.group("budget"), int(match.group("index")), bool(match.group("loo")))
        found[key] = table
    return found


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def rows_for(agent: str, results: Path) -> list[dict]:
    tables = load_tables(results, agent)
    weights_path = results / f"len-{agent}-weights.json"
    weights = {}
    if weights_path.is_file():
        for row in json.loads(weights_path.read_text())["rows"]:
            weights[(row["arm_steps"], row["index"])] = row
    budgets = {"400k": 400_000, "3M": 3_000_000}
    rows = []
    keys = sorted({(budgets[b], i) for (_, b, i, _) in tables} | set(weights))
    for steps, index in keys:
        tag = "400k" if steps == 400_000 else "3M"
        o = {s: tables.get((s, tag, index, False)) for s in ("o0", "o1")}
        loo = {s: tables.get((s, tag, index, True)) for s in ("o0", "o1")}
        k = {s: tables.get((s, tag, index, False)) for s in ("k0", "k1")}
        present = [t for t in o.values() if t is not None]
        read_at = present[0].read_at if present else weights.get((steps, index), {}).get("read_at")
        row = {"agent": agent, "arm_steps": steps, "index": index, "read_at": read_at}
        w = weights.get((steps, index))
        if w:
            row.update(
                drift=w["shares"]["drift"],
                write=w["shares"]["write"],
                remainder=w["shares"]["remainder"],
                reliability=mean_or_nan([w["sweeps"]["o0"]["reliability"], w["sweeps"]["o1"]["reliability"]]),
                cos01=w["cos_axis0_axis1"],
                drift_norm=mean_or_nan([w["sweeps"]["o0"]["drift_norm"], w["sweeps"]["o1"]["drift_norm"]]),
                axis_norm=mean_or_nan([w["sweeps"]["o0"]["axis_norm"], w["sweeps"]["o1"]["axis_norm"]]),
            )
        slopes, errors, signed, readback, floors, reached = [], [], [], [], {}, []
        for sweep, table in o.items():
            if table is None:
                continue
            sign = -1.0 if sweep == "o1" else 1.0
            slopes.append(sign * slope({v: m.point for v, m in table.fine_tuned.items()}))
            absolute, mean_signed = write_error(table)
            errors.append(absolute)
            signed.append(mean_signed)
            if table.implied:
                readback.append(slope(table.implied))
            far = table.unseen.get(EXTRAPOLATED[sweep])
            if far is not None:
                floors[sweep] = far.point
            reached.extend(m.reached for m in table.fine_tuned.values())
        row.update(
            base=mean_or_nan([t.base.point for t in present if t.base]),
            ft_slope=mean_or_nan(slopes),
            write_err=mean_or_nan(errors),
            write_bias=mean_or_nan(signed),
            readback=mean_or_nan(readback),
            floor_o1=floors.get("o1", float("nan")),
            floor_o0=floors.get("o0", float("nan")),
            reached=mean_or_nan(reached),
        )
        loo_errors = [abs(w_ - a) for t in loo.values() if t is not None for a, w_ in t.loo.values()]
        row["loo_err"] = mean_or_nan(loo_errors)
        # The 400k k-arms were measured by the crossing chain into results/g1-*;
        # they are the ladder's first rung, read at the grid's last checkpoint.
        if steps == 400_000 and index == 3:
            for sweep in ("k0", "k1"):
                path = results / f"g1-{agent}-{sweep}.txt"
                if k.get(sweep) is None and path.is_file():
                    k[sweep] = parse_measure_table(path.read_text())
        for sweep, table in k.items():
            if table is None:
                continue
            far = table.fine_tuned.get(REVERSED[sweep])
            row[f"tau_{sweep}"] = far.point if far else float("nan")
            row[f"tau_{sweep}_ci"] = [far.low, far.high] if far else None
        # The same doses written rather than trained, from the o tables' unseen rows.
        for sweep, dose, source in (("k1", 1.2, "o1"), ("k0", 0.3, "o0")):
            table = o.get(source)
            if table is not None and dose in table.unseen:
                row[f"written_{sweep}"] = table.unseen[dose].point
        rows.append(row)
    return rows


def fmt(value, width=7, digits=1, percent=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{'—':>{width}}"
    if percent:
        return f"{100 * value:>{width}.{digits}f}"
    return f"{value:>{width}.{digits}f}"


def main() -> None:
    args = parse_args()
    everything = []
    for agent in args.agents:
        rows = rows_for(agent, args.results)
        everything.extend(rows)
        print(f"\n=== {agent} ===  (task slope {TASK_SLOPE:.0f} steps per unit; shares at |dv| = 0.3)\n")
        print(
            f"{'grid':>5}{'read at':>11}{'drift%':>8}{'write%':>8}{'resid%':>8}{'rel':>6}{'cos01':>7}"
            f"{'ft slope':>10}{'readback':>10}{'write err':>11}{'loo err':>9}{'floor o1':>10}{'floor o0':>10}{'tau k1':>8}{'tau k0':>8}{'reach':>7}"
        )
        for row in rows:
            print(
                f"{('400k' if row['arm_steps'] == 400_000 else '3M'):>5}{row['read_at'] or 0:>11,}"
                f"{fmt(row.get('drift'), 8, 1, True)}{fmt(row.get('write'), 8, 1, True)}{fmt(row.get('remainder'), 8, 1, True)}"
                f"{fmt(row.get('reliability'), 6, 2)}{fmt(row.get('cos01'), 7, 2)}"
                f"{fmt(row.get('ft_slope'), 10, 1)}{fmt(row.get('readback'), 10, 2)}{fmt(row.get('write_err'), 11, 2)}{fmt(row.get('loo_err'), 9, 2)}"
                f"{fmt(row.get('floor_o1'), 10, 1)}{fmt(row.get('floor_o0'), 10, 1)}{fmt(row.get('tau_k1'), 8, 1)}{fmt(row.get('tau_k0'), 8, 1)}"
                f"{fmt(row.get('reached'), 7, 1, True)}"
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"task_slope": TASK_SLOPE, "rows": everything}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
