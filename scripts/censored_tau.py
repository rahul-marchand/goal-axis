"""Does the far-dose threshold curve keep its shape once timeouts stop being ignored?

    uv run python scripts/censored_tau.py /workspace/data/figures-data/axis-far-dose/novalue11g1.s1234

The threshold is fitted on episodes that reached an objective, because a timeout
reveals no choice. Far out along a dose that is a shrinking and non-random
subset -- a damaged network may time out precisely on the levels it dithers on --
so the flattening near zero could be selection rather than the write. Levels are
paired across conditions (same seed, same order, same levels), which the distance
arrays are checked against, so two corrections are available with no new decoding:

``common``   the threshold on the levels *every* condition of a family finished,
             so each dose is read on identical levels;
``base here`` the untouched network's own threshold on the levels that condition
             finished. If that stays put, the surviving levels are not a biased
             sample of the trade-off; if it moves, the subset itself is the story.

Reads only the ``.npz`` files :mod:`scripts.axis_far_dose` wrote, so it is CPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen.analysis.behaviour import indifference_point  # noqa: E402
from goalmisgen.envs.solver import UNREACHABLE  # noqa: E402
from goalmisgen.provenance import header  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="an axis-far-dose output directory with manifest.json")
    parser.add_argument("--families", nargs="*", default=None, help="only these families")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def family_of(row: dict) -> tuple[str, float]:
    """A condition's family and its position along it."""
    kind = row.get("kind", "?")
    if kind == "axis":
        return "axis", float(row["offset"])
    if kind == "bright":
        return "bright c1", float(row["gain"])
    if kind == "coarse":
        plane = row.get("plane")
        return f"coarse {'two-plane' if plane == 'two' else f'c{plane}'}", float(row.get("gain", row.get("s", 0.0)))
    if kind == "part":
        return f"part: {row['part']}", float(row["offset"])
    if kind == "random":
        return f"random {row.get('family', '')} {row.get('draw', 0)}", float(row["offset"])
    if kind == "matched":
        return f"matched {row.get('family', '')}", float(row["offset"])
    return kind, float(row.get("offset", 0.0))


def main() -> None:
    args = parse_args()
    print(header())
    print()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    conditions = manifest["conditions"]
    print(f"{args.directory}  ({len(conditions)} conditions, {manifest['episodes']:,} episodes each)")

    loaded: dict[str, dict] = {}
    for name in conditions:
        path = args.directory / f"{name}.npz"
        if path.exists():
            loaded[name] = dict(np.load(path))
    print(f"loaded {len(loaded)} episode files")

    reference = loaded["base"]
    # UNREACHABLE is -1, the sentinel value_distance_decisions drops along with value ties.
    finite = (reference["d_rich"] != UNREACHABLE) & (reference["d_poor"] != UNREACHABLE) & ~reference["value_tie"]
    gaps = (reference["d_rich"] - reference["d_poor"]).astype(float)
    # Only the full decodes share the base's episode count, and only they are level-paired
    # with it: a shorter run spreads its episodes over the same slots differently, so its
    # rows are a different set of levels rather than a prefix. The coarse curves, which are
    # the short ones, are interpolation grids rather than points on the dose ladder.
    dropped = []
    for name in list(loaded):
        data = loaded[name]
        if len(data["d_rich"]) != len(reference["d_rich"]):
            dropped.append(name)
            del loaded[name]
            continue
        if not np.array_equal(data["d_rich"], reference["d_rich"]) or not np.array_equal(data["d_poor"], reference["d_poor"]):
            sys.exit(f"{name}: same length as the base but different levels; pairing is broken")
    if dropped:
        print(f"skipped {len(dropped)} shorter decodes (not level-paired): "
              + ", ".join(sorted(dropped)[:4]) + (" ..." if len(dropped) > 4 else ""))

    def tau_on(data: dict, rows: np.ndarray) -> float:
        rows = rows & finite & data["reached"]
        if rows.sum() < 50:
            return float("nan")
        took = (data["reached_fid"] == data["colour_of_rich"]).astype(float)
        return indifference_point(gaps[rows], took[rows])

    families: dict[str, list[tuple[float, str]]] = {}
    for name, row in conditions.items():
        if name not in loaded or name == "base":
            continue
        family, position = family_of(row)
        families.setdefault(family, []).append((position, name))

    out = {"directory": str(args.directory), "families": {}}
    everything = np.ones(len(gaps), bool)
    base_tau = tau_on(reference, everything)
    print(f"\nbase threshold {base_tau:.2f} on {(finite & reference['reached']).sum():,} finished levels\n")
    for family, members in sorted(families.items()):
        if args.families and family not in args.families:
            continue
        members.sort()
        common = finite & reference["reached"]
        for _, name in members:
            common = common & loaded[name]["reached"]
        print(f"=== {family} ===  common set {common.sum():,} levels "
              f"({common.sum() / max(int(finite.sum()), 1):.1%} of the trade-off levels)")
        print(f"  {'at':>8}{'reached':>9}{'tau':>8}{'common':>9}{'base here':>11}{'mean gap':>10}{'common gap':>12}")
        base_common = tau_on(reference, common)
        rows = []
        for position, name in members:
            data = loaded[name]
            keep = finite & data["reached"]
            row = {
                "name": name, "at": position, "reached": float(data["reached"].mean()),
                "tau": tau_on(data, everything), "tau_common": tau_on(data, common),
                "base_tau_here": tau_on(reference, keep),
                "mean_gap": float(gaps[keep].mean()) if keep.any() else float("nan"),
                "mean_gap_common": float(gaps[common].mean()) if common.any() else float("nan"),
            }
            rows.append(row)
            print(f"  {position:>8.3g}{row['reached']:>9.1%}{row['tau']:>8.2f}{row['tau_common']:>9.2f}"
                  f"{row['base_tau_here']:>11.2f}{row['mean_gap']:>10.2f}{row['mean_gap_common']:>12.2f}")
        print(f"  {'base':>8}{reference['reached'].mean():>9.1%}{base_tau:>8.2f}{base_common:>9.2f}"
              f"{base_tau:>11.2f}{gaps[finite & reference['reached']].mean():>10.2f}{gaps[common].mean():>12.2f}\n")
        out["families"][family] = {"common_levels": int(common.sum()), "base_tau_common": base_common, "rows": rows}

    out["base_tau"] = base_tau
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float))
        print(f"wrote {args.json}")
    print("CENSORED_DONE")


if __name__ == "__main__":
    main()
