"""Pull the wide value-grid numbers out of the results files into figures/data.

    uv run python figures/extract_wide_axis.py

Companion to ``extract_value_axis.py``, which serves the original seven-point
grid and asserts exactly seven rows. That assertion is deliberate there and
wrong here: the wide grid has twenty-five arms a sweep, three seeds and two
objectives, and the count legitimately differs between files — a sweep cut short
by time is still worth plotting, which is why the design orders arms
widest-offset first.

So this asserts the thing that is still true rather than a row count: a file that
parses to nothing is a format change and fails loudly, and every series records
how many points it actually found so a thin panel cannot be mistaken for a
complete one.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run with plain python3 as well as uv

from goalmisgen.analysis import tables  # noqa: E402

RESULTS = ROOT / "results"
OUT = Path(__file__).parent / "data"

# results/wide-novalue11.s1234-o1-heldout.txt -> seed 1234, sweep o1.
# ``x`` sweeps are the one-sided arms past the flip, held out of every fit.
NAME = re.compile(r"^wide-(?P<agent>[\w.]+?)-(?P<sweep>[ox]\d)(?P<heldout>-heldout)?$")

KIND = {
    "fine-tuned": "trained",
    "written, held out": "written_heldout",
    "written (grid)": "written",
    "written (unseen)": "written_unseen",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    series: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    bases: dict[str, float] = {}
    parsed_files = 0

    for path in sorted(RESULTS.glob("wide-*.txt")):
        match = NAME.fullmatch(path.stem)
        if match is None:
            print(f"  skipping {path.name}: not a wide-sweep result")
            continue
        text = path.read_text()
        # "        v=0.05 written, held out    16.6  [ 16.2,  17.1]     100.0%"
        rows = tables.value_rows(text)
        if not rows:
            raise SystemExit(f"{path.name} parsed to no rows — has the table format changed?")
        parsed_files += 1

        agent, sweep = match.group("agent"), match.group("sweep")
        for row in rows:
            # A held-out file also reprints the fine-tuned rows; keep one copy.
            kind = KIND[row.kind]
            if kind == "trained" and match.group("heldout"):
                continue
            series[(agent, sweep)][kind].append(
                {"value": row.value, "steps": row.point, "lo": row.low, "hi": row.high, "reach": row.reach}
            )
        found = tables.base_row(text)
        if found:
            bases[agent] = found.point

    if not parsed_files:
        raise SystemExit("no wide-sweep results found — run scripts on the wide grid first")

    # cfg.loss.gamma of the agents the arms were fine-tuned from, inferred from
    # the family tag: g1 names the undiscounted retrain, everything earlier
    # trained at 0.995. These scripts see only results text, not the volume, so
    # the name is the one place the fact is available — and a mixed batch has
    # no single optimal curve, so it is refused rather than averaged.
    g1_flags = {agent: "g1" in agent for agent, _ in series}
    if len(set(g1_flags.values())) > 1:
        raise SystemExit(f"mixed discount families in one extraction: {g1_flags}; run them separately")
    payload = {
        "step_penalty": 0.05,
        "discount": 1.0 if any(g1_flags.values()) else 0.995,
        # What the *other* objective pays, per sweep: o1 sweeps colour 1 with
        # colour 0 held at 1.0, o0 sweeps colour 0 with colour 1 held at 0.5.
        "other_objective": {"o1": 1.0, "o0": 0.5},
        "base_exchange_rate": bases,
        "series": [
            {
                "agent": agent,
                "sweep": sweep,
                "n": {kind: len(points) for kind, points in kinds.items()},
                **{kind: sorted(points, key=lambda r: r["value"]) for kind, points in kinds.items()},
            }
            for (agent, sweep), kinds in sorted(series.items())
        ],
    }
    (OUT / "wide_value_axis.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {OUT / 'wide_value_axis.json'} from {parsed_files} file(s)")
    for entry in payload["series"]:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(entry["n"].items()))
        print(f"  {entry['agent']:<20} {entry['sweep']}  {counts}")


if __name__ == "__main__":
    main()
