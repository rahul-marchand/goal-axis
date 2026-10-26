"""Summarise the retargeting-cost pilot: one table, and the data the figures read.

    uv run python scripts/studies/retarget/retarget_report.py /workspace/data/offline/results/retarget \
        --table results/retarget/pilot.txt --data figures/src/retarget-pilot/data.json

Reads every ``<agent>/<arm>.json`` record under the root (``goalmisgen.retarget.Record``;
the pilot's flat records too; any coding mechanism, so SDL means its total bits). Per run: MDL (= SDL, since L* = 0 on these mazes),
EDL, the learning curve L(t) on T, the ceiling n L(0), and where the bits went
(levels whose choice the goal flips against those it leaves alone). Per agent:
each arm's SDL minus the null arm's with the same spec, which is the spec's
headline quantity. The arithmetic is :mod:`goalmisgen.retarget.metric`; this
script lays it out. Runs are grouped by their records' specs, not their file
names: the agent is the directory a run is filed under, the arm its file name.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from goalmisgen.offline.retarget import NULL_TAU
from goalmisgen.provenance import header
from goalmisgen.retarget import Optimiser, Record, diagnostics, metric

SUBSPACES = {
    "axis": "axis",
    "axis-heldout": "axis held out",
    "axis-drift": "axis + drift",
    "pca1": "PCA k=1",
    "pca2": "PCA k=2",
    "pca4": "PCA k=4",
    "random1": "random 1-D",
    "random4": "random 4-D",
}
"""Display names of the subspace learners (``make_subspace.py``'s files)."""

REFERENCE = {"ordering": 0, "seed": 0, "n_valid": 1024, "learning_rate": Optimiser.learning_rate}
"""The settings a "main" run has; a run differing in one of them is a replicate of main, named for it."""


def short_rate(rate: float) -> str:
    mantissa, exponent = f"{rate:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def variant_of(spec: dict) -> str:
    """``main`` for the reference settings; a restricted learner's name; otherwise the one stage that was varied.

    A requential run is named for its teacher, since that is its main choice.
    """
    coding = spec.get("coding", {"kind": "prequential"})
    if coding["kind"] != "prequential":
        teacher = coding.get("teacher", {})
        trains = teacher.get("trains", {})
        what = teacher.get("name") or ("" if trains.get("kind", "all") == "all" else trains.get("name", ""))
        return " ".join(part for part in (coding["kind"], teacher.get("kind", ""), what) if part)
    trains = spec["learner"]["trains"]
    if trains["kind"] == "tensor":
        return "dial only" if trains["name"] == "goal_embedding" else f"only {trains['name']}"
    if trains["kind"] == "subspace":
        return SUBSPACES.get(trains["name"], f"subspace {trains['name']}")
    parts = []
    if spec["ordering"] != REFERENCE["ordering"]:
        parts.append(f"ordering {spec['ordering']}")
    if spec["seed"] != REFERENCE["seed"]:
        parts.append(f"learner seed {spec['seed']}")
    if spec["n_valid"] != REFERENCE["n_valid"]:
        parts.append(f"|V| {spec['n_valid']}")
    rate = spec["learner"]["optimiser"]["learning_rate"]
    if not np.isclose(rate, REFERENCE["learning_rate"]):
        parts.append(f"lr {short_rate(rate)}")
    return ", ".join(parts) or "main"


def summarise(agent: str, arm: str, r: Record) -> dict:
    result = r.result
    bounds = result.bounds
    return {
        "agent": agent,
        "arm": arm,
        "variant": variant_of(r.spec),
        "tau": float(r.goal),
        "n": result.n,
        "bounds": bounds,
        "test_bits": result.test_bits.tolist(),
        "cumulative": metric.cumulative(result.example_bits, bounds),
        "cumulative_flipped": [float(result.example_bits[:t][r.changed[:t]].sum()) for t in bounds],
        "best_step": result.best_step.tolist() if hasattr(result, "best_step") else None,
        "steps_run": result.steps_run.tolist() if hasattr(result, "steps_run") else None,
        "mdl": result.total_bits,
        "edl": metric.edl(r),
        "ceiling": metric.ceiling(r),
        "never_learned": float(result.n * result.test_bits[-1]),
        "flipped_fraction": float(r.changed.mean()),
        "seconds": float(np.sum(result.seconds)),
        "block_bits": metric.block_sums(result.example_bits, bounds),
        "block_tokens": None if r.units is None else [int(x) for x in metric.block_sums(r.units, bounds)],
        "tokens_per_route": metric.units_per_example(r),
    }


def with_null_differences(row: dict, r: Record, null: Record | None) -> None:
    """The run's SDL and ceiling less its null arm's, per route (``*_raw``) and per token within each block.

    Routes shorten as tau falls (8.9 tokens at 0.5, 10.2 at the null, 11.8 at
    18.5) and every agent keeps some error per move, so subtracting the null's
    bits per *route* credits a short-route arm with residual error it never had
    the chance to make (THREADS.md RC.9).
    """
    if null is None:
        row.update(sdl_minus_null_raw=None, ceiling_minus_null_raw=None)
        row.update(sdl_minus_null=None, ceiling_minus_null=None, excess_cumulative=None)
        return
    row["sdl_minus_null_raw"] = metric.excess_over_null(r, null, "route")
    row["ceiling_minus_null_raw"] = metric.ceiling_over_null(r, null, "route")
    excess = metric.excess_by_block(r, null)
    if excess is None:
        row.update(sdl_minus_null=None, ceiling_minus_null=None, excess_cumulative=None)
        return
    row["excess_cumulative"] = [0.0, *np.cumsum(excess).tolist()]
    row["sdl_minus_null"] = float(sum(excess))
    row["ceiling_minus_null"] = metric.ceiling_over_null(r, null, "unit")


def with_oracle(row: dict, r: Record, root: pathlib.Path) -> None:
    """``oracle_bits``: what the agent pays on the run's own D with its dial set by oracle_dial.py.

    ``sdl_minus_oracle`` is then the cost of finding the dial. It needs no null
    arm, because the oracle encodes the very routes the run encodes.
    """
    path = root / row["agent"] / f"tau{row['tau']:g}.json"
    if not path.exists():
        return
    oracle = json.loads(path.read_text())
    by_example = np.asarray(oracle["train_bits_by_example"])
    row["oracle_bits"] = float(oracle["train_bits_total"])
    row["oracle_test_bits"] = float(oracle["test_bits"])
    row["oracle_setting"] = oracle["best"]
    row["oracle_cumulative"] = metric.cumulative(by_example, row["bounds"])
    row["sdl_minus_oracle"] = metric.excess_over_oracle(r, by_example)


def table(rows: list[dict]) -> str:
    lines = [
        f"{'agent':<20} {'arm':<26} {'tau':>5} {'flip':>6} {'L(0)':>7} {'L(n)':>7} "
        f"{'SDL':>9} {'EDL':>9} {'ceiling':>9} {'SDL-null':>9} {'raw':>9} {'min':>5}"
    ]
    for r in sorted(rows, key=lambda r: (r["agent"], r["variant"] != "main", r["variant"], r["tau"])):
        diff = "" if r["sdl_minus_null"] is None else f"{r['sdl_minus_null']:9.0f}"
        raw = "" if r["sdl_minus_null_raw"] is None else f"{r['sdl_minus_null_raw']:9.0f}"
        lines.append(
            f"{r['agent']:<20} {r['arm']:<26} {r['tau']:>5g} {r['flipped_fraction']:>6.1%} "
            f"{r['test_bits'][0]:>7.3f} {r['test_bits'][-1]:>7.3f} {r['mdl']:>9.0f} {r['edl']:>9.0f} "
            f"{r['ceiling']:>9.0f} {diff:>9} {raw:>9} {r['seconds'] / 60:>5.1f}"
        )
    return "\n".join(lines)


def fractions(rows: list[dict], records: dict) -> str:
    """Per agent and learner, the share of the behaviour change each goal paid for.

    ``paid`` is :func:`metric.fraction_paid` per route: the SDL over the null
    arm, against what the unchanged agent would pay over its null arm on the
    same routes. 0 is a free dial, 1 is never learning. ``L(n)`` is the test
    loss at the end of the stream, against the full-weight learner's (``main``)
    on the same goal, which says whether a restricted learner's plateau
    reproduces the routes as well; ``t90`` is the stream length at which the
    learner first covers 90% of its own final drop in test loss.
    """
    taus = sorted({r["tau"] for r in rows if r["tau"] != NULL_TAU})
    main = {(r["agent"], r["tau"]): r for r in rows if r["variant"] == "main"}
    lines = [
        "Fraction of the behaviour change paid, per route: (SDL - null) / (ceiling - null).",
        "Then L(n) - L(n) of the same agent's full-weight learner, bits per route, and t90 (see the docstring).",
        "",
        f"{'agent':<16} {'learner':<15} " + " ".join(f"{t:>7g}" for t in taus),
    ]
    keys = sorted({(r["agent"], r["variant"]) for r in rows}, key=lambda k: (k[0], k[1] != "main", k[1]))
    for agent, variant in keys:
        mine = {r["tau"]: r for r in rows if (r["agent"], r["variant"]) == (agent, variant)}
        if not any(t in mine for t in taus):
            continue

        def cell(tau, value):
            r = mine.get(tau)
            out = None if r is None else value(r)
            return f"{'-':>7}" if out is None else out

        def paid(r):
            if r["sdl_minus_null_raw"] is None or not r["ceiling_minus_null_raw"]:
                return None
            return f"{r['sdl_minus_null_raw'] / r['ceiling_minus_null_raw']:>7.1%}"

        def excess_loss(r):
            reference = main.get((agent, r["tau"]))
            return None if reference is None else f"{r['test_bits'][-1] - reference['test_bits'][-1]:>+7.3f}"

        def t90(r):
            reached = metric.t90(records[(r["agent"], r["arm"])])
            return f"{'flat':>7}" if reached is None else f"{reached:>7}"

        lines.append(f"{agent:<16} {variant:<15} " + " ".join(cell(t, paid) for t in taus))
        if variant != "main":
            lines.append(f"{'':<16} {'  L(n) - main':<15} " + " ".join(cell(t, excess_loss) for t in taus))
        lines.append(f"{'':<16} {'  t90':<15} " + " ".join(cell(t, t90) for t in taus))
    return "\n".join(lines)


LIGHT = {
    diagnostics.Status.OK: ".",
    diagnostics.Status.WARN: "!",
    diagnostics.Status.FAIL: "X",
    diagnostics.Status.SKIP: " ",
}


def checks(records: dict, nulls: dict, thresholds: diagnostics.Thresholds) -> tuple[str, dict]:
    """The warning lights: one row of run checks per run, the details of every one not green, and the study checks.

    Returns the text and each run's statuses by check name, for the figure data.
    """
    order = sorted(
        records,
        key=lambda key: (key[0], variant_of(records[key].spec) != "main", variant_of(records[key].spec), records[key].goal),
    )
    names = [check.__name__.removeprefix("check_") for check in diagnostics.RUN_CHECKS]
    lines = [
        "Checks (goalmisgen.retarget.diagnostics): . ok   ! warn   X fail   blank: could not run.",
        "",
        f"{'agent':<20} {'arm':<42} " + " ".join(f"{i + 1:>2}" for i in range(len(names))),
    ]
    details, statuses = [], {}
    for agent, arm in order:
        r = records[(agent, arm)]
        findings = diagnostics.run_checks(r, diagnostics.finish_line(r, metric.null_of(nulls, agent, r)), thresholds)
        statuses[(agent, arm)] = {name: f.status.value for name, f in zip(names, findings)}
        lines.append(f"{agent:<20} {arm:<42} " + " ".join(f"{LIGHT[f.status]:>2}" for f in findings))
        bad = [f for f in findings if f.status in (diagnostics.Status.WARN, diagnostics.Status.FAIL)]
        if bad:
            details.append(f"{agent} {arm}\n{diagnostics.render(bad)}")
    lines += ["", "  " + "   ".join(f"{i + 1} {name.replace('_', ' ')}" for i, name in enumerate(names)), ""]
    lines += details or ["every run check passed or could not run"]
    study = diagnostics.study_checks(((f"{agent}/{arm}", r) for (agent, arm), r in records.items()), NULL_TAU, thresholds)
    flagged = [f for f in study if f.status is not diagnostics.Status.OK]
    lines += [
        "",
        f"Study checks: {len(study) - len(flagged)} of {len(study)} ok",
        diagnostics.render(flagged) if flagged else "",
    ]
    return "\n".join(lines).rstrip(), statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--table", type=pathlib.Path, default=None)
    parser.add_argument("--data", type=pathlib.Path, default=None)
    parser.add_argument(
        "--oracle",
        type=pathlib.Path,
        default=None,
        help="oracle_dial.py's output root (<agent>/tau<tau>.json); adds each goal-vector run's oracle cost.",
    )
    parser.add_argument(
        "--competence",
        type=float,
        default=None,
        help="Bits per route the base may pay on its own goal (diagnostics.Thresholds.competence); unset reports it unjudged.",
    )
    args = parser.parse_args()

    records = {(path.parent.name, path.stem): Record.load(path) for path in sorted(args.root.glob("*/*.json"))}
    nulls = metric.null_arms(((agent, r) for (agent, _), r in records.items()), NULL_TAU)
    rows = []
    for (agent, arm), r in records.items():
        row = summarise(agent, arm, r)
        with_null_differences(row, r, metric.null_of(nulls, agent, r))
        if args.oracle:
            with_oracle(row, r, args.oracle)
        rows.append(row)
    lights, statuses = checks(records, nulls, diagnostics.Thresholds(competence=args.competence))
    for row in rows:
        row["checks"] = statuses[(row["agent"], row["arm"])]
    kinds = sorted({r.result.kind for r in records.values()})
    mechanisms = (
        ""
        if kinds == ["prequential"]
        else f"Coding mechanisms here: {', '.join(kinds)}. For requential runs SDL is the total code (cumulative\n"
        "teacher-student KL plus per-call overhead) over the coded samples, n is the number of samples, and\n"
        "L(0), L(n), EDL and ceiling read the student's curve on T: not comparable with prequential rows.\n\n"
    )
    text = (
        f"{header()}\n\n"
        f"{mechanisms}"
        "Retargeting cost of route models: block-prequential code, doubling blocks, refit from the base\n"
        "before every block, early-stopped on V. Bits per route; SDL = MDL since L* = 0 (perfect mazes,\n"
        "tied levels dropped). flip = share of D whose expert choice differs from the base rate's.\n"
        "ceiling = n L(0), never leaving the base. SDL-null = SDL minus the agent's null arm (tau 10)\n"
        "at the same replicate settings, compared per token within each block (routes shorten as tau\n"
        "falls); raw = the same per route. min = wall-clock minutes for the run.\n\n"
        f"{table(rows)}\n\n"
        f"{fractions(rows, records)}\n\n"
        f"{lights}\n"
    )
    print(text)
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(text)
    if args.data:
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
