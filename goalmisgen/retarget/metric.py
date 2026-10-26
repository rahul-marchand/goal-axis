"""What a study reports from records: references, normalisations, and the fraction paid.

A record's MDL is the bits a (base, learner) pays to encode D at the new goal.
On its own that mixes the cost of the *change* with the agent's residual error
at its own goal, so the pilot reads it against a reference:

- the **null arm**: the same spec at the null goal (:func:`comparison_key`
  pairs them). Subtracted per route, or per unit (token) within each block,
  since routes change length with the goal and residual error is paid per
  token. The two disagree wherever residual error is large (THREADS.md RC.9);
- the **oracle**: the agent's own best dial setting encoding the very same
  examples, where an agent has a known dial. No composition bias, and only for
  agents built with a dial.

The **ceiling** is ``n L(0)``, the bits of never leaving the base, and the
**fraction paid** is the cost over the null divided by the ceiling over the
null: 0 is a free dial, 1 learns nothing reusable. The ceiling, EDL and the
fraction paid scale T's curve by the examples of D transmitted, so they are
defined for prequential codes only and refuse any other
(:func:`require_prequential`).

**Competence** is a gate, not something the null subtracts away (RC.8): an
agent whose null arm codes its own goal badly is relearning the task, not
retargeting. :func:`competent` applies a threshold, which the spec has not
fixed yet.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from goalmisgen.retarget.spec import Record, comparison_key

Per = Literal["route", "unit"]


def block_sums(values: np.ndarray, bounds: list[int]) -> list[float]:
    values = np.asarray(values)
    return [float(values[a:b].sum()) for a, b in zip(bounds, bounds[1:])]


def cumulative(values: np.ndarray, bounds: list[int]) -> list[float]:
    """The running total of ``values`` at each block boundary, from 0 at ``t_0``."""
    values = np.asarray(values)
    return [float(values[:t].sum()) for t in bounds]


def require_prequential(r: Record, what: str) -> None:
    """Refuse ``what`` on a record whose code is not prequential.

    ``n L(0)``, ``n L(n)`` and everything built from them scale the curve on T
    by ``result.n``, the number of examples of D the code transmitted. Only a
    prequential code transmits D: a requential result's ``n`` is the number of
    *samples* coded (calls x batch: 64 x 4,096 in RC.14), and
    its bits are a teacher-student KL, not a code for D, so the product is no
    ceiling of anything it paid. Raises rather than returning a number that
    reads like the prequential one.
    """
    kind = getattr(r.result, "kind", "prequential")
    if kind != "prequential":
        raise ValueError(
            f"{what} is defined for prequential codes only (it scales L(t) on T by n, the examples of D "
            f"transmitted); this record's code is {kind!r}, whose n = {r.result.n} counts coded samples"
        )


def ceiling(r: Record) -> float:
    """``n L(0)``: what D costs if the learner never leaves the base. Prequential codes only."""
    require_prequential(r, "the ceiling n L(0)")
    return float(r.result.n * r.result.test_bits[0])


def edl(r: Record) -> float:
    """MDL less ``n L(n)``. Kept for the record: at the pilot's n, n times T's sampling error is about +-330 bits (RC.2).

    Prequential codes only (:func:`require_prequential`).
    """
    require_prequential(r, "EDL = MDL - n L(n)")
    return float(r.result.total_bits - r.result.n * r.result.test_bits[-1])


def units_per_example(r: Record) -> float | None:
    return None if r.units is None else float(np.mean(r.units))


# ----------------------------------------------------------------------
# References
# ----------------------------------------------------------------------


def null_arms(records: Iterable[tuple[str, Record]], null_goal) -> dict[tuple[str, str], Record]:
    """The null arm of each (group, comparison key). ``group`` is the agent a caller files runs under."""
    return {(group, comparison_key(r.spec)): r for group, r in records if r.goal == null_goal}


def null_of(nulls: dict, group: str, r: Record) -> Record | None:
    return nulls.get((group, comparison_key(r.spec)))


def excess_by_block(arm: Record, null: Record) -> list[float] | None:
    """Per block, the arm's bits less the null's bits per unit charged at the arm's unit count."""
    if arm.units is None or null.units is None:
        return None
    bounds = arm.result.bounds
    if null.result.bounds != bounds:
        raise ValueError("arm and null arm were coded in different blocks")
    return [
        a_bits - a_units * n_bits / n_units
        for a_bits, a_units, n_bits, n_units in zip(
            block_sums(arm.result.example_bits, bounds),
            block_sums(arm.units, bounds),
            block_sums(null.result.example_bits, bounds),
            block_sums(null.units, bounds),
        )
    ]


def excess_over_null(arm: Record, null: Record, per: Per = "route") -> float | None:
    """MDL less the null arm's: per route, or per unit within each block (``None`` without unit counts)."""
    if per == "route":
        return arm.result.total_bits - null.result.total_bits
    excess = excess_by_block(arm, null)
    return None if excess is None else float(sum(excess))


def ceiling_over_null(arm: Record, null: Record, per: Per = "route") -> float | None:
    """The ceiling less the null arm's: the size of the behaviour change, in bits.

    Per unit, the null's ``L(0)`` per unit is charged at the arm's units per
    example. Both are read as the pilot read them: ``L(0)`` on T, units per
    example on D.
    """
    if per == "route":
        return ceiling(arm) - ceiling(null)
    require_prequential(arm, "the ceiling over the null arm")
    require_prequential(null, "the ceiling over the null arm")
    arm_units, null_units = units_per_example(arm), units_per_example(null)
    if arm_units is None or null_units is None:
        return None
    return arm.result.n * (arm.result.test_bits[0] - null.result.test_bits[0] / null_units * arm_units)


def fraction_paid(arm: Record, null: Record, per: Per = "route") -> float | None:
    """(MDL - null) / (ceiling - null): 0 is a free dial, 1 learns nothing reusable. Prequential codes only.

    Numerator and denominator always use the same normalisation ``per``.
    """
    cost, change = excess_over_null(arm, null, per), ceiling_over_null(arm, null, per)
    return _ratio(cost, change)


ROW_FIELDS: dict[str, tuple[str, str]] = {
    "route": ("sdl_minus_null_raw", "ceiling_minus_null_raw"),
    "unit": ("sdl_minus_null", "ceiling_minus_null"),
}
"""The (cost, change) fields of a report row (``retarget_report.py --data``) under each normalisation."""


def row_fraction_paid(row: dict, per: Per = "route") -> float | None:
    """:func:`fraction_paid` read from a report row, numerator and denominator from the same normalisation.

    A per-token cost over a per-route change mixes two normalisations; plot_core.py
    once did exactly that. Looking both fields up by ``per`` makes it impossible.
    """
    cost, change = (row.get(name) for name in ROW_FIELDS[per])
    return _ratio(cost, change)


def _ratio(cost: float | None, change: float | None) -> float | None:
    if cost is None or not change:
        return None
    return cost / change


def excess_over_oracle(arm: Record, oracle_bits: np.ndarray) -> float:
    """MDL less the oracle's bits on the same D: the cost of finding the dial. ``oracle_bits`` per example of D."""
    require_prequential(arm, "the excess over the oracle")
    oracle_bits = np.asarray(oracle_bits)
    if len(oracle_bits) != arm.result.n:
        raise ValueError(f"the oracle coded {len(oracle_bits)} examples, the run {arm.result.n}")
    return float(arm.result.total_bits - oracle_bits.sum())


# ----------------------------------------------------------------------
# The gate, and the shape of the curve
# ----------------------------------------------------------------------


def competent(null: Record, max_bits: float) -> bool:
    """Whether the null arm's base codes its own goal on T within ``max_bits`` per example."""
    return float(null.result.test_bits[0]) <= max_bits


def t90(r: Record) -> int | None:
    """The stream length at which the learning curve on T first covers 90% of its drop; ``None`` if it never drops."""
    curve = np.asarray(r.result.test_bits)
    drop = curve[0] - curve[-1]
    if drop <= 0:
        return None
    return int(r.result.bounds[int(np.flatnonzero(curve <= curve[0] - 0.9 * drop)[0])])
