"""How a fitted value axis depends on how long its arms trained for.

Every quantity Table 1 reports -- the drift, the write and what is left over --
scales with arm length, and not at the same rate, so the shares at one budget are
a property of that budget as much as of the network. This module computes the
decomposition at one checkpoint of one sweep; the ladder script calls it at every
checkpoint the arms saved, and the table is the same numbers read down a column
of budgets.

The remainder is the vector it actually is, ``diff - drift - offset * axis``,
not the Pythagorean leftover ``|diff|^2 - |drift|^2 - |offset * axis|^2``. Least
squares orthogonalises in arm-index space, not in weight space, so the three
parts need not be orthogonal and the Pythagorean form can go negative. The
cross term is reported rather than assumed away, so the shares sum to one by
construction and the reader can see how far from orthogonal the parts were.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from goalmisgen.analysis.weights import fit_axis_and_drift, split_half_reliability


@dataclass(frozen=True)
class ArmShares:
    """One arm's diff, split three ways, as fractions of ``|diff|^2``."""

    offset: float
    norm: float
    drift: float
    write: float
    remainder: float
    cross: float
    """What the three squared parts leave over from one: zero when they are orthogonal."""


@dataclass(frozen=True)
class SweepDecomposition:
    """One sweep at one checkpoint, decomposed."""

    drift_norm: float
    axis_norm: float
    reliability: float
    """Split-half reliability of the axis, Spearman-Brown corrected; nan below four pairs."""
    arms: tuple[ArmShares, ...]

    def at(self, magnitude: float) -> tuple[ArmShares, ...]:
        """The arms at ``+magnitude`` and ``-magnitude``, the dose the table is read at."""
        return tuple(a for a in self.arms if abs(abs(a.offset) - magnitude) < 1e-6)

    def to_json(self) -> dict:
        return {
            "drift_norm": self.drift_norm,
            "axis_norm": self.axis_norm,
            "reliability": self.reliability,
            "arms": [asdict(a) for a in self.arms],
        }


def decompose(offsets: np.ndarray, diffs: np.ndarray, splits: int = 200, seed: int = 0) -> SweepDecomposition:
    """Fit ``diff = drift + offset * axis`` and split every arm against that fit.

    ``offsets`` are signed distances from the base value and ``diffs`` the
    matching weight diffs, one row each. The null arm, if present, is excluded
    from the fit exactly as ``014`` excludes it, so what it reports is the fit's
    out-of-sample account of pure drift.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    diffs = np.asarray(diffs)
    if offsets.ndim != 1 or diffs.ndim != 2 or len(offsets) != len(diffs):
        raise ValueError(f"need one offset per diff, got {offsets.shape} and {diffs.shape}")
    fitted = np.abs(offsets) > 1e-9
    if fitted.sum() < 3:
        raise ValueError("an axis needs at least three arms away from the base value")
    axis, drift = fit_axis_and_drift(offsets[fitted], diffs[fitted])
    drift_sq = float(drift @ drift)
    axis_sq = float(axis @ axis)
    arms = []
    for offset, diff in zip(offsets, diffs):
        diff = np.asarray(diff, dtype=np.float64)
        total = float(diff @ diff)
        if total < 1e-30:
            raise ValueError(f"the arm at offset {offset:+.2f} did not move")
        remainder = diff - drift - offset * axis
        parts = (drift_sq, offset * offset * axis_sq, float(remainder @ remainder))
        arms.append(
            ArmShares(
                offset=float(offset),
                norm=float(np.sqrt(total)),
                drift=parts[0] / total,
                write=parts[1] / total,
                remainder=parts[2] / total,
                cross=1.0 - sum(parts) / total,
            )
        )
    reliability = split_half_reliability(offsets[fitted], diffs[fitted], splits=splits, seed=seed)
    return SweepDecomposition(
        drift_norm=float(np.sqrt(drift_sq)),
        axis_norm=float(np.sqrt(axis_sq)),
        reliability=float(reliability),
        arms=tuple(arms),
    )


def table_row(decompositions: dict[str, SweepDecomposition], magnitude: float = 0.3) -> dict[str, float]:
    """Table 1's convention: shares at ``|offset| = magnitude``, averaged over every such arm of every sweep.

    Averaging shares (rather than sharing averages) keeps each arm weighted equally
    whatever its total movement, which is how the published table was read.
    """
    chosen = [arm for d in decompositions.values() for arm in d.at(magnitude)]
    if not chosen:
        raise ValueError(f"no arm sits at |offset| = {magnitude}")
    return {
        "drift": float(np.mean([a.drift for a in chosen])),
        "write": float(np.mean([a.write for a in chosen])),
        "remainder": float(np.mean([a.remainder for a in chosen])),
        "cross": float(np.mean([a.cross for a in chosen])),
        "arms": len(chosen),
    }



# --- reading the measurement tables back -----------------------------------
#
# ``014`` and ``arm_thresholds`` print for people, and the ladder is a few
# hundred of their tables read down a column. These parsers pull the numbers
# out of that prose so the appendix table is assembled from the files in
# ``results/`` rather than retyped. They match the line shapes, not positions,
# so a stray warning line between rows is skipped rather than fatal.

import re  # noqa: E402

_MEASURE = re.compile(
    r"^\s*(?P<label>.+?)\s{2,}(?P<point>-?\d+\.\d)\s+\[\s*(?P<low>-?\d+\.\d),\s*(?P<high>-?\d+\.\d)\]\s+(?P<reached>\d+\.\d)%\s*$"
)
_VALUE = re.compile(r"^v=(?P<value>\d+\.\d+) (?P<kind>fine-tuned|written \(grid\)|written \(unseen\)|written, held out)$")
_ARM_CP = re.compile(r"^\s*v=(?P<value>\d+\.\d+)\s+cp_(?P<steps>\d+)\s+\(update \d+\)\s+\|delta\| (?P<norm>[\d.]+)")
_IMPLIED = re.compile(r"^\s*(?P<value>\d+\.\d+)\s+(?P<offset>-?\d+\.\d+)\s+[\d.]+\s+[\d.]+\s+(?P<r2>-?[\d.]+)\s+(?P<cos>-?[\d.]+)\s+(?P<implied>[+-]\d+\.\d+)\s*$")
_LOO = re.compile(r"^\s*(?P<value>\d+\.\d+)\s+(?P<arm>-?\d+\.\d)\s+(?P<written>-?\d+\.\d)\s+(?P<error>[+-]\d+\.\d)\s*$")


@dataclass(frozen=True)
class Measured:
    point: float
    low: float
    high: float
    reached: float


@dataclass
class MeasureTable:
    """What one ``014`` (or ``arm_thresholds``) table said, by value."""

    read_at: int | None = None
    """Steps the arms had trained for when read, from their checkpoint names."""
    base: Measured | None = None
    fine_tuned: dict[float, Measured] | None = None
    written: dict[float, Measured] | None = None
    unseen: dict[float, Measured] | None = None
    random: dict[float, Measured] | None = None
    implied: dict[float, float] | None = None
    """LOO read-back of each arm's trained offset, keyed by trained offset."""
    loo: dict[float, tuple[float, float]] | None = None
    """Leave-one-out (fine-tuned, written) per value, from a ``--leave-one-out`` run."""

    def __post_init__(self) -> None:
        for name in ("fine_tuned", "written", "unseen", "random", "implied", "loo"):
            if getattr(self, name) is None:
                setattr(self, name, {})


def parse_measure_table(text: str) -> MeasureTable:
    """Read a ``014`` / ``arm_thresholds`` / ``--leave-one-out`` table."""
    table = MeasureTable()
    steps: set[int] = set()
    in_loo = False
    for line in text.splitlines():
        if "writable out of sample" in line:
            in_loo = True
        arm = _ARM_CP.match(line)
        if arm:
            steps.add(int(arm.group("steps")))
            continue
        implied = _IMPLIED.match(line)
        if implied:
            table.implied[round(float(implied.group("offset")), 6)] = float(implied.group("implied"))
            continue
        if in_loo:
            loo = _LOO.match(line)
            if loo:
                table.loo[float(loo.group("value"))] = (float(loo.group("arm")), float(loo.group("written")))
                continue
        found = _MEASURE.match(line)
        if not found:
            continue
        label = found.group("label").strip()
        measured = Measured(
            float(found.group("point")), float(found.group("low")), float(found.group("high")), float(found.group("reached")) / 100
        )
        if label == "base, untouched":
            table.base = measured
            continue
        random = re.match(r"^random, matched to (\d+\.\d)$", label)
        if random:
            table.random[float(random.group(1))] = measured
            continue
        by_value = _VALUE.match(label)
        if not by_value:
            continue
        value, kind = float(by_value.group("value")), by_value.group("kind")
        if kind == "fine-tuned":
            table.fine_tuned[value] = measured
        elif kind == "written (grid)":
            table.written[value] = measured
        elif kind == "written (unseen)":
            table.unseen[value] = measured
    if len(steps) > 1:
        raise ValueError(f"arms read at different budgets {sorted(steps)} in one table")
    table.read_at = steps.pop() if steps else None
    return table


def slope(points: dict[float, float]) -> float:
    """Least-squares steps per unit value through ``{value: threshold}``.

    The sign is the task's: raising an objective's value should *lower* the
    threshold for the other, so a well-behaved sweep gives a negative number for
    colour 1 and a positive one for colour 0. Callers compare against the task's
    own rate, +-20 steps per unit at a 0.05 step penalty.
    """
    if len(points) < 2:
        raise ValueError("a slope needs at least two points")
    x = np.array(sorted(points), dtype=np.float64)
    y = np.array([points[v] for v in sorted(points)], dtype=np.float64)
    x = x - x.mean()
    return float(x @ (y - y.mean()) / (x @ x))


def write_error(table: MeasureTable) -> tuple[float, float]:
    """Mean absolute and mean signed ``written - fine-tuned`` over the grid values both were measured at."""
    shared = sorted(set(table.written) & set(table.fine_tuned))
    if not shared:
        raise ValueError("no value has both a fine-tuned and a written threshold")
    gaps = np.array([table.written[v].point - table.fine_tuned[v].point for v in shared])
    return float(np.abs(gaps).mean()), float(gaps.mean())
