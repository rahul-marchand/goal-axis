"""Reading ``014``-format measure tables back: one parser for every script that consumes them.

``014`` prints its thresholds as text -- ``  v=0.30 fine-tuned      12.3  [ 11.8,  12.9]      100.0%``
-- and so do the scripts that print "014's own table format so the figure's
parser reads them" (``arm_thresholds``, ``far_axis_fit``, ``bc_arm_thresholds``).
Four parsers grew around that format independently -- the arm-length report's,
the value-axis and wide-grid extractors', and figure 3's -- and had begun to
disagree: one read negative values and one did not, one wanted the line to end
at the reach column and the others did not care. This is the one they share.

A row is any line holding a label, a threshold to one decimal, a bracketed
interval and a reach percentage, in that order. Its label is kept verbatim, and
when it names a value -- ``v=<value> <kind>`` -- the value (signed) and the kind
are split out. What a consumer does with the rows -- which kinds it keeps, a
reach floor, the order it plots them in -- stays with the consumer.

Standard library only: the figure scripts import this outside the project's
environment.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

ROW = re.compile(
    r"^\s*(?P<label>\S.*?)\s+(?P<point>-?\d+\.\d)\s+\[\s*(?P<low>-?\d+\.\d),\s*(?P<high>-?\d+\.\d)\]\s+(?P<reach>\d+\.\d)%"
)
"""One measured line: label, threshold, ``[low, high]``, reach in percent. Trailing columns are ignored."""

_NUMBER_OR_NAN = r"(?:-?\d+\.\d|nan)"
NAN_ROW = re.compile(
    rf"^\s*(?P<label>\S.*?)\s+{_NUMBER_OR_NAN}\s+\[\s*{_NUMBER_OR_NAN},\s*{_NUMBER_OR_NAN}\]\s+(?:\d+\.\d|nan)%"
)
""":data:`ROW` with ``nan`` allowed in any number: a measured line :data:`ROW` cannot read.

A threshold is ``nan`` when the agent never chose both ways, and an interval
when the bootstrap was skipped; such lines are in the format but hold no
number, so the parser skips them. :func:`dropped_rows` names them, and the
readers here warn with the count, so a table missing a value says so.
"""


class DroppedRowsWarning(UserWarning):
    """A table held measured lines with a NaN threshold, interval or reach, which the parser skipped."""


def dropped_rows(text: str) -> list[str]:
    """The lines of ``text`` in the row format that :data:`ROW` skips because a number is ``nan``."""
    return [line.strip() for line in text.splitlines() if ROW.match(line) is None and NAN_ROW.match(line)]


def _warn_dropped(text: str) -> None:
    dropped = dropped_rows(text)
    if dropped:
        warnings.warn(
            f"{len(dropped)} measured row(s) with a NaN value skipped: " + "; ".join(dropped[:5]),
            DroppedRowsWarning,
            stacklevel=3,
        )


KINDS = ("fine-tuned", "written, held out", "written (grid)", "written (unseen)")
VALUE_LABEL = re.compile(r"^v=(?P<value>-?\d+\.\d+) (?P<kind>" + "|".join(re.escape(k) for k in KINDS) + r")$")
BASE_LABEL = "base, untouched"
RANDOM_LABEL = re.compile(r"^random, matched to (?P<magnitude>\d+\.\d)$")

ARM_CHECKPOINT = re.compile(r"^\s*v=(?P<value>\d+\.\d+)\s+cp_(?P<steps>\d+)\s+\(update \d+\)\s+\|delta\| (?P<norm>[\d.]+)")
"""014's arm listing, ``  v=0.30  cp_199680  (update 39)  |delta| 2.4``: which checkpoint each arm was read at."""
IMPLIED = re.compile(
    r"^\s*(?P<value>\d+\.\d+)\s+(?P<offset>-?\d+\.\d+)\s+[\d.]+\s+[\d.]+\s+(?P<r2>-?[\d.]+)\s+(?P<cos>-?[\d.]+)\s+(?P<implied>[+-]\d+\.\d+)\s*$"
)
"""014's graded-and-predictive table: each arm's offset read back through the axis fitted without it."""
LOO = re.compile(r"^\s*(?P<value>-?\d+\.\d+)\s+(?P<arm>-?\d+\.\d)\s+(?P<written>-?\d+\.\d)\s+(?P<error>[+-]\d+\.\d)\s*$")
"""``--leave-one-out``'s summary: value, fine-tuned, written, error."""
LOO_SECTION = "writable out of sample"


@dataclass(frozen=True)
class Row:
    """One measured line of a table."""

    label: str
    point: float
    low: float
    high: float
    reach: float
    """In percent, as printed."""
    value: float | None = None
    """The value a ``v=<value> <kind>`` label names; ``None`` for any other label."""
    kind: str | None = None

    @property
    def reached(self) -> float:
        """Reach as a fraction."""
        return self.reach / 100


def parse_row(line: str) -> Row | None:
    """The row on ``line``, or ``None`` if it holds none."""
    found = ROW.match(line)
    if found is None:
        return None
    label = found.group("label").strip()
    named = VALUE_LABEL.match(label)
    return Row(
        label=label,
        point=float(found.group("point")),
        low=float(found.group("low")),
        high=float(found.group("high")),
        reach=float(found.group("reach")),
        value=float(named.group("value")) if named else None,
        kind=named.group("kind") if named else None,
    )


def rows(text: str) -> list[Row]:
    """Every measured row in ``text``, in file order. Warns (:class:`DroppedRowsWarning`) for rows holding a NaN."""
    _warn_dropped(text)
    return [row for row in map(parse_row, text.splitlines()) if row is not None]


def value_rows(text: str, kinds: tuple[str, ...] = KINDS) -> list[Row]:
    """The rows naming a value, of the given kinds, in file order."""
    return [row for row in rows(text) if row.kind in kinds]


def base_row(text: str) -> Row | None:
    """The first ``base, untouched`` row, or ``None``."""
    return next((row for row in rows(text) if row.label == BASE_LABEL), None)


def loo_rows(text: str) -> list[tuple[float, float, float, float]]:
    """``(value, fine-tuned, written, error)`` from a ``--leave-one-out`` summary, in file order."""
    out = []
    for line in text.splitlines():
        found = LOO.match(line)
        if found:
            out.append(tuple(float(found.group(k)) for k in ("value", "arm", "written", "error")))
    return out


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
    fine_tuned: dict[float, Measured] = field(default_factory=dict)
    written: dict[float, Measured] = field(default_factory=dict)
    unseen: dict[float, Measured] = field(default_factory=dict)
    random: dict[float, Measured] = field(default_factory=dict)
    implied: dict[float, float] = field(default_factory=dict)
    """LOO read-back of each arm's trained offset, keyed by trained offset."""
    loo: dict[float, tuple[float, float]] = field(default_factory=dict)
    """Leave-one-out (fine-tuned, written) per value, from a ``--leave-one-out`` run."""


def parse_measure_table(text: str) -> MeasureTable:
    """Read a ``014`` / ``arm_thresholds`` / ``--leave-one-out`` table into one record.

    Warns (:class:`DroppedRowsWarning`) for rows holding a NaN, which are left out.
    """
    _warn_dropped(text)
    table = MeasureTable()
    steps: set[int] = set()
    in_loo = False
    for line in text.splitlines():
        if LOO_SECTION in line:
            in_loo = True
        arm = ARM_CHECKPOINT.match(line)
        if arm:
            steps.add(int(arm.group("steps")))
            continue
        implied = IMPLIED.match(line)
        if implied:
            table.implied[round(float(implied.group("offset")), 6)] = float(implied.group("implied"))
            continue
        if in_loo:
            loo = LOO.match(line)
            if loo:
                table.loo[float(loo.group("value"))] = (float(loo.group("arm")), float(loo.group("written")))
                continue
        row = parse_row(line)
        if row is None:
            continue
        measured = Measured(row.point, row.low, row.high, row.reached)
        if row.label == BASE_LABEL:
            table.base = measured
            continue
        random = RANDOM_LABEL.match(row.label)
        if random:
            table.random[float(random.group("magnitude"))] = measured
            continue
        if row.kind == "fine-tuned":
            table.fine_tuned[row.value] = measured
        elif row.kind == "written (grid)":
            table.written[row.value] = measured
        elif row.kind == "written (unseen)":
            table.unseen[row.value] = measured
    if len(steps) > 1:
        raise ValueError(f"arms read at different budgets {sorted(steps)} in one table")
    table.read_at = steps.pop() if steps else None
    return table
