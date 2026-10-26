"""Warning lights: whether a run's number can be read at all, checked on every run.

A retargeting cost is only a cost of retargeting if the run did what it was
meant to: the base was competent at its own goal, the coded model learned the
new one, it did not pay for damage elsewhere, and nothing reached it unpaid.
Each check here reads a :class:`~goalmisgen.retarget.spec.Record` through the
:class:`~goalmisgen.retarget.code.CodeResult` protocol only, so it applies to
any coding mechanism; a check whose inputs a mechanism or an old record lacks
reports ``skip`` rather than guessing.

Run checks take the run and its finish line (:func:`finish_line`: the base's
own score on T at its own goal). Study checks take every run of a study, since
some failures (a null arm still learning, replicates that disagree) only show
across runs. A check is a function returning a :class:`Finding`; adding one is
adding it to :data:`RUN_CHECKS` or :data:`STUDY_CHECKS`.

The pilot found each of these the slow way: a null arm still learning
navigation (bcgv11, RC.11), a memoriser failing competence (RC.8), learners
still improving at the end of the stream (RC.1), and drift on levels the goal
leaves alone (RC.3).
"""

from __future__ import annotations

import dataclasses
import enum
import json
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from goalmisgen.retarget.spec import Record, comparison_key


class Status(enum.Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


SEVERITY = {Status.SKIP: 0, Status.OK: 1, Status.WARN: 2, Status.FAIL: 3}


@dataclasses.dataclass(frozen=True)
class Finding:
    check: str
    status: Status
    message: str
    value: float | None = None
    subject: str = ""
    """What the finding is about, for study checks (a run, an agent); empty for a run's own checks."""


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """Where a light turns amber or red. In bits per example of the code (a route, for the route model).

    The defaults suit the maze pilot (base ~0.33 bits per route). Set them per
    environment from its controls: the positive control should pass every check.
    """

    competence: float | None = None
    """The base must code its own goal within this many bits per example. ``None`` reports the value without judging it."""

    finish: float = 0.02
    """How close to the finish line counts as reaching it."""

    min_drop: float = 0.01
    """A learning curve that falls by less than this has not learned."""

    still_improving: float = 0.1
    """A curve whose last checkpoint brought more than this share of its whole drop has not settled."""

    drift: float = 0.02
    """How far loss may rise, above the base's, on levels the goal leaves alone."""

    keep_up: float = 0.02
    """How far a student may end behind its teacher on T."""

    overhead_share: float = 0.2
    side_channel_share: float = 0.1
    replicate_spread: float = 0.1
    """Largest (max - min) / mean of total bits across replicates of one arm."""


def finish_line(r: Record, null: Record | None = None) -> float | None:
    """What the base pays per example of T when doing its own goal, charged at this goal's example sizes.

    The base's own-goal bits per unit, times this goal's mean units per example
    of T: examples change size with the goal (routes shorten as tau falls), and
    a per-example finish line would let a goal with short examples cross it
    without learning anything (THREADS.md RC.9). Records without unit counts
    fall back to the base's own-goal bits per example, or for pilot records to
    the null arm's L(0), unscaled.
    """
    if r.own_goal_test_bits is not None:
        if r.own_goal_test_units and r.test_units is not None:
            return float(r.own_goal_test_bits / r.own_goal_test_units * np.mean(r.test_units))
        return float(r.own_goal_test_bits)
    if null is not None:
        return float(null.result.test_bits[0])
    return None


# ----------------------------------------------------------------------
# Curves the checks and the plots read
# ----------------------------------------------------------------------


def split_curve(per_example: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
    """The mean of ``per_example[k, mask]`` at each checkpoint ``k``; ``None`` if either is missing or the mask is empty."""
    if per_example is None or mask is None or not np.any(mask):
        return None
    return np.asarray(per_example)[:, np.asarray(mask, dtype=bool)].mean(axis=1)


def curves(r: Record) -> dict[str, np.ndarray]:
    """Every learning curve a record holds, by name, each with one value per checkpoint (``r.result.bounds``)."""
    result = r.result
    unchanged = None if r.test_changed is None else ~np.asarray(r.test_changed, dtype=bool)
    named = {
        "coded model": np.asarray(result.test_bits),
        "coded model, levels the goal changes": split_curve(result.test_example_bits, r.test_changed),
        "coded model, levels it leaves alone": split_curve(result.test_example_bits, unchanged),
        "teacher": None if result.teacher_test_bits is None else np.asarray(result.teacher_test_bits),
        "teacher, levels it leaves alone": split_curve(result.teacher_test_example_bits, unchanged),
    }
    return {name: curve for name, curve in named.items() if curve is not None}


# ----------------------------------------------------------------------
# Run checks
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunContext:
    finish: float | None
    thresholds: Thresholds


def check_competence(r: Record, c: RunContext) -> Finding:
    name = "base competent at its own goal"
    if c.finish is None:
        return Finding(name, Status.SKIP, "no finish line (no own-goal score, no null arm)")
    if c.thresholds.competence is None:
        return Finding(name, Status.SKIP, f"{c.finish:.3f} bits/example; no threshold set", c.finish)
    if c.finish <= c.thresholds.competence:
        return Finding(name, Status.OK, f"{c.finish:.3f} <= {c.thresholds.competence:g} bits/example", c.finish)
    return Finding(
        name,
        Status.FAIL,
        f"{c.finish:.3f} > {c.thresholds.competence:g} bits/example: retargeting it would mean relearning the task",
        c.finish,
    )


def check_finish(r: Record, c: RunContext) -> Finding:
    name = "coded model reached the finish line"
    if c.finish is None:
        return Finding(name, Status.SKIP, "no finish line")
    end = float(r.result.test_bits[-1])
    short = end - c.finish
    if short <= c.thresholds.finish:
        return Finding(name, Status.OK, f"ends {end:.3f}, finish {c.finish:.3f}", short)
    return Finding(
        name,
        Status.FAIL,
        f"ends {short:+.3f} bits/example above the finish line ({end:.3f} vs {c.finish:.3f}): the cost is a lower bound",
        short,
    )


def check_learned(r: Record, c: RunContext) -> Finding:
    name = "coded model learned"
    curve = np.asarray(r.result.test_bits)
    drop = float(curve[0] - curve[-1])
    if c.finish is not None and curve[0] - c.finish <= c.thresholds.finish:
        return Finding(name, Status.OK, "the base already meets the finish line: nothing to learn", drop)
    if drop >= c.thresholds.min_drop:
        return Finding(name, Status.OK, f"T loss fell {drop:.3f} bits/example", drop)
    return Finding(name, Status.FAIL, f"T loss fell only {drop:.3f} bits/example", drop)


def check_settled(r: Record, c: RunContext) -> Finding:
    name = "learning curve settled"
    curve = np.asarray(r.result.test_bits)
    drop = float(curve[0] - curve[-1])
    if len(curve) < 2 or drop < c.thresholds.min_drop:
        return Finding(name, Status.SKIP, "no drop to judge")
    last = float(curve[-2] - curve[-1]) / drop
    if last <= c.thresholds.still_improving:
        return Finding(name, Status.OK, f"last checkpoint brought {last:.0%} of the drop", last)
    return Finding(name, Status.WARN, f"last checkpoint brought {last:.0%} of the drop: more data would change the cost", last)


def _drift(name: str, curve: np.ndarray | None, bounds: list[int], c: RunContext, who: str) -> Finding:
    if curve is None:
        return Finding(name, Status.SKIP, "no per-example T curve or no unchanged levels")
    rise = curve - curve[0]
    worst = int(np.argmax(rise))
    if rise[worst] <= c.thresholds.drift:
        return Finding(name, Status.OK, f"peak rise {rise[worst]:+.3f} bits/example", float(rise[worst]))
    return Finding(
        name,
        Status.WARN,
        f"{who} got {rise[worst]:+.3f} bits/example worse on levels the goal leaves alone (at t={bounds[worst]}): "
        "the cost includes repairing damage",
        float(rise[worst]),
    )


def check_drift(r: Record, c: RunContext) -> Finding:
    unchanged = None if r.test_changed is None else ~np.asarray(r.test_changed, dtype=bool)
    curve = split_curve(r.result.test_example_bits, unchanged)
    return _drift("no drift on unchanged levels", curve, r.result.bounds, c, "the coded model")


def check_teacher_learned(r: Record, c: RunContext) -> Finding:
    name = "teacher reached the finish line"
    teacher = r.result.teacher_test_bits
    if teacher is None:
        return Finding(name, Status.SKIP, "no teacher")
    if c.finish is None:
        return Finding(name, Status.SKIP, "no finish line")
    short = float(teacher[-1]) - c.finish
    if short <= c.thresholds.finish:
        return Finding(name, Status.OK, f"ends {float(teacher[-1]):.3f}, finish {c.finish:.3f}", short)
    return Finding(name, Status.FAIL, f"teacher ends {short:+.3f} bits/example above the finish line", short)


def check_keeps_up(r: Record, c: RunContext) -> Finding:
    name = "coded model kept up with its teacher"
    teacher = r.result.teacher_test_bits
    if teacher is None:
        return Finding(name, Status.SKIP, "no teacher")
    gap = float(r.result.test_bits[-1] - teacher[-1])
    if gap <= c.thresholds.keep_up:
        return Finding(name, Status.OK, f"ends {gap:+.3f} bits/example from the teacher", gap)
    return Finding(name, Status.WARN, f"ends {gap:+.3f} bits/example behind the teacher", gap)


def check_teacher_drift(r: Record, c: RunContext) -> Finding:
    unchanged = None if r.test_changed is None else ~np.asarray(r.test_changed, dtype=bool)
    curve = split_curve(r.result.teacher_test_example_bits, unchanged)
    if r.result.teacher_test_bits is None:
        return Finding("no teacher drift on unchanged levels", Status.SKIP, "no teacher")
    return _drift("no teacher drift on unchanged levels", curve, r.result.bounds, c, "the teacher")


def check_overhead(r: Record, c: RunContext) -> Finding:
    name = "overhead small"
    share = r.result.overhead_bits / r.result.total_bits if r.result.total_bits else 0.0
    status = Status.OK if share <= c.thresholds.overhead_share else Status.WARN
    return Finding(name, status, f"{r.result.overhead_bits:,.0f} bits, {share:.0%} of the code", share)


def check_side_channel(r: Record, c: RunContext) -> Finding:
    name = "unpaid information small"
    bits = r.result.side_channel_bits
    if bits is None:
        return Finding(name, Status.SKIP, "not recorded")
    share = bits / r.result.total_bits if r.result.total_bits else 0.0
    status = Status.OK if share <= c.thresholds.side_channel_share else Status.WARN
    return Finding(name, status, f"at most {bits:,.0f} bits reached the model unpaid, {share:.1%} of the code", share)


def check_resolution(r: Record, c: RunContext) -> Finding:
    name = "T resolves the finish line"
    per_example = r.result.test_example_bits
    if per_example is None:
        return Finding(name, Status.SKIP, "no per-example T curve")
    final = np.asarray(per_example)[-1]
    error = float(final.std(ddof=1) / np.sqrt(len(final))) if len(final) > 1 else float("inf")
    status = Status.OK if error <= c.thresholds.finish / 2 else Status.WARN
    return Finding(name, status, f"standard error of the final T loss {error:.4f} bits/example", error)


RUN_CHECKS: tuple[Callable[[Record, RunContext], Finding], ...] = (
    check_competence,
    check_learned,
    check_finish,
    check_settled,
    check_drift,
    check_teacher_learned,
    check_keeps_up,
    check_teacher_drift,
    check_overhead,
    check_side_channel,
    check_resolution,
)


def run_checks(r: Record, finish: float | None = None, thresholds: Thresholds = Thresholds()) -> list[Finding]:
    """Every run check on ``r``. ``finish`` defaults to the record's own (:func:`finish_line`)."""
    context = RunContext(finish if finish is not None else finish_line(r), thresholds)
    return [check(r, context) for check in RUN_CHECKS]


# ----------------------------------------------------------------------
# Study checks
# ----------------------------------------------------------------------

Runs = list[tuple[str, Record]]
"""Every run of a study, each with a name to report it by. Runs are paired by their specs, never by name."""


def check_null_flat(runs: Runs, null_goal, t: Thresholds) -> list[Finding]:
    """A null arm that keeps learning means the base was not done with its own task, and every subtraction inherits it."""
    out = []
    for name, r in runs:
        if r.goal != null_goal:
            continue
        drop = float(r.result.test_bits[0] - r.result.test_bits[-1])
        status = Status.OK if drop <= t.min_drop else Status.WARN
        out.append(Finding("null arm flat", status, f"T loss fell {drop:.3f} bits/example on its own goal", drop, name))
    return out


def check_has_null(runs: Runs, null_goal, t: Thresholds) -> list[Finding]:
    nulls = {comparison_key(r.spec) for _, r in runs if r.goal == null_goal}
    return [
        Finding("has a null arm", Status.WARN, "no run at the null goal with the same spec", None, name)
        for name, r in runs
        if r.goal != null_goal and comparison_key(r.spec) not in nulls
    ]


def check_replicates(runs: Runs, null_goal, t: Thresholds) -> list[Finding]:
    """Runs that differ only in data order or learner seed should cost about the same."""
    groups: dict[str, list[tuple[str, Record]]] = defaultdict(list)
    for name, r in runs:
        rest = {k: v for k, v in r.spec.items() if k not in ("ordering", "seed")}
        groups[json.dumps(rest, sort_keys=True)].append((name, r))
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        totals = np.array([r.result.total_bits for _, r in members])
        spread = float((totals.max() - totals.min()) / abs(totals.mean())) if totals.mean() else 0.0
        status = Status.OK if spread <= t.replicate_spread else Status.WARN
        subject = ", ".join(name for name, _ in members)
        out.append(Finding("replicates agree", status, f"total bits spread {spread:.1%} of their mean", spread, subject))
    return out


STUDY_CHECKS: tuple[Callable[[Runs, object, Thresholds], list[Finding]], ...] = (
    check_has_null,
    check_null_flat,
    check_replicates,
)


def study_checks(runs: Iterable[tuple[str, Record]], null_goal, thresholds: Thresholds = Thresholds()) -> list[Finding]:
    runs = list(runs)
    return [finding for check in STUDY_CHECKS for finding in check(runs, null_goal, thresholds)]


# ----------------------------------------------------------------------
# Reading the lights
# ----------------------------------------------------------------------


def worst(findings: Iterable[Finding]) -> Status:
    return max((f.status for f in findings), key=SEVERITY.__getitem__, default=Status.SKIP)


def render(findings: Iterable[Finding], skipped: bool = True) -> str:
    """One line per finding, in the order given; ``skipped=False`` leaves out checks that could not run."""
    lines = []
    for f in findings:
        if f.status is Status.SKIP and not skipped:
            continue
        subject = f"[{f.subject}] " if f.subject else ""
        lines.append(f"  {f.status.value.upper():<5} {f.check:<40} {subject}{f.message}")
    return "\n".join(lines)
