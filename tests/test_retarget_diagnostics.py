"""Tests for the warning lights.

Load-bearing: each check must turn on the failure it names and stay green on a
clean run; a check whose inputs a record lacks must skip rather than guess; the
finish line must be charged at the goal's example sizes; and study checks must
pair runs by spec.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from goalmisgen.retarget import Prequential, Record, RunSpec, diagnostics
from goalmisgen.retarget.diagnostics import Status, Thresholds

BOUNDS = [0, 1, 2, 4, 8]


@dataclasses.dataclass
class Teaching(Prequential):
    """A prequential result with a teacher curve and overhead, standing in for a mechanism that has them."""

    teacher: np.ndarray | None = None
    teacher_examples: np.ndarray | None = None
    overhead: float = 0.0

    @property
    def teacher_test_bits(self):
        return self.teacher

    @property
    def teacher_test_example_bits(self):
        return self.teacher_examples

    @property
    def overhead_bits(self) -> float:
        return self.overhead

    @property
    def total_bits(self) -> float:
        return float(self.example_bits.sum()) + self.overhead


def per_example(changed_curve, unchanged_curve, n_changed=2, n_unchanged=2) -> np.ndarray:
    return np.array([[c] * n_changed + [u] * n_unchanged for c, u in zip(changed_curve, unchanged_curve)], dtype=float)


def record(
    changed_curve=(2.0, 1.5, 1.0, 0.6, 0.5),
    unchanged_curve=(0.5, 0.5, 0.5, 0.5, 0.5),
    goal=2.5,
    own_goal_bits=0.5,
    result_type=Prequential,
    seed=0,
    **result_fields,
) -> Record:
    examples = per_example(changed_curve, unchanged_curve)
    result = result_type(
        bounds=BOUNDS,
        example_bits=np.full(8, 50.0),
        test_bits=examples.mean(axis=1),
        valid_bits=examples.mean(axis=1),
        best_step=np.zeros(5, dtype=int),
        steps_run=np.zeros(5, dtype=int),
        seconds=np.ones(5),
        test_example_bits=examples,
        candidates=4,
        **result_fields,
    )
    spec = RunSpec(base="agent", pool="pool", goal=goal, study_goals=(2.5, 10.0), n=8, seed=seed).to_dict()
    return Record(
        spec=spec,
        result=result,
        changed=np.zeros(8, dtype=bool),
        units=np.full(8, 10),
        changed_test_fraction=0.5,
        test_changed=np.array([True, True, False, False]),
        own_goal_test_bits=own_goal_bits,
    )


def status(findings, check: str) -> Status:
    (found,) = [f for f in findings if f.check == check]
    return found.status


def test_a_clean_run_passes_every_check_it_can_run():
    findings = diagnostics.run_checks(record(), thresholds=Thresholds(competence=1.0, finish=0.1))
    assert diagnostics.worst(findings) in (Status.OK, Status.WARN)
    assert {f.check for f in findings if f.status is Status.FAIL} == set()
    assert status(findings, "coded model reached the finish line") is Status.OK
    assert status(findings, "no drift on unchanged levels") is Status.OK
    assert status(findings, "teacher reached the finish line") is Status.SKIP  # prequential has no teacher


def test_an_incompetent_base_fails_the_gate_and_an_unset_gate_only_reports():
    findings = diagnostics.run_checks(record(own_goal_bits=90.0), thresholds=Thresholds(competence=1.0))
    assert status(findings, "base competent at its own goal") is Status.FAIL
    unjudged = diagnostics.run_checks(record(own_goal_bits=90.0))
    assert status(unjudged, "base competent at its own goal") is Status.SKIP


def test_a_run_that_ends_above_the_finish_line_is_a_lower_bound():
    findings = diagnostics.run_checks(record(changed_curve=(2.0, 1.8, 1.7, 1.6, 1.5)))
    (finish,) = [f for f in findings if f.check == "coded model reached the finish line"]
    assert finish.status is Status.FAIL and "lower bound" in finish.message


def test_a_flat_curve_has_not_learned():
    findings = diagnostics.run_checks(record(changed_curve=(2.0,) * 5))
    assert status(findings, "coded model learned") is Status.FAIL


def test_a_curve_still_falling_at_the_end_has_not_settled():
    findings = diagnostics.run_checks(record(changed_curve=(2.0, 2.0, 2.0, 2.0, 0.5)))
    assert status(findings, "learning curve settled") is Status.WARN


def test_drift_on_levels_the_goal_leaves_alone_is_flagged():
    findings = diagnostics.run_checks(record(unchanged_curve=(0.5, 0.9, 0.7, 0.6, 0.5)))
    (drift,) = [f for f in findings if f.check == "no drift on unchanged levels"]
    assert drift.status is Status.WARN and drift.value == pytest.approx(0.4)


def test_checks_without_their_inputs_skip():
    bare = record()
    bare.result.test_example_bits = None
    bare.test_changed = None
    bare.own_goal_test_bits = None
    bare.result.candidates = None
    findings = diagnostics.run_checks(bare)
    for check in (
        "no drift on unchanged levels",
        "coded model reached the finish line",
        "unpaid information small",
        "T resolves the finish line",
    ):
        assert status(findings, check) is Status.SKIP, check


def test_teacher_checks_run_when_the_mechanism_has_a_teacher():
    teacher = np.array([2.0, 1.0, 0.6, 0.5, 0.5])
    lagging = record(
        changed_curve=(2.0, 1.9, 1.8, 1.7, 1.6),
        result_type=Teaching,
        teacher=teacher,
        teacher_examples=per_example(teacher, (0.5, 1.2, 0.5, 0.5, 0.5)),
        overhead=1000.0,
    )
    findings = diagnostics.run_checks(lagging)
    assert status(findings, "teacher reached the finish line") is Status.OK
    assert status(findings, "coded model kept up with its teacher") is Status.WARN
    assert status(findings, "no teacher drift on unchanged levels") is Status.WARN
    assert status(findings, "overhead small") is Status.WARN  # 1000 of 1400 bits


def test_the_finish_line_is_charged_at_the_goals_example_sizes():
    r = record(own_goal_bits=1.0)
    r.own_goal_test_units = 10.0
    r.test_units = np.array([5, 5, 5, 5])
    assert diagnostics.finish_line(r) == pytest.approx(0.5)  # 0.1 bits per unit, 5 units per example
    r.test_units = None
    assert diagnostics.finish_line(r) == pytest.approx(1.0)
    r.own_goal_test_bits = None
    null = record(goal=10.0)
    assert diagnostics.finish_line(r, null) == pytest.approx(float(null.result.test_bits[0]))


def test_study_checks_pair_by_spec():
    runs = [
        ("arm", record(goal=2.5)),
        ("arm replicate", record(goal=2.5, seed=1)),
        ("null", record(goal=10.0, changed_curve=(2.0, 1.0, 1.0, 1.0, 1.0))),
    ]
    findings = diagnostics.study_checks(runs, 10.0)
    (flat,) = [f for f in findings if f.check == "null arm flat"]
    assert flat.status is Status.WARN and flat.subject == "null"
    missing = [f.subject for f in findings if f.check == "has a null arm"]
    assert missing == ["arm replicate"]  # its null would need seed 1
    (replicates,) = [f for f in findings if f.check == "replicates agree"]
    assert replicates.status is Status.OK and "arm replicate" in replicates.subject


def test_render_and_worst():
    findings = [
        diagnostics.Finding("a", Status.OK, "fine"),
        diagnostics.Finding("b", Status.SKIP, "n/a"),
        diagnostics.Finding("c", Status.WARN, "hmm", subject="run 1"),
    ]
    assert diagnostics.worst(findings) is Status.WARN
    text = diagnostics.render(findings, skipped=False)
    assert "n/a" not in text and "[run 1] hmm" in text
