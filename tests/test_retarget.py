"""Tests for the task-agnostic parts of the retargeting cost: splits, specs and records, and the metric.

Load-bearing: a study's split must not depend on the arm; a record must carry
its spec and read back identically, pilot records included; runs pair with
their null arm by spec, not file name; and each reference and normalisation
must be the arithmetic its docstring states, checked here on records whose
answers are known by hand. The parts that need a model are tested in
``test_offline_retarget.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from goalmisgen.retarget import EarlyStopping, Learner, OneTensor, Optimiser, Prequential, Record, RunSpec, metric, study_split
from goalmisgen.retarget.code import result_from_dict
from goalmisgen.retarget.spec import comparison_key

GOALS = (2.5, 10.0)


def spec(goal=2.5, **changes) -> dict:
    return RunSpec(base="agent", pool="pool", goal=goal, study_goals=GOALS, n=4, **changes).to_dict()


def record(goal=2.5, bits=(1.0, 2.0, 3.0, 4.0), units=(2, 2, 4, 4), test=(3.0, 1.0, 0.5, 0.5), **changes) -> Record:
    bounds = [0, 1, 2, 4]
    result = Prequential(
        bounds=bounds,
        example_bits=np.asarray(bits, dtype=float),
        test_bits=np.asarray(test, dtype=float),
        valid_bits=np.asarray(test, dtype=float),
        best_step=np.zeros(4, dtype=int),
        steps_run=np.zeros(4, dtype=int),
        seconds=np.ones(4),
    )
    changed = np.array([True, False, True, False])
    return Record(spec(goal, **changes), result, changed, None if units is None else np.asarray(units), 0.5)


# ----------------------------------------------------------------------
# Splits and specs
# ----------------------------------------------------------------------


def test_the_split_is_t_then_d_then_v_and_ordering_moves_only_d():
    eligible = np.arange(100, 200)
    base = study_split(eligible, n=10, n_valid=5, n_test=3)
    assert len(set(base.rows)) == 18 and set(base.rows) <= set(eligible)
    np.testing.assert_array_equal(base.test, np.arange(3))
    np.testing.assert_array_equal(base.valid, np.arange(13, 18))
    shuffled = study_split(eligible, n=10, n_valid=5, n_test=3, ordering=1)
    np.testing.assert_array_equal(shuffled.rows, base.rows)
    assert sorted(shuffled.train) == list(base.train) and list(shuffled.train) != list(base.train)
    smaller_v = study_split(eligible, n=10, n_valid=2, n_test=3)
    np.testing.assert_array_equal(smaller_v.rows[:13], base.rows[:13])
    with pytest.raises(ValueError, match="fewer"):
        study_split(eligible, n=100, n_valid=5, n_test=3)


def test_a_spec_refuses_a_goal_outside_its_study():
    with pytest.raises(ValueError, match="study's goals"):
        RunSpec(base="a", pool="p", goal=6.5, study_goals=GOALS)


def test_runs_pair_across_goals_by_everything_else():
    assert comparison_key(spec(2.5)) == comparison_key(spec(10.0))
    assert comparison_key(spec(2.5, seed=1)) != comparison_key(spec(10.0))
    slow = Learner(optimiser=Optimiser(learning_rate=1e-5))
    assert comparison_key(spec(2.5, learner=slow)) != comparison_key(spec(10.0))
    dial = Learner(trains=OneTensor("goal_embedding"))
    assert comparison_key(spec(2.5, learner=dial)) == comparison_key(spec(10.0, learner=dial))


def test_a_subspace_pairs_by_name_not_by_file():
    """A held-out axis is one file per goal under one name; each arm pairs with that name's null."""
    a, b = spec(2.5), spec(10.0)
    a["learner"]["trains"] = {"kind": "subspace", "name": "axis-heldout", "k": 1, "path": "tau2.5.npz", "meta": {"tau": 2.5}}
    b["learner"]["trains"] = {"kind": "subspace", "name": "axis-heldout", "k": 1, "path": "tau10.npz", "meta": {"tau": 10}}
    assert comparison_key(a) == comparison_key(b)


def test_a_record_reads_back_as_written(tmp_path):
    r = dataclasses.replace(record(), rows_train=np.array([7, 3, 9, 1]), provenance={"commit": "abc"})
    r.save(tmp_path / "r.json")
    back = Record.load(tmp_path / "r.json")
    assert back.spec == r.spec and back.provenance == r.provenance
    for name in ("example_bits", "test_bits", "bounds"):
        np.testing.assert_array_equal(getattr(back.result, name), getattr(r.result, name))
    np.testing.assert_array_equal(back.changed, r.changed)
    np.testing.assert_array_equal(back.units, r.units)
    np.testing.assert_array_equal(back.rows_train, r.rows_train)


def test_a_pilot_record_loads_with_its_spec(tmp_path):
    import json

    flat = {
        "base": "/runs/bcnv11.s1",
        "pool": "/pool",
        "tau": 2.5,
        "taus_dropped": [10.0],
        "ordering": 0,
        "seed": 0,
        "n_valid": 1024,
        "n_test": 2048,
        "learner": {
            "learning_rate": 3e-3,
            "batch_size": 256,
            "weight_decay": 0.01,
            "clip_norm": 1.0,
            "warmup_steps": 20,
            "eval_first": 5,
            "eval_ratio": 1.25,
            "patience": 6,
            "max_steps": 2000,
            "only": "goal_embedding",
        },
        "flipped_train": [1, 0],
        "flipped_test_fraction": 0.3,
        "tokens_train": [5, 7],
        "bounds": [0, 1, 2],
        "example_bits": [1.5, 2.5],
        "test_bits": [2.0, 1.0, 0.5],
        "valid_bits": [2.0, 1.0, 0.5],
        "best_step": [0, 5, 6],
        "steps_run": [0, 9, 9],
        "seconds": [1, 1, 1],
        "mdl": 4.0,
        "edl": 3.0,
    }
    (tmp_path / "old.json").write_text(json.dumps(flat))
    r = Record.load(tmp_path / "old.json")
    assert r.goal == 2.5 and r.spec["n"] == 2
    assert r.spec["learner"]["trains"] == {"kind": "tensor", "name": "goal_embedding"}
    assert r.spec["learner"]["optimiser"]["learning_rate"] == 3e-3
    assert r.result.total_bits == 4.0 and metric.edl(r) == 3.0
    assert r.result.candidates == EarlyStopping(**r.spec["learner"]["stopping"]).candidates
    np.testing.assert_array_equal(r.units, [5, 7])


# ----------------------------------------------------------------------
# The metric
# ----------------------------------------------------------------------


def test_ceiling_and_cumulative():
    r = record()
    assert metric.ceiling(r) == 4 * 3.0
    assert metric.cumulative(r.result.example_bits, r.result.bounds) == [0.0, 1.0, 3.0, 10.0]
    assert metric.block_sums(r.result.example_bits, r.result.bounds) == [1.0, 2.0, 7.0]


def test_excess_over_the_null_per_route_and_per_unit():
    arm = record(bits=(2.0, 2.0, 6.0, 6.0), units=(2, 2, 2, 2), test=(4.0, 1.0, 1.0, 1.0))
    null = record(goal=10.0, bits=(1.0, 1.0, 4.0, 4.0), units=(1, 1, 4, 4), test=(2.0, 1.0, 1.0, 1.0))
    assert metric.excess_over_null(arm, null, "route") == 16.0 - 10.0
    # per block: null bits per unit 1, 1, 1 -> charged at the arm's units 2, 2, 4
    assert metric.excess_by_block(arm, null) == [2.0 - 2.0, 2.0 - 2.0, 12.0 - 4.0]
    assert metric.excess_over_null(arm, null, "unit") == 8.0
    assert metric.ceiling_over_null(arm, null, "route") == 4 * 4.0 - 4 * 2.0
    # null L(0) per unit = 2 / 2.5; arm units per example = 2
    assert metric.ceiling_over_null(arm, null, "unit") == pytest.approx(4 * (4.0 - 2.0 / 2.5 * 2))
    assert metric.fraction_paid(arm, null) == pytest.approx(6.0 / 8.0)


def test_per_unit_references_need_unit_counts():
    arm, null = record(), record(goal=10.0, units=None)
    assert metric.excess_over_null(arm, null, "unit") is None
    assert metric.ceiling_over_null(arm, null, "unit") is None
    assert metric.fraction_paid(arm, null, "unit") is None


def test_the_null_of_a_run_is_the_same_spec_at_the_null_goal():
    runs = [
        ("a", record(2.5)),
        ("a", record(10.0)),
        ("a", record(10.0, seed=1)),
        ("b", record(10.0)),
    ]
    nulls = metric.null_arms(runs, 10.0)
    assert metric.null_of(nulls, "a", runs[0][1]) is runs[1][1]
    assert metric.null_of(nulls, "a", record(2.5, seed=1)) is runs[2][1]
    assert metric.null_of(nulls, "c", runs[0][1]) is None


def test_the_oracle_reference_codes_the_same_examples():
    r = record()
    assert metric.excess_over_oracle(r, [0.5, 0.5, 0.5, 0.5]) == 10.0 - 2.0
    with pytest.raises(ValueError, match="coded 3"):
        metric.excess_over_oracle(r, [0.5, 0.5, 0.5])


def test_competence_gate_and_t90():
    r = record(test=(3.0, 1.0, 0.5, 0.0))
    assert metric.competent(r, 3.0) and not metric.competent(r, 2.9)
    assert metric.t90(r) == 4  # 90% of a drop of 3 is reached at 0.3, first at t = 4
    assert metric.t90(record(test=(1.0, 1.0, 1.0, 1.0))) is None


def test_a_record_names_its_mechanism_and_unknown_ones_are_refused():
    r = record()
    assert r.result.to_dict()["kind"] == "prequential"
    assert isinstance(result_from_dict(r.result.to_dict()), Prequential)
    with pytest.raises(ValueError, match="no coding mechanism"):
        result_from_dict({**r.result.to_dict(), "kind": "mystery"})


def test_a_spec_names_its_coding_and_pilot_specs_pair_as_prequential():
    from goalmisgen.retarget import RequentialCoding

    new = spec(2.5)
    assert new["coding"] == {"kind": "prequential", "blocks": "doubling"}
    pilot = {k: v for k, v in spec(10.0).items() if k != "coding"}
    assert comparison_key(new) == comparison_key(pilot)
    assert comparison_key(spec(2.5, coding=RequentialCoding())) != comparison_key(spec(10.0))
