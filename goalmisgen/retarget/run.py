"""One run of the metric: a base, a goal family, a pool and a spec in; a record out."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import numpy as np

from goalmisgen.retarget.family import Goal, GoalFamily
from goalmisgen.retarget.scoring import Scorer, example_bits
from goalmisgen.retarget.spec import Record, RunSpec
from goalmisgen.retarget.splits import Split, study_split


@dataclasses.dataclass
class ArmData:
    """One goal's demonstrations on the study's split: what a run codes, and what an oracle is scored on."""

    split: Split
    data: Any
    changed: np.ndarray
    """Per row of ``split.rows``: the goal's expert decides differently from the null goal's."""

    eligible: int


def arm_data(
    family: GoalFamily,
    pool,
    goal: Goal,
    study_goals: Sequence[Goal],
    n: int,
    n_valid: int,
    n_test: int,
    ordering: int = 0,
) -> ArmData:
    """The study's split of ``pool``, demonstrated at ``goal``. Every script that scores an arm goes through this."""
    if goal not in study_goals:
        raise ValueError(f"goal {goal} is not one of the study's goals {list(study_goals)}")
    family.check_pool(pool)
    eligible = family.deterministic(pool, study_goals)
    split = study_split(eligible, n, n_valid, n_test, ordering=ordering)
    data = family.demonstrations(pool, split.rows, goal)
    return ArmData(split, data, family.changed(pool, split.rows, data), len(eligible))


def measure(
    spec: RunSpec,
    family: GoalFamily,
    scorer: Scorer,
    params0,
    pool,
    log: Callable[[str], None] = print,
    provenance: dict | None = None,
) -> Record:
    """The code of ``spec.goal`` for the base ``params0`` (by ``spec.coding``), on the study's split of ``pool``."""
    arm = arm_data(family, pool, spec.goal, spec.study_goals, spec.n, spec.n_valid, spec.n_test, spec.ordering)
    split, data, changed = arm.split, arm.data, arm.changed
    log(f"goal {spec.goal}  ordering {spec.ordering}  seed {spec.seed}")
    log(f"levels: D {len(split.train):,}  V {len(split.valid):,}  T {len(split.test):,}  (of {arm.eligible:,} eligible)")
    log(
        f"the goal's expert decides differently from the null's on {changed[split.train].mean():.1%} of D, "
        f"{changed[split.test].mean():.1%} of T\n"
    )
    result = spec.coding.run(scorer, params0, data, split, spec.learner, spec.seed, log)
    own_goal = family.demonstrations(pool, split.rows[split.test], family.null)
    own_rows = np.arange(len(split.test))
    own_goal_bits = float(example_bits(scorer, params0, own_goal, own_rows).mean())
    return Record(
        spec=spec.to_dict(),
        result=result,
        changed=changed[result.inputs],
        units=result.units,
        changed_test_fraction=float(changed[split.test].mean()),
        rows_train=split.rows[result.inputs],
        test_changed=changed[split.test],
        test_units=scorer.units(data, split.test),
        own_goal_test_bits=own_goal_bits,
        own_goal_test_units=float(np.mean(scorer.units(own_goal, own_rows))),
        provenance=dict(provenance or {}),
    )
