"""What defines a run, and the record it writes.

A :class:`RunSpec` is everything that decides a run's numbers. It is written
whole into the run's :class:`Record`, and a report groups and pairs runs by it
(:func:`comparison_key`) rather than by their file names. Records written by the
pilot's ``retarget_cost.py`` before this format existed (flat JSON, no ``spec``)
load through the same :meth:`Record.load`.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from goalmisgen.retarget.code import CodeResult, Coding, Prequential, PrequentialCoding, result_from_dict
from goalmisgen.retarget.learners import EarlyStopping, Learner

FORMAT = 2


@dataclasses.dataclass(frozen=True)
class RunSpec:
    base: str
    """The agent, as the loader that made ``params0`` names it (a run directory, ``random:<run dir>``)."""

    pool: str
    goal: Any
    study_goals: tuple
    """Every goal of the study, the null included: which levels are eligible depends on all of them (``splits.py``)."""

    n: int = 16384
    n_valid: int = 1024
    n_test: int = 2048
    ordering: int = 0
    seed: int = 0
    learner: Learner = Learner()
    """How the coded model trains: the prequential learner, or a requential code's student (whose early stopping is unused)."""

    coding: Coding = PrequentialCoding()

    def __post_init__(self) -> None:
        if self.goal not in self.study_goals:
            raise ValueError(f"goal {self.goal} is not one of the study's goals {self.study_goals}")

    def to_dict(self) -> dict:
        return {
            "base": self.base,
            "pool": self.pool,
            "goal": self.goal,
            "study_goals": list(self.study_goals),
            "n": self.n,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "ordering": self.ordering,
            "seed": self.seed,
            "learner": self.learner.to_dict(),
            "coding": self.coding.to_dict(),
        }


def comparison_key(spec: dict) -> str:
    """Everything that must match for two runs to be compared across goals: the spec less its goal.

    A learner's parameterisation counts by kind and name only, so a held-out
    axis refitted per goal (one file per goal, one name) pairs with its own
    null arm.
    """
    rest = {k: v for k, v in spec.items() if k not in ("goal", "study_goals")}
    rest.setdefault("coding", PrequentialCoding().to_dict())  # records from before the field were all prequential
    learner = dict(rest["learner"])
    learner["trains"] = {k: v for k, v in learner["trains"].items() if k in ("kind", "name")}
    rest["learner"] = learner
    return json.dumps(rest, sort_keys=True)


@dataclasses.dataclass
class Record:
    """One run: its spec, its code, and the facts about D and T that references and diagnostics read."""

    spec: dict
    result: CodeResult
    changed: np.ndarray
    """Per coded example: on its maze, the goal's expert decides differently from the null goal's."""

    units: np.ndarray | None
    """Per coded example: coded units (tokens). ``None`` in pilot records that predate it."""

    changed_test_fraction: float
    rows_train: np.ndarray | None = None
    """Per coded example, the pool row of its maze, in coding order."""

    test_changed: np.ndarray | None = None
    """Per example of T: whether the goal changes the expert's decision. ``None`` in pilot records."""

    test_units: np.ndarray | None = None
    """Per example of T: coded units (tokens) at this run's goal."""

    own_goal_test_bits: float | None = None
    """The base's mean bits per example on T demonstrated at the null goal: how well it does its own task."""

    own_goal_test_units: float | None = None
    """Mean units per example of T demonstrated at the null goal, to put the finish line per unit."""

    provenance: dict = dataclasses.field(default_factory=dict)

    @property
    def goal(self):
        return self.spec["goal"]

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "spec": self.spec,
                    "provenance": self.provenance,
                    "result": self.result.to_dict(),
                    "train": {
                        "rows": None if self.rows_train is None else np.asarray(self.rows_train).tolist(),
                        "units": None if self.units is None else np.asarray(self.units).tolist(),
                        "changed": np.asarray(self.changed).astype(int).tolist(),
                    },
                    "changed_test_fraction": self.changed_test_fraction,
                    "test": {
                        "changed": None if self.test_changed is None else np.asarray(self.test_changed).astype(int).tolist(),
                        "units": None if self.test_units is None else np.asarray(self.test_units).tolist(),
                        "own_goal_bits": self.own_goal_test_bits,
                        "own_goal_units": self.own_goal_test_units,
                    },
                }
            )
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "Record":
        data = json.loads(pathlib.Path(path).read_text())
        if "format" not in data:
            return _pilot_record(data)
        train = data["train"]
        test = data.get("test", {})
        return cls(
            spec=data["spec"],
            result=result_from_dict(data["result"]),
            changed=np.asarray(train["changed"], dtype=bool),
            units=None if train["units"] is None else np.asarray(train["units"]),
            changed_test_fraction=float(data["changed_test_fraction"]),
            rows_train=None if train["rows"] is None else np.asarray(train["rows"]),
            test_changed=None if test.get("changed") is None else np.asarray(test["changed"], dtype=bool),
            test_units=None if test.get("units") is None else np.asarray(test["units"]),
            own_goal_test_bits=test.get("own_goal_bits"),
            own_goal_test_units=test.get("own_goal_units"),
            provenance=data.get("provenance", {}),
        )


_OPTIMISER = ("learning_rate", "batch_size", "weight_decay", "clip_norm", "warmup_steps")
_STOPPING = ("eval_first", "eval_ratio", "patience", "max_steps")


def _pilot_record(data: dict) -> Record:
    """A record from the pilot's flat format (2026-10-10..14)."""
    flat = data["learner"]
    if "subspace" in data:
        trains = {"kind": "subspace", **data["subspace"]}
    elif flat.get("only"):
        trains = {"kind": "tensor", "name": flat["only"]}
    else:
        trains = {"kind": "all"}
    spec = {
        "base": data["base"],
        "pool": data["pool"],
        "goal": data["tau"],
        "study_goals": sorted({*data.get("taus_dropped", []), data["tau"]}),
        "n": len(data["example_bits"]),
        "n_valid": data["n_valid"],
        "n_test": data["n_test"],
        "ordering": data["ordering"],
        "seed": data["seed"],
        "learner": {
            "optimiser": {k: flat[k] for k in _OPTIMISER},
            "stopping": {k: flat[k] for k in _STOPPING},
            "trains": trains,
        },
    }
    result = Prequential.from_dict(data)
    result.candidates = EarlyStopping(**spec["learner"]["stopping"]).candidates
    return Record(
        spec=spec,
        result=result,
        changed=np.asarray(data["flipped_train"], dtype=bool),
        units=np.asarray(data["tokens_train"]) if "tokens_train" in data else None,
        changed_test_fraction=float(data["flipped_test_fraction"]),
        provenance={"format": "pilot", "level_index_train": data.get("level_index_train")},
    )
