"""The coding mechanisms, and what every one of them reports.

A mechanism is a :class:`Coding`: a small config that runs itself on a base, the
goal's data and the study's split, and reports a :class:`CodeResult`: the bits
it paid, the coded model's learning curve on a test set T, and whatever it pays
outside the per-example cost. Everything downstream (references, diagnostics,
reports) reads only those two protocols, so a new mechanism is one more module
that registers its result kind (:data:`RESULT_KINDS`) and nothing else. The
block-prequential code is here; requential coding is in ``requential.py``.

**The block-prequential code** (:func:`prequential`). Blocks of the stream ``D``
are encoded in order. Before block ``k`` the learner fits afresh from the base
on everything before it, and the block is encoded by that fit; block 0 is
encoded by the base itself. The sum of those code lengths is the stream's MDL
for this (base, learner). Refitting from the base, rather than continuing the
last fit, keeps each block's model a function of the data before it alone,
whatever order the fits ran in. The block boundaries are a parameter
(:func:`doubling_blocks` is the pilot's). Doubling puts most of the stream in the
last blocks: in the pilot the last two carried 63-82% of the cost.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, ClassVar, Protocol

import numpy as np

from goalmisgen.retarget.learners import Learner, make_fitter
from goalmisgen.retarget.scoring import Scorer, example_bits
from goalmisgen.retarget.splits import Split


class CodeResult(Protocol):
    """What a coding mechanism reports. Checkpoints ``k = 0..K`` are at ``bounds[k]`` coded examples."""

    kind: ClassVar[str]
    bounds: list[int]

    example_bits: np.ndarray
    """The per-example cost, in coding order: the bits each coded example was charged."""

    inputs: np.ndarray | None
    """Per coded example, the row of the goal's data it was made from (its maze); ``None`` in pilot records."""

    units: np.ndarray | None
    """Per coded example, its coded units (tokens); ``None`` in pilot records."""

    test_bits: np.ndarray
    """``(K + 1,)``: the coded model's mean bits per example on T at each checkpoint, the learning curve."""

    test_example_bits: np.ndarray | None
    """``(K + 1, |T|)``: the same per example of T, so the curve can be split (levels the goal changes or not)."""

    @property
    def n(self) -> int:
        ...

    @property
    def total_bits(self) -> float:
        """The whole code length: per-example cost plus overhead."""
        ...

    @property
    def overhead_bits(self) -> float:
        """Bits paid per message rather than per example (a requential code's index overhead); 0 if none."""
        ...

    @property
    def side_channel_bits(self) -> float | None:
        """An upper bound on information that reached the coded model without being paid for (selection on V); ``None`` if unknown."""
        ...

    @property
    def teacher_test_bits(self) -> np.ndarray | None:
        """``(K + 1,)``: a teacher's learning curve on T, for mechanisms that have one."""
        ...

    @property
    def teacher_test_example_bits(self) -> np.ndarray | None:
        ...

    def to_dict(self) -> dict:
        ...


class Coding(Protocol):
    """A coding mechanism's settings, which run themselves. Part of a run's spec."""

    kind: ClassVar[str]

    def run(
        self, scorer: Scorer, params0, data, split: Split, learner: Learner, seed: int, log: Callable[[str], None]
    ) -> CodeResult:
        """Code the goal's ``data`` on ``split`` for the base ``params0``; ``learner`` trains the coded model."""
        ...

    def to_dict(self) -> dict:
        ...


def doubling_blocks(n: int) -> list[int]:
    """``0, 1, 2, 4, ..., n``: the boundaries ``t_k`` of the blocks."""
    bounds, t = [0], 1
    while t < n:
        bounds.append(t)
        t *= 2
    bounds.append(n)
    return bounds


@dataclasses.dataclass
class Prequential:
    """The block-prequential code's :class:`CodeResult`; the coded model at checkpoint ``k`` is the fit on ``t_k`` examples."""

    kind: ClassVar[str] = "prequential"

    bounds: list[int]
    example_bits: np.ndarray
    """``-log2 p`` of each example of D, in order, under the model fitted before its block."""

    test_bits: np.ndarray
    valid_bits: np.ndarray
    best_step: np.ndarray
    steps_run: np.ndarray
    seconds: np.ndarray
    recorded: np.ndarray | None = None
    """``(K + 1, ...)``: what each fit's parameterisation recorded (a subspace's coefficients), if anything."""

    test_example_bits: np.ndarray | None = None
    inputs: np.ndarray | None = None
    units: np.ndarray | None = None
    candidates: int | None = None
    """How many points each fit chose among on V (``EarlyStopping.candidates``); ``None`` in records that predate it."""

    @property
    def n(self) -> int:
        return int(self.bounds[-1])

    @property
    def total_bits(self) -> float:
        return float(self.example_bits.sum())

    @property
    def overhead_bits(self) -> float:
        return 0.0

    @property
    def side_channel_bits(self) -> float | None:
        """``log2`` of the candidates, once per fit that encoded a block after the first (the base encodes block 0)."""
        if self.candidates is None:
            return None
        return max(len(self.bounds) - 2, 0) * math.log2(self.candidates)

    @property
    def teacher_test_bits(self) -> None:
        return None

    @property
    def teacher_test_example_bits(self) -> None:
        return None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "bounds": list(self.bounds),
            "example_bits": self.example_bits.tolist(),
            "test_bits": self.test_bits.tolist(),
            "valid_bits": self.valid_bits.tolist(),
            "best_step": self.best_step.tolist(),
            "steps_run": self.steps_run.tolist(),
            "seconds": self.seconds.tolist(),
            **({} if self.recorded is None else {"recorded": self.recorded.tolist()}),
            **({} if self.test_example_bits is None else {"test_example_bits": self.test_example_bits.tolist()}),
            **({} if self.inputs is None else {"inputs": self.inputs.tolist()}),
            **({} if self.units is None else {"units": self.units.tolist()}),
            "candidates": self.candidates,
            "total_bits": self.total_bits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prequential":
        recorded = data.get("recorded", data.get("coefficients"))
        test_example_bits = data.get("test_example_bits")
        return cls(
            bounds=list(data["bounds"]),
            example_bits=np.asarray(data["example_bits"], dtype=np.float64),
            test_bits=np.asarray(data["test_bits"], dtype=np.float64),
            valid_bits=np.asarray(data["valid_bits"], dtype=np.float64),
            best_step=np.asarray(data["best_step"]),
            steps_run=np.asarray(data["steps_run"]),
            seconds=np.asarray(data["seconds"], dtype=np.float64),
            recorded=None if recorded is None else np.asarray(recorded),
            test_example_bits=None if test_example_bits is None else np.asarray(test_example_bits, dtype=np.float64),
            inputs=None if data.get("inputs") is None else np.asarray(data["inputs"]),
            units=None if data.get("units") is None else np.asarray(data["units"]),
            candidates=data.get("candidates"),
        )


RESULT_KINDS: dict[str, Callable[[dict], CodeResult]] = {Prequential.kind: Prequential.from_dict}
"""How a record's ``result`` is read back, by its ``kind``. A new mechanism registers itself here."""


def result_from_dict(data: dict) -> CodeResult:
    kind = data.get("kind", Prequential.kind)  # records before kinds were all prequential
    if kind not in RESULT_KINDS:
        raise ValueError(f"no coding mechanism {kind!r}; known: {sorted(RESULT_KINDS)}")
    return RESULT_KINDS[kind](data)


def prequential(
    scorer: Scorer,
    params0,
    data,
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    learner: Learner,
    seed: int = 0,
    blocks: Callable[[int], list[int]] = doubling_blocks,
    log: Callable[[str], None] = print,
) -> Prequential:
    """Encode ``train`` block by block, refitting from ``params0`` before each block.

    ``train`` is in the order the code reads it; the caller permutes it for an
    ordering replicate. ``seed`` drives the learner's batch sampling, and fit
    ``k`` uses ``seed`` offset by ``k`` so that two orderings at one seed differ
    in data and nothing else.
    """
    train = np.asarray(train)
    fit = make_fitter(scorer, learner)
    bounds = blocks(len(train))
    if bounds[0] != 0 or bounds[-1] != len(train) or any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"block boundaries must rise strictly from 0 to {len(train)}, got {bounds}")
    coded = np.empty(len(train), dtype=np.float64)
    test_curve, valid_bits, best_step, steps_run, seconds, recorded = [], [], [], [], [], []
    for k in range(len(bounds)):
        started = time.perf_counter()
        fitted = fit(params0, data, train[: bounds[k]], valid, seed=seed * 1000 + k)
        if k + 1 < len(bounds):
            coded[bounds[k] : bounds[k + 1]] = example_bits(scorer, fitted.params, data, train[bounds[k] : bounds[k + 1]])
        test_curve.append(example_bits(scorer, fitted.params, data, test))
        valid_bits.append(fitted.valid_bits)
        best_step.append(fitted.best_step)
        steps_run.append(fitted.steps_run)
        seconds.append(time.perf_counter() - started)
        recorded.append(fitted.recorded)
        extra = "" if fitted.recorded is None else "  c " + " ".join(f"{x:+.3f}" for x in np.ravel(fitted.recorded))
        log(
            f"  t={bounds[k]:>6}  best step {fitted.best_step:>5}/{fitted.steps_run:<5}  "
            f"valid {fitted.valid_bits:8.3f}  test {test_curve[-1].mean():8.3f} bits  ({seconds[-1]:.0f}s){extra}"
        )
    return Prequential(
        bounds=bounds,
        example_bits=coded,
        test_bits=np.asarray([float(t.mean()) for t in test_curve]),
        valid_bits=np.asarray(valid_bits),
        best_step=np.asarray(best_step),
        steps_run=np.asarray(steps_run),
        seconds=np.asarray(seconds),
        recorded=None if recorded[0] is None else np.stack(recorded),
        test_example_bits=np.stack(test_curve),
        inputs=train,
        units=scorer.units(data, train),
        candidates=learner.stopping.candidates,
    )


@dataclasses.dataclass(frozen=True)
class PrequentialCoding:
    """The block-prequential code of D: refit from the base before each block, early-stopped on V."""

    kind: ClassVar[str] = "prequential"
    blocks: str = "doubling"

    def run(self, scorer, params0, data, split, learner, seed, log) -> Prequential:
        if self.blocks != "doubling":
            raise ValueError(f"no block schedule {self.blocks!r}")
        return prequential(scorer, params0, data, split.train, split.valid, split.test, learner, seed=seed, log=log)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "blocks": self.blocks}
