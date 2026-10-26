"""Requential coding: the bits to *teach* the base a goal, rather than to transmit the goal's data.

Qiu, Finzi, Zheng, Zhang & Wilson (2026), arXiv 2607.11883. A student, starting
at the base, trains on examples it draws from its own distribution; a teacher
chooses which of the student's draws to accept, by relative entropy coding
(REC), so that each accepted batch is a draw from the teacher. The message is
the accepted index, about ``KL(Q_t || P_t)`` bits, and the decoder rebuilds the
student from the messages alone: the teacher is never sent and may be trained
however one likes. The code is then an upper bound on the student's description
length given the base, independent of parameter count and of the data's
entropy.

For retargeting, the student is the base and so is the teacher at the start
(the first call costs nothing); the teacher then moves toward the goal, here by
fine-tuning on the goal's demonstrations (:class:`FineTunedTeacher`), or stays
at a fixed setting (:class:`FixedTeacher`: an oracle dial). Residual error the
teacher and student share is never charged, so at the agent's own goal a
teacher that stays at the base costs nothing at all.

Per REC call (one batch of ``batch`` examples) the expected message is at most
``KL + 2 log2(1 + KL) + kappa`` bits, ``kappa < 5.21`` (the paper's Eq. 2 with
an Elias-delta index code). The KL is estimated as the paper does, by
``log Q(x) - log P(x)`` over the teacher's draws, and REC itself is never run:
measuring the code needs only those draws. Large batches amortise the per-call
overhead; the student trains on each call's batch in minibatches of its own
optimiser's size (``student_epochs`` passes), so the REC batch and the
optimiser batch are separate choices.

Not here yet: the paper's iso-loss projection (reset the teacher to the student
and retrain it to its old loss, which shortens the code).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, ClassVar, Protocol

import jax
import numpy as np

from goalmisgen.retarget.code import RESULT_KINDS
from goalmisgen.retarget.learners import AllWeights, Learner, Optimiser, Parameterisation, Trainer
from goalmisgen.retarget.scoring import Sampler, example_bits
from goalmisgen.retarget.splits import Split

BETA = math.exp(-1) * math.log2(math.e) + 1
KAPPA = 1 + BETA + 2 * math.log2(1 + BETA)
"""The constant in the per-call bound, below 5.21 (the paper's Appendix A)."""


def call_overhead(kl: float) -> float:
    """Bits a REC call pays beyond its KL: the universal index code's ``2 log2(1 + KL) + kappa``."""
    return 2 * math.log2(1 + max(kl, 0.0)) + KAPPA


# ----------------------------------------------------------------------
# Teachers
# ----------------------------------------------------------------------


class TeacherRun(Protocol):
    def params(self) -> Any:
        """The distribution the next call's examples are drawn from."""
        ...

    def advance(self) -> None:
        """One call's worth of teacher training."""
        ...


class Teacher(Protocol):
    """How the teacher moves. Never transmitted, so any procedure gives a valid code."""

    def begin(self, scorer: Sampler, params0, data, rows: np.ndarray, steps_per_call: int, seed: int) -> TeacherRun:
        """Start from the base ``params0``; ``data[rows]`` is the goal's real data it may train on."""
        ...

    def describe(self) -> dict:
        ...


@dataclasses.dataclass(frozen=True)
class EMA:
    """An exponential moving average of the teacher, the paper's teacher smoothing.

    The timescale at step ``t`` is ``max(min_steps, fraction * t)``, so the
    average forgets on a scale that grows with training, never shorter than
    ``min_steps``. The student pays to follow every wobble of the teacher it
    learns from; averaging removes the SGD noise it would otherwise pay for.
    """

    fraction: float = 0.01
    min_steps: int = 50

    def decay(self, step: int) -> float:
        return math.exp(-1.0 / max(self.min_steps, self.fraction * step))


_blend = jax.jit(
    lambda average, latest, decay: jax.tree_util.tree_map(lambda a, b: decay * a + (1 - decay) * b, average, latest)
)


@dataclasses.dataclass(frozen=True, eq=False)
class FineTunedTeacher:
    """A teacher fine-tuned on the goal's demonstrations alongside the student.

    ``steps_per_call`` defaults to the student's, so the two move at one pace,
    the paper's choice: their divergence stays small because their training
    dynamics match. ``trains`` restricts the teacher like any learner (a goal
    vector, a subspace), which keeps it from changing what the goal does not
    need and so keeps the student from paying for that.
    """

    optimiser: Optimiser = Optimiser()
    trains: Parameterisation = AllWeights()
    steps_per_call: int | None = None
    ema: EMA | None = None

    def begin(self, scorer, params0, data, rows, steps_per_call, seed) -> TeacherRun:
        return _FineTuning(self, scorer, params0, data, rows, steps_per_call, seed)

    def describe(self) -> dict:
        return {
            "kind": "fine-tuned",
            "optimiser": dataclasses.asdict(self.optimiser),
            "trains": self.trains.describe(),
            "steps_per_call": self.steps_per_call,
            "ema": None if self.ema is None else dataclasses.asdict(self.ema),
        }


class _FineTuning:
    def __init__(self, spec: FineTunedTeacher, scorer: Sampler, params0, data, rows, steps_per_call: int, seed: int):
        self.spec, self.scorer, self.data, self.rows = spec, scorer, data, np.asarray(rows)
        self.trainer = Trainer(scorer, spec.optimiser, spec.trains)
        self.steps_per_call = spec.steps_per_call or steps_per_call
        self.state = self.trainer.start(params0)
        self.average = params0
        self.rng = np.random.default_rng(seed)
        self.step = 0

    def params(self):
        return self.average if self.spec.ema is not None else self.trainer.params(self.state)

    def advance(self) -> None:
        batch = min(self.trainer.batch_size, len(self.rows))
        for _ in range(self.steps_per_call):
            index = np.sort(self.rng.choice(self.rows, size=batch, replace=False))
            self.state = self.trainer.step(self.state, self.scorer.batch(self.data, index))
            self.step += 1
            if self.spec.ema is not None:
                self.average = _blend(self.average, self.trainer.params(self.state), self.spec.ema.decay(self.step))


@dataclasses.dataclass(frozen=True, eq=False)
class FixedTeacher:
    """A teacher that sits at given parameters throughout: the agent with its dial set by an oracle, say."""

    params: Any
    name: str

    def begin(self, scorer, params0, data, rows, steps_per_call, seed) -> TeacherRun:
        return _Fixed(self.params)

    def describe(self) -> dict:
        return {"kind": "fixed", "name": self.name}


@dataclasses.dataclass
class _Fixed:
    fixed: Any

    def params(self):
        return self.fixed

    def advance(self) -> None:
        pass


# ----------------------------------------------------------------------
# The result
# ----------------------------------------------------------------------


@dataclasses.dataclass
class Requential:
    """Requential coding's :class:`~goalmisgen.retarget.code.CodeResult`; checkpoints are after whole calls."""

    kind: ClassVar[str] = "requential"

    bounds: list[int]
    example_bits: np.ndarray
    """Per coded example, ``log2 Q(x) - log2 P(x)``: its share of its call's estimated KL."""

    call_kl: np.ndarray
    """Per call, the estimated ``KL(Q_t || P_t)`` of its whole batch, in bits."""

    test_bits: np.ndarray
    test_example_bits: np.ndarray
    teacher_curve: np.ndarray
    teacher_example_curve: np.ndarray
    inputs: np.ndarray
    units: np.ndarray
    truncated: np.ndarray
    """Per coded example: the sampler had to cut the draw short, so it is not an exact draw from the teacher.

    A cut draw is still scored as a whole route, end token included, under both
    teacher and student, so its share of the KL charges an end token nobody
    drew. :func:`goalmisgen.retarget.diagnostics.check_draws_complete` lights
    when any are.
    """

    seconds: np.ndarray
    """Wall-clock seconds spent reaching each checkpoint from the one before."""

    @property
    def n(self) -> int:
        return int(self.bounds[-1])

    @property
    def overhead_bits(self) -> float:
        return float(sum(call_overhead(kl) for kl in self.call_kl))

    @property
    def total_bits(self) -> float:
        return float(self.example_bits.sum()) + self.overhead_bits

    @property
    def side_channel_bits(self) -> float:
        return 0.0  # nothing is selected on held-out data; the teacher is free by construction

    @property
    def teacher_test_bits(self) -> np.ndarray:
        return self.teacher_curve

    @property
    def teacher_test_example_bits(self) -> np.ndarray:
        return self.teacher_example_curve

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "bounds": list(self.bounds),
            "example_bits": self.example_bits.tolist(),
            "call_kl": self.call_kl.tolist(),
            "test_bits": self.test_bits.tolist(),
            "test_example_bits": self.test_example_bits.tolist(),
            "teacher_curve": self.teacher_curve.tolist(),
            "teacher_example_curve": self.teacher_example_curve.tolist(),
            "inputs": self.inputs.tolist(),
            "units": self.units.tolist(),
            "truncated": self.truncated.astype(int).tolist(),
            "seconds": self.seconds.tolist(),
            "total_bits": self.total_bits,
            "overhead_bits": self.overhead_bits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Requential":
        return cls(
            bounds=list(data["bounds"]),
            example_bits=np.asarray(data["example_bits"], dtype=np.float64),
            call_kl=np.asarray(data["call_kl"], dtype=np.float64),
            test_bits=np.asarray(data["test_bits"], dtype=np.float64),
            test_example_bits=np.asarray(data["test_example_bits"], dtype=np.float64),
            teacher_curve=np.asarray(data["teacher_curve"], dtype=np.float64),
            teacher_example_curve=np.asarray(data["teacher_example_curve"], dtype=np.float64),
            inputs=np.asarray(data["inputs"]),
            units=np.asarray(data["units"]),
            truncated=np.asarray(data["truncated"], dtype=bool),
            seconds=np.asarray(data["seconds"], dtype=np.float64),
        )


RESULT_KINDS[Requential.kind] = Requential.from_dict


# ----------------------------------------------------------------------
# The code
# ----------------------------------------------------------------------


def checkpoint_calls(calls: int, count: int) -> list[int]:
    """``0``, then about ``count`` log-spaced call counts up to ``calls``, distinct and rising."""
    if calls < 1:
        raise ValueError("a requential code needs at least one call")
    spaced = np.unique(np.round(np.geomspace(1, calls, num=max(count, 1))).astype(int))
    return [0, *spaced.tolist()]


@dataclasses.dataclass(frozen=True, eq=False)
class RequentialCoding:
    """The requential code of the goal: ``calls`` REC calls of ``batch`` examples each.

    The student is the run's learner (its optimiser and what it trains; early
    stopping does not apply). Inputs for each call are drawn from D's rows; the
    teacher trains on D's demonstrations.
    """

    kind: ClassVar[str] = "requential"
    teacher: Teacher = FineTunedTeacher()
    calls: int = 64
    batch: int = 1024
    student_epochs: int = 1
    checkpoints: int = 10

    def run(self, scorer, params0, data, split: Split, learner: Learner, seed, log) -> Requential:
        return requential(scorer, params0, data, split, learner, self, seed=seed, log=log)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "teacher": self.teacher.describe(),
            "calls": self.calls,
            "batch": self.batch,
            "student_epochs": self.student_epochs,
            "checkpoints": self.checkpoints,
        }


def requential(
    scorer: Sampler,
    params0,
    data,
    split: Split,
    learner: Learner,
    coding: RequentialCoding,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Requential:
    """Code the student's path from the base toward the teacher, call by call.

    Each call draws ``coding.batch`` distinct rows of D. Draws are keyed by row
    (the decoder replays them from the row ids), so a row drawn twice in one
    call would get the same sample twice: a batch of copies, not of independent
    draws, whose KL the sum counts twice. A D smaller than the batch is refused.
    """
    train, test = np.asarray(split.train), np.asarray(split.test)
    if len(train) < coding.batch:
        raise ValueError(
            f"a requential call draws {coding.batch} distinct rows of D, which has only {len(train)}: rows drawn "
            "twice in a call get identical samples (draws are keyed by row), so shrink the call batch or grow D"
        )
    student = Trainer(scorer, learner.optimiser, learner.trains)
    state = student.start(params0)
    minibatch = min(student.batch_size, coding.batch)
    steps_per_call = coding.student_epochs * math.ceil(coding.batch / minibatch)
    teacher = coding.teacher.begin(scorer, params0, data, train, steps_per_call, seed * 1000 + 1)
    rng = np.random.default_rng(seed)
    at = set(checkpoint_calls(coding.calls, coding.checkpoints))

    coded, call_kl, inputs, units, truncated = [], [], [], [], []
    student_curve, teacher_curve, seconds, bounds = [], [], [], []
    started = time.perf_counter()

    def checkpoint(calls_done: int) -> None:
        student_params = params0 if calls_done == 0 else student.params(state)
        student_curve.append(example_bits(scorer, student_params, data, test))
        teacher_curve.append(example_bits(scorer, teacher.params(), data, test))
        bounds.append(calls_done * coding.batch)
        seconds.append(time.perf_counter() - started - sum(seconds))
        log(
            f"  t={bounds[-1]:>7}  student {student_curve[-1].mean():8.3f}  teacher {teacher_curve[-1].mean():8.3f} bits  "
            f"KL so far {sum(call_kl):,.1f}  ({sum(seconds):.0f}s)"
        )

    checkpoint(0)
    for call in range(coding.calls):
        rows = np.sort(rng.choice(train, size=coding.batch, replace=len(train) < coding.batch))  # never True: refused above
        drawn = scorer.sample(teacher.params(), data, rows, seed=seed * 1_000_003 + call)
        everything = np.arange(len(rows))
        teacher_bits = example_bits(scorer, teacher.params(), drawn, everything)
        student_bits = example_bits(scorer, student.params(state), drawn, everything)
        kl = student_bits - teacher_bits  # log2 Q(x) - log2 P(x), x ~ Q
        coded.append(kl)
        call_kl.append(float(kl.sum()))
        inputs.append(rows)
        units.append(scorer.units(drawn, everything))
        truncated.append(np.asarray(drawn.truncated, dtype=bool))

        for _ in range(coding.student_epochs):
            order = rng.permutation(len(rows))
            for start in range(0, len(rows), minibatch):
                state = student.step(state, scorer.batch(drawn, np.sort(order[start : start + minibatch])))
        teacher.advance()
        if call + 1 in at:
            checkpoint(call + 1)

    cut = int(np.concatenate(truncated).sum())
    if cut:
        log(
            f"  WARNING: {cut} of {coding.calls * coding.batch} draws were cut short by the sampler and are scored "
            "with an end token that was never drawn: not exact draws from the teacher (diagnostics: draws complete)"
        )
    return Requential(
        bounds=bounds,
        example_bits=np.concatenate(coded),
        call_kl=np.asarray(call_kl),
        test_bits=np.asarray([float(c.mean()) for c in student_curve]),
        test_example_bits=np.stack(student_curve),
        teacher_curve=np.asarray([float(c.mean()) for c in teacher_curve]),
        teacher_example_curve=np.stack(teacher_curve),
        inputs=np.concatenate(inputs),
        units=np.concatenate(units),
        truncated=np.concatenate(truncated),
        seconds=np.asarray(seconds),
    )
