"""The learner A: what it trains, how it steps, and when it stops.

A learner has three parts, each swappable without touching the others:

- a :class:`Parameterisation` says which variables are trained and how they make
  a full parameter tree: every weight (:class:`AllWeights`), one top-level tensor
  (:class:`OneTensor`), or coefficients along fixed directions in weight space
  (:class:`InSubspace`). Restricting the learner asks whether a cheap edit
  *exists* in that part of the model, which full-weight SGD may or may not find
  (THREADS.md RC.7);
- an :class:`Optimiser`: clipped AdamW with a linear warm-up;
- a stopping rule, :class:`EarlyStopping`: validate at log-spaced steps, stop
  after ``patience`` validations without improvement, and return the best point
  seen, the starting point included.

The early-stopping rule selects on V, data the prequential code does not pay
for. What that selection can carry is bounded by ``log2`` of the number of
candidates per fit (THREADS.md RC.4); a rule that selects on held-out past data,
or a fixed step budget, would be another implementation of the same interface.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from goalmisgen.retarget.scoring import Scorer, example_bits

# ----------------------------------------------------------------------
# What the learner trains
# ----------------------------------------------------------------------


class Parameterisation(Protocol):
    """The trained variables, and the parameter tree they stand for.

    ``context`` holds whatever arrays ``materialise`` needs besides the
    variables (the base's weights, a set of directions). It is passed to the
    compiled step as an argument rather than closed over, so one compilation
    serves every fit from the same base.
    """

    def context(self, params0) -> Any:
        ...

    def initial(self, params0, context) -> Any:
        """The variables at the base: materialising them gives ``params0`` exactly."""
        ...

    def materialise(self, variables, context) -> Any:
        """The full parameter tree. Pure and jit-able."""
        ...

    def record(self, variables) -> np.ndarray | None:
        """What a result keeps of the chosen variables (a subspace's coefficients), or ``None``."""
        ...

    def describe(self) -> dict:
        """JSON for a run's record. ``kind`` and ``name`` identify it; anything else is detail."""
        ...


@dataclasses.dataclass(frozen=True)
class AllWeights:
    """Every weight: the spec's learner."""

    def context(self, params0) -> None:
        return None

    def initial(self, params0, context):
        return params0

    def materialise(self, variables, context):
        return variables

    def record(self, variables) -> None:
        return None

    def describe(self) -> dict:
        return {"kind": "all"}


@dataclasses.dataclass(frozen=True)
class OneTensor:
    """One top-level parameter (e.g. ``goal_embedding``); every other weight stays the base's."""

    name: str

    def context(self, params0):
        if self.name not in params0["params"]:
            raise ValueError(f"the model has no parameter {self.name!r} to train")
        return params0

    def initial(self, params0, context):
        return params0["params"][self.name]

    def materialise(self, variables, context):
        return {**context, "params": {**context["params"], self.name: variables}}

    def record(self, variables) -> None:
        return None

    def describe(self) -> dict:
        return {"kind": "tensor", "name": self.name}


@dataclasses.dataclass(frozen=True, eq=False)
class Subspace:
    """``k`` orthonormal directions over the flat parameter vector.

    Rows are orthonormal (:meth:`spanning`), so a coefficient is in units of
    parameter-space distance whatever the directions were fitted in, and one
    learning rate means the same step along any of them. The flat order is
    :func:`jax.flatten_util.ravel_pytree`'s, the order every weight-diff tool in
    the repo uses (``offline/axis.py``).
    """

    directions: np.ndarray
    """``(k, P)`` float32, orthonormal rows."""

    name: str = ""
    meta: dict = dataclasses.field(default_factory=dict)
    """How the directions were made; carried into every result that uses them."""

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions, dtype=np.float32)
        if directions.ndim != 2 or directions.shape[0] < 1:
            raise ValueError(f"directions must be (k, P) with k >= 1, got {directions.shape}")
        gram = directions.astype(np.float64) @ directions.T.astype(np.float64)
        if not np.allclose(gram, np.eye(len(directions)), atol=1e-4):
            raise ValueError("directions must be orthonormal; build them with Subspace.spanning")
        object.__setattr__(self, "directions", directions)

    @classmethod
    def spanning(cls, vectors: np.ndarray, name: str = "", meta: dict | None = None) -> "Subspace":
        """The orthonormal basis of ``span(vectors)``, in their order.

        Gram-Schmidt order and sign: the first direction is ``vectors[0]`` made
        unit, so a positive coefficient on it means "more of the first vector".
        """
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        q, r = np.linalg.qr(vectors.T)
        if np.any(np.abs(np.diag(r)) < 1e-9 * max(np.abs(r).max(), 1e-30)):
            raise ValueError("the vectors are linearly dependent")
        q = q * np.sign(np.diag(r))
        return cls(q.T.astype(np.float32), name=name, meta=dict(meta or {}))

    @property
    def k(self) -> int:
        return int(self.directions.shape[0])

    @property
    def size(self) -> int:
        return int(self.directions.shape[1])

    def point(self, params0, coefficients) -> dict:
        """``theta0 + sum_k c_k u_k`` as a parameter tree; ``c = 0`` returns ``theta0``'s values exactly."""
        flat0, unravel = ravel_pytree(params0)
        self.check(flat0)
        offset = jnp.asarray(coefficients, dtype=flat0.dtype) @ jnp.asarray(self.directions, dtype=flat0.dtype)
        return unravel(flat0 + offset)

    def check(self, flat0) -> None:
        if flat0.shape[0] != self.size:
            raise ValueError(f"subspace {self.name!r} spans {self.size:,} parameters; the model has {flat0.shape[0]:,}")

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            np.savez(handle, directions=self.directions, name=self.name, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: pathlib.Path) -> "Subspace":
        with np.load(pathlib.Path(path)) as stored:
            return cls(stored["directions"], name=str(stored["name"]), meta=json.loads(str(stored["meta"])))


@dataclasses.dataclass(frozen=True, eq=False)
class InSubspace:
    """``theta = theta0 + sum_k c_k u_k``: the ``k`` coefficients are the only variables.

    ``c = 0`` is the base exactly, and every weight moves in lockstep with the
    coefficients. ``path`` is where the directions were loaded from, for the
    record.
    """

    subspace: Subspace
    path: str | None = None
    _unravel: dict = dataclasses.field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: pathlib.Path) -> "InSubspace":
        return cls(Subspace.load(path), path=str(path))

    def context(self, params0):
        flat0, unravel = ravel_pytree(params0)
        self.subspace.check(flat0)
        # unravel depends only on the tree's structure; keep the first, so the
        # compiled step that closes over it stays valid for every fit.
        self._unravel.setdefault("fn", unravel)
        return flat0, jnp.asarray(self.subspace.directions, dtype=flat0.dtype)

    def initial(self, params0, context):
        flat0, _ = context
        return jnp.zeros(self.subspace.k, dtype=flat0.dtype)

    def materialise(self, variables, context):
        flat0, directions = context
        return self._unravel["fn"](flat0 + variables @ directions)

    def record(self, variables) -> np.ndarray:
        return np.asarray(variables, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "kind": "subspace",
            "name": self.subspace.name,
            "k": self.subspace.k,
            "path": self.path,
            "meta": self.subspace.meta,
        }


# ----------------------------------------------------------------------
# How it steps, and when it stops
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Optimiser:
    """Clipped AdamW, linear warm-up to a constant rate.

    Batch, weight decay and clipping are the route model's value-axis arm
    recipe. The rate is not: the arms' 1e-4 drifts a converged base faster than
    a few hundred demonstrations can teach it anything, and 3e-5 was the best of
    3e-6..1e-4 on a rate outside every arm grid
    (``scripts/studies/retarget/pick_lr.py``).
    """

    learning_rate: float = 3e-5
    batch_size: int = 256
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    warmup_steps: int = 20

    def build(self) -> optax.GradientTransformation:
        warmup = max(self.warmup_steps, 1)
        schedule = optax.join_schedules(
            [optax.linear_schedule(0.0, self.learning_rate, warmup), optax.constant_schedule(self.learning_rate)],
            [warmup],
        )
        return optax.chain(optax.clip_by_global_norm(self.clip_norm), optax.adamw(schedule, weight_decay=self.weight_decay))


@dataclasses.dataclass(frozen=True)
class EarlyStopping:
    """Validate at log-spaced steps from ``eval_first``; stop after ``patience`` without improvement.

    Log rather than linear spacing because the best step is early and its
    position matters: a fresh Adam on a converged base moves every weight by
    about the learning rate per step, so the held-out loss usually falls for
    tens of steps and then climbs as that drift accumulates (the pilot found
    unchanged levels going from 0.26 to 1.0 bits per route within 100 steps at
    1e-4). ``max_steps`` bounds a fit.
    """

    eval_first: int = 5
    eval_ratio: float = 1.25
    patience: int = 6
    max_steps: int = 2000

    def eval_steps(self) -> list[int]:
        steps, step = set(), float(self.eval_first)
        while step < self.max_steps:
            steps.add(int(round(step)))
            step *= self.eval_ratio
        return sorted(steps | {self.max_steps})

    @property
    def candidates(self) -> int:
        """How many points a fit chooses among (the base included): the selection's side channel is at most log2 of this."""
        return len(self.eval_steps()) + 1


@dataclasses.dataclass(frozen=True)
class Learner:
    optimiser: Optimiser = Optimiser()
    stopping: EarlyStopping = EarlyStopping()
    trains: Parameterisation = AllWeights()

    def to_dict(self) -> dict:
        return {
            "optimiser": dataclasses.asdict(self.optimiser),
            "stopping": dataclasses.asdict(self.stopping),
            "trains": self.trains.describe(),
        }


@dataclasses.dataclass
class Fit:
    params: Any
    best_step: int
    steps_run: int
    valid_bits: float
    """Mean bits per example on the validation set, at the chosen step."""

    recorded: np.ndarray | None = None
    """What the parameterisation records of the chosen variables (a subspace's ``c``; zeros when the base won)."""


@dataclasses.dataclass
class TrainState:
    variables: Any
    opt_state: Any
    context: Any


class Trainer:
    """One model under training: its parameterisation, its optimiser and their compiled step.

    The one training step in the package: early-stopped fits, a requential
    student and a requential teacher all step through it, so a learner means
    the same thing under every coding mechanism.
    """

    def __init__(self, scorer: Scorer, optimiser: Optimiser, trains: Parameterisation) -> None:
        self.trains = trains
        self.batch_size = optimiser.batch_size
        transform = optimiser.build()
        self._transform = transform

        @jax.jit
        def step(variables, opt_state, context, *batch):
            loss, grads = jax.value_and_grad(lambda v: scorer.loss(trains.materialise(v, context), *batch))(variables)
            updates, opt_state = transform.update(grads, opt_state, variables)
            return optax.apply_updates(variables, updates), opt_state, loss

        self._step = step
        self._materialise = jax.jit(trains.materialise)

    def start(self, params0) -> TrainState:
        context = self.trains.context(params0)
        variables = self.trains.initial(params0, context)
        return TrainState(variables, self._transform.init(variables), context)

    def step(self, state: TrainState, batch: tuple) -> TrainState:
        variables, opt_state, _ = self._step(state.variables, state.opt_state, state.context, *batch)
        return TrainState(variables, opt_state, state.context)

    def params(self, state: TrainState):
        return self._materialise(state.variables, state.context)

    def record(self, state: TrainState) -> np.ndarray | None:
        return self.trains.record(state.variables)


Fitter = Callable[..., Fit]


def make_fitter(scorer: Scorer, learner: Learner) -> Fitter:
    """``fit(params0, data, train, valid, seed) -> Fit``: one early-stopped fit from ``params0``.

    Batches are drawn without replacement within a step, from ``train`` only,
    by a generator seeded with ``seed``. The base, at step 0, is always a
    candidate, so a fit never scores worse on V than where it started.
    """
    stopping = learner.stopping
    trainer = Trainer(scorer, learner.optimiser, learner.trains)

    def fit(params0, data, train: np.ndarray, valid: np.ndarray, seed: int) -> Fit:
        train = np.asarray(train)
        start = float(example_bits(scorer, params0, data, valid).mean())
        state = trainer.start(params0)
        at_start = trainer.record(state)
        if len(train) == 0:
            return Fit(params0, 0, 0, start, at_start)
        rng = np.random.default_rng(seed)
        batch = min(trainer.batch_size, len(train))
        best_bits, best_step, best_params, best_record = start, 0, params0, at_start
        since_best = 0
        steps_run = 0
        evaluate_at = set(stopping.eval_steps())
        for n in range(1, stopping.max_steps + 1):
            index = np.sort(rng.choice(train, size=batch, replace=False))
            state = trainer.step(state, scorer.batch(data, index))
            steps_run = n
            if n in evaluate_at:
                params = trainer.params(state)
                score = float(example_bits(scorer, params, data, valid).mean())
                if score < best_bits:
                    best_bits, best_step, best_params, best_record = score, n, params, trainer.record(state)
                    since_best = 0
                else:
                    since_best += 1
                    if since_best >= stopping.patience:
                        break
        return Fit(best_params, best_step, steps_run, best_bits, best_record)

    return fit
