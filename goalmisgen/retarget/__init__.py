"""Retargeting cost: how many bits it takes to move a trained agent onto a new goal.

The metric is assembled from parts, one module each, so that a study can change
one part (a learner, a reference, an environment, a model) without touching the
others:

``family``    what a goal is: demonstrations of the expert at goal ``g`` on given
              levels, the null goal (the agent's own), and which levels leave the
              expert deterministic (:class:`GoalFamily`)
``scoring``   how a model scores data: bits per example and the training loss
              (:class:`Scorer`)
``learners``  the learner A: which variables it trains (every weight, one tensor,
              a subspace of weight space), its optimiser and its stopping rule
``code``      the :class:`Coding` and :class:`CodeResult` protocols every coding
              mechanism implements, and the block-prequential code
``requential`` requential coding: a teacher picks which of the student's own
              draws it trains on, and the code is their cumulative KL
``splits``    one split of a level pool into the stream D, validation V and test T,
              shared by every arm of a study
``spec``      what defines a run, and the record a run writes
``metric``    what a study reports from records: references (null arm, oracle),
              normalisations (per route, per token, the ceiling), the fraction of
              the behaviour change paid, and the competence gate
``diagnostics`` warning lights on every run and study: competence, whether the
              coded model (and a teacher) learned and settled, drift on levels the
              goal leaves alone, overhead, unpaid information, noise
``run``       :func:`measure`, which strings the parts together for one run

Nothing here knows about mazes or route models. The maze family and the route
model's scorer are in :mod:`goalmisgen.offline.retarget`; another environment or
model implements :class:`GoalFamily` and :class:`Scorer` there and reuses the rest.
"""

from goalmisgen.retarget import diagnostics, metric
from goalmisgen.retarget.code import CodeResult, Coding, Prequential, PrequentialCoding, doubling_blocks, prequential
from goalmisgen.retarget.family import GoalFamily
from goalmisgen.retarget.learners import (
    AllWeights,
    EarlyStopping,
    Fit,
    InSubspace,
    Learner,
    OneTensor,
    Optimiser,
    Parameterisation,
    Subspace,
    make_fitter,
)
from goalmisgen.retarget.requential import EMA, FineTunedTeacher, FixedTeacher, Requential, RequentialCoding, Teacher
from goalmisgen.retarget.run import ArmData, arm_data, measure
from goalmisgen.retarget.scoring import Sampler, Scorer, example_bits
from goalmisgen.retarget.spec import Record, RunSpec
from goalmisgen.retarget.splits import Split, study_split

__all__ = [
    "Coding",
    "EMA",
    "FineTunedTeacher",
    "FixedTeacher",
    "PrequentialCoding",
    "Requential",
    "RequentialCoding",
    "Sampler",
    "Teacher",
    "AllWeights",
    "ArmData",
    "CodeResult",
    "EarlyStopping",
    "Fit",
    "GoalFamily",
    "InSubspace",
    "Learner",
    "OneTensor",
    "Optimiser",
    "Parameterisation",
    "Prequential",
    "Record",
    "RunSpec",
    "Scorer",
    "Split",
    "Subspace",
    "arm_data",
    "diagnostics",
    "doubling_blocks",
    "example_bits",
    "make_fitter",
    "measure",
    "metric",
    "prequential",
    "study_split",
]
