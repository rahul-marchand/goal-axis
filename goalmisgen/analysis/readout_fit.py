"""The decision variable at the agent's cell, fitted rather than assumed.

:mod:`goalmisgen.analysis.field_decomp` reads each objective's field with
:func:`~goalmisgen.analysis.field_decomp.cell_norms`, the length of the
activation attributable to that objective. At the agent's own cell the
difference of those lengths tracks the level's distance gap and crosses zero
near the measured threshold, which is what a planner comparing two fields
should do -- but a length throws away *which way* the field points, and the
comparison the network actually performs is presumably a projection onto some
direction of the cell state, not its norm.

So fit the direction. With ``A_c`` the field of objective ``c`` and ``p`` the
agent's own cell,

    x = A_rich[p] - A_poor[p]                  (K,) per level
    s = w . x                                  the readout under direction w

and ``w`` is fitted to predict the model's own choice on a training split of
levels. The **no-intercept** logistic is the one to read: its decision boundary
is ``w . A_rich = w . A_poor``, i.e. "the two fields project equally", which is
the same claim the norm makes and keeps the zero crossing a genuine prediction
rather than a free parameter. A fit with an intercept can place its boundary
anywhere, so it is reported only as an upper bound on what any linear readout at
this cell could achieve.

Three controls decide whether a good fit means anything: random unit directions
give the floor, a fit on shuffled labels shows the procedure cannot manufacture
a crossing, and a direction fitted at one site and evaluated at another says
whether one direction serves the whole recurrence.
"""

from __future__ import annotations

import numpy as np


def agent_vectors(field: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """``(N, S, S, K)`` and ``(N, 2)`` -> ``(N, K)``: the field at each level's own cell."""
    field = np.asarray(field)
    rows = np.arange(len(field))
    return np.asarray(field[rows, cells[:, 0], cells[:, 1]], dtype=np.float64)


def unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` scaled to unit length; unchanged if it is all but zero."""
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) / norm if norm > 1e-12 else np.asarray(vector, dtype=np.float64)


def logistic_direction(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3, intercept: bool = False,
                       iterations: int = 100) -> tuple[np.ndarray, float]:
    """IRLS logistic fit of ``y`` on ``x``; returns a unit direction and its bias.

    The direction is normalised afterwards, so the bias is on the scale of the
    unnormalised fit and is reported only for the with-intercept variant. A
    ridge keeps the 32 channels from being fitted at machine precision when the
    classes separate.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([x, np.ones(len(x))]) if intercept else x
    beta = np.zeros(design.shape[1])
    penalty = ridge * np.eye(design.shape[1])
    if intercept:
        penalty[-1, -1] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
        weights = np.clip(p * (1 - p), 1e-6, None)
        gram = (design * weights[:, None]).T @ design + penalty
        gradient = design.T @ (y - p) - penalty @ beta
        try:
            step = np.linalg.solve(gram, gradient)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    direction = beta[:-1] if intercept else beta
    bias = float(beta[-1]) if intercept else 0.0
    length = float(np.linalg.norm(direction))
    return unit(direction), bias / length if length > 1e-12 else bias


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve, by the rank identity; ties count a half."""
    score, label = np.asarray(score, dtype=np.float64), np.asarray(label).astype(bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average the ranks of tied scores, so a constant score scores 0.5
    sorted_scores = score[order]
    start = 0
    for end in range(1, len(score) + 1):
        if end == len(score) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def crossing_in_gap(score: np.ndarray, gap: np.ndarray) -> dict:
    """Where the readout's zero sits in distance-gap units.

    Least squares of the readout on the gap, then the gap at which the fitted
    line is zero. Under "the agent takes whichever field is stronger", that gap
    is the behavioural threshold.
    """
    score, gap = np.asarray(score, dtype=np.float64), np.asarray(gap, dtype=np.float64)
    if len(score) < 10 or np.std(gap) < 1e-9 or np.std(score) < 1e-12:
        return {"slope": float("nan"), "crossing": float("nan"), "corr": float("nan")}
    slope, intercept = np.polyfit(gap, score, 1)
    return {"slope": float(slope), "corr": float(np.corrcoef(gap, score)[0, 1]),
            "crossing": float(-intercept / slope) if abs(slope) > 1e-12 else float("nan")}


def evaluate_readout(score: np.ndarray, took_rich: np.ndarray, gap: np.ndarray) -> dict:
    """The three numbers every candidate readout is judged by, on held-out levels."""
    out = crossing_in_gap(score, gap)
    out["auc"] = auc(score, took_rich)
    out["n"] = int(len(score))
    return out


def random_directions(n: int, width: int, seed: int = 0) -> np.ndarray:
    """``(n, width)`` unit vectors, the floor for any fitted direction."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(n, width))
    return draws / np.linalg.norm(draws, axis=1, keepdims=True)


def norm_readout(a_rich: np.ndarray, a_poor: np.ndarray) -> np.ndarray:
    """The log length ratio field_decomp uses: ``ln|A_rich| - ln|A_poor|`` per level."""
    rich = np.linalg.norm(np.asarray(a_rich, dtype=np.float64), axis=-1)
    poor = np.linalg.norm(np.asarray(a_poor, dtype=np.float64), axis=-1)
    with np.errstate(divide="ignore"):
        return np.log(rich) - np.log(poor)
