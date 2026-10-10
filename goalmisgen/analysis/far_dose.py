"""Past the fitted window: does a written axis keep acting like the gain it matches?

Inside the window the o1 axis's threshold equals that of a colour-1 brightness
``g(dv)``, and brightness moves the threshold as ``tau = a - b ln g``. If the axis
*is* a gain, ``g`` is roughly linear in the written offset -- ``g = 1 + k dv`` --
so its threshold should fall only logarithmically in the dose,

    tau(dv) = a - b ln(1 + k dv),

and reach parity late. Where the axis leaves that curve, and whether it does so
exactly where the agent stops reaching objectives, separates "a gain written
too far" from "something that is not a gain".

The helpers here are the arithmetic of that comparison; decoding lives in
``scripts/studies/gain-vs-factorised/axis_far_dose.py`` and the tables in ``scripts/studies/gain-vs-factorised/axis_far_dose_report.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def interpolate_x(points: list[tuple[float, float]], target: float) -> float | None:
    """The ``x`` at which ``y`` first crosses ``target`` along ``points`` sorted by ``x``; None if it never does.

    Piecewise linear between neighbours. Non-finite ``y`` are dropped.
    """
    points = sorted((float(x), float(y)) for x, y in points if np.isfinite(y))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            return x0 + (target - y0) / (y1 - y0) * (x1 - x0)
    return None


@dataclasses.dataclass(frozen=True)
class LogLaw:
    """``tau = a - b ln g``."""

    a: float
    b: float
    rms: float

    def tau(self, gain) -> np.ndarray:
        return self.a - self.b * np.log(np.asarray(gain, float))

    def gain(self, tau) -> np.ndarray:
        return np.exp((self.a - np.asarray(tau, float)) / self.b)


def fit_log_law(gains, taus) -> LogLaw:
    """Least squares of threshold on log gain over the finite points."""
    g, t = np.asarray(gains, float), np.asarray(taus, float)
    ok = np.isfinite(g) & np.isfinite(t) & (g > 0)
    if ok.sum() < 2:
        raise ValueError("need two finite points to fit a log law")
    x = np.stack([np.ones(ok.sum()), -np.log(g[ok])], axis=1)
    (a, b), *_ = np.linalg.lstsq(x, t[ok], rcond=None)
    rms = float(np.sqrt(np.mean((x @ np.array([a, b]) - t[ok]) ** 2)))
    return LogLaw(float(a), float(b), rms)


def fit_linear_gain(offsets, gains) -> float:
    """``k`` in ``g = 1 + k dv``, least squares through the base (``dv = 0``, ``g = 1``)."""
    dv, g = np.asarray(offsets, float), np.asarray(gains, float)
    ok = np.isfinite(dv) & np.isfinite(g)
    if not ok.any() or not (dv[ok] ** 2).sum():
        raise ValueError("need a nonzero offset with a finite gain")
    return float(dv[ok] @ (g[ok] - 1) / (dv[ok] @ dv[ok]))


def random_direction(size: int, norm: float, rng: np.random.Generator) -> np.ndarray:
    """An isotropic Gaussian direction scaled to ``norm``."""
    v = rng.standard_normal(size)
    return v * (norm / np.linalg.norm(v))


def module_matched_direction(reference: np.ndarray, groups: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Random inside every module, each module's slice as long as ``reference``'s there.

    Parameters in no group keep ``reference``'s slice length too, as one extra block,
    so the result's total norm equals ``reference``'s exactly. See
    :func:`goalmisgen.analysis.plan_invariance.module_matched_random` for how
    the plan-invariance control differs.
    """
    reference = np.asarray(reference, float)
    out = np.zeros_like(reference)
    covered = np.zeros(len(reference), bool)
    blocks = list(groups.values())
    for rows in blocks:
        covered[rows] = True
    rest = np.flatnonzero(~covered)
    if len(rest):
        blocks.append(rest)
    for rows in blocks:
        length = np.linalg.norm(reference[rows])
        if length > 0:
            out[rows] = random_direction(len(rows), length, rng)
    return out


def first_departure(offsets, observed, low, high, predicted, tolerance: float) -> float | None:
    """Smallest offset whose interval, widened by ``tolerance``, excludes the prediction; None if none does."""
    for dv, lo, hi, pred in sorted(zip(offsets, low, high, predicted)):
        if np.isfinite(pred) and np.isfinite(lo) and np.isfinite(hi) and not (lo - tolerance <= pred <= hi + tolerance):
            return float(dv)
    return None
