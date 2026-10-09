"""Does a write only *shift* the choice curve, or does it bend it with distance?

The behavioural half of "is the value axis a gain on the input, or a separate
value variable?". Two stories for what a write does to the choice between the
richer and the poorer objective:

* **additive utility** -- the write moves a value term that is combined with
  distance afterwards, ``logit = a(write) + h(d_rich, d_poor)``: the iso-choice
  contours in the ``(d_rich, d_poor)`` plane shift, they do not change shape;
* **gain** -- the write scales how strongly one colour drives a field that also
  carries distance. Whether *that* bends the contours depends on where the
  decision noise sits. If the agent compares ``log phi`` (the ratio model
  ``g_r phi(d_r) > g_p phi(d_p)`` with logistic noise on the log), a gain is a
  constant in log-odds whatever ``phi`` is and is indistinguishable from an
  additive value. If the noise is on the field itself, a gain multiplies
  ``phi(d)`` and its effect shrinks with absolute distance -- a write x
  ``d_near`` interaction at a fixed gap -- and it rescales the curve's slope.

So the test is only as informative as the reference: brightness, which is a gain
by construction (:mod:`goalmisgen.analysis.input_gain`), shows what a gain does
to *this* agent's contours. If brightness is separable too, the test cannot
tell the stories apart; if brightness bends the contours, the question is
whether a written axis at the same threshold bends them the same way.

Everything here works on counts per ``(condition, d_rich, d_poor)`` cell, so a
bootstrap over levels is a recount rather than a refit on episodes. Conditions
decoded with the same seed and the same number of environments visit the same
levels in the same order (each environment's sampler advances once per reset,
whatever the policy does), so rows are *paired* across conditions and one
resample of row indices is applied to all of them.
"""

from __future__ import annotations

import dataclasses

import numpy as np

UNREACHABLE = -1


# --- data --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Decisions:
    """Per episode row: the two distances and whether the richer objective won.

    ``usable`` drops timeouts and walled-off objectives -- the agent faced no
    trade-off there. Rows keep their position so conditions stay paired.
    """

    d_rich: np.ndarray
    d_poor: np.ndarray
    took_rich: np.ndarray
    usable: np.ndarray

    @classmethod
    def from_arrays(cls, d_rich, d_poor, reached, reached_fid, colour_of_rich) -> "Decisions":
        d_rich, d_poor = np.asarray(d_rich, int), np.asarray(d_poor, int)
        reached = np.asarray(reached, bool)
        usable = reached & (d_rich != UNREACHABLE) & (d_poor != UNREACHABLE)
        took = np.asarray(reached_fid) == np.asarray(colour_of_rich)
        return cls(d_rich, d_poor, took & usable, usable)

    def __len__(self) -> int:
        return len(self.d_rich)


@dataclasses.dataclass(frozen=True)
class CellTable:
    """Binomial counts per (condition, d_rich, d_poor)."""

    condition: np.ndarray  # (cells,) int
    d_rich: np.ndarray
    d_poor: np.ndarray
    n: np.ndarray  # trials
    k: np.ndarray  # took richer

    @property
    def gap(self) -> np.ndarray:
        return (self.d_rich - self.d_poor).astype(float)

    @property
    def near(self) -> np.ndarray:
        return np.minimum(self.d_rich, self.d_poor).astype(float)


def tabulate(decisions: list[Decisions], rows: np.ndarray | None = None, max_distance: int = 80) -> CellTable:
    """Count cells for every condition, over ``rows`` (a resample of row indices) if given."""
    side = max_distance + 1
    out = {"condition": [], "d_rich": [], "d_poor": [], "n": [], "k": []}
    for index, dec in enumerate(decisions):
        take = np.arange(len(dec)) if rows is None else rows
        use = dec.usable[take]
        dr, dp, took = dec.d_rich[take][use], dec.d_poor[take][use], dec.took_rich[take][use]
        if (dr > max_distance).any() or (dp > max_distance).any():
            raise ValueError(f"distance above max_distance={max_distance}")
        flat = dr * side + dp
        n = np.bincount(flat, minlength=side * side)
        k = np.bincount(flat, weights=took.astype(float), minlength=side * side)
        cells = np.flatnonzero(n)
        out["condition"].append(np.full(len(cells), index))
        out["d_rich"].append(cells // side)
        out["d_poor"].append(cells % side)
        out["n"].append(n[cells].astype(float))
        out["k"].append(k[cells])
    return CellTable(**{key: np.concatenate(value) for key, value in out.items()})


# --- logistic regression on counts --------------------------------------------


def fit_binomial_logit(x: np.ndarray, n: np.ndarray, k: np.ndarray, ridge: float = 1e-6, iterations: int = 100):
    """Newton-Raphson on grouped binomial data. Returns (weights, deviance).

    A small ridge keeps separable cells (all-or-nothing choices far from the
    threshold) from sending coefficients to infinity; it is negligible wherever
    the data carry information.
    """
    w = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    for _ in range(iterations):
        eta = np.clip(x @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        gradient = x.T @ (k - n * p) - penalty @ w
        hessian = (x * (n * p * (1 - p))[:, None]).T @ x + penalty
        step = np.linalg.solve(hessian, gradient)
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w, binomial_deviance(x @ w, n, k)


def binomial_deviance(eta: np.ndarray, n: np.ndarray, k: np.ndarray) -> float:
    eta = np.clip(eta, -30, 30)
    log_p = -np.logaddexp(0, -eta)
    log_q = -np.logaddexp(0, eta)
    fitted = k * log_p + (n - k) * log_q
    p_hat = np.where(n > 0, k / np.maximum(n, 1), 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturated = np.where(k > 0, k * np.log(p_hat), 0.0) + np.where(n - k > 0, (n - k) * np.log(1 - p_hat), 0.0)
    return float(2 * np.sum(saturated - fitted))


# --- the distance surface h(d_rich, d_poor) -----------------------------------


def hat_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation basis; values outside the knots are clamped."""
    x = np.clip(np.asarray(x, float), knots[0], knots[-1])
    out = np.zeros((len(x), len(knots)))
    for j in range(len(knots)):
        if j > 0:
            left = (x >= knots[j - 1]) & (x <= knots[j])
            out[left, j] = (x[left] - knots[j - 1]) / (knots[j] - knots[j - 1])
        if j < len(knots) - 1:
            right = (x >= knots[j]) & (x < knots[j + 1])
            out[right, j] = (knots[j + 1] - x[right]) / (knots[j + 1] - knots[j])
    out[x == knots[-1], -1] = 1.0
    return out


@dataclasses.dataclass(frozen=True)
class Surface:
    """Knots and centring for ``h``: hats in gap and in d_near, plus gap x d_near."""

    gap_knots: np.ndarray
    near_knots: np.ndarray
    near_centre: float
    gap_centre: float

    @classmethod
    def for_table(cls, table: CellTable, gap_step: float = 2.0, near_step: float = 2.0) -> "Surface":
        gap, near, n = table.gap, table.near, table.n
        g_lo, g_hi = _weighted_quantile(gap, n, [0.005, 0.995])
        n_lo, n_hi = _weighted_quantile(near, n, [0.0, 0.99])
        gap_knots = np.arange(np.floor(g_lo), np.ceil(g_hi) + gap_step, gap_step)
        near_knots = np.arange(np.floor(n_lo), np.ceil(n_hi) + near_step, near_step)
        return cls(gap_knots, near_knots, float(np.average(near, weights=n)), float(np.average(gap, weights=n)))

    def columns(self, gap: np.ndarray, near: np.ndarray) -> np.ndarray:
        """``h`` without a constant (one hat of each basis dropped)."""
        g = hat_basis(gap, self.gap_knots)[:, 1:]
        m = hat_basis(near, self.near_knots)[:, 1:]
        cross = ((np.asarray(gap, float) - self.gap_centre) * (np.asarray(near, float) - self.near_centre))[:, None] / 10.0
        return np.hstack([g, m, cross])


def _weighted_quantile(x, w, qs):
    order = np.argsort(x)
    cumulative = np.cumsum(w[order]) / np.sum(w)
    return [float(x[order][min(np.searchsorted(cumulative, q), len(x) - 1)]) for q in qs]


# --- the models ---------------------------------------------------------------

MODELS = ("separable", "near", "width", "full")
"""``separable``: a(condition) + h.  ``near``: + condition x (d_near - centre).
``width``: + condition x (gap - centre).  ``full``: both interactions."""


@dataclasses.dataclass
class Fit:
    model: str
    reference: int
    n_conditions: int
    surface: Surface
    weights: np.ndarray
    deviance: float
    n_params: int

    def _split(self):
        c = self.n_conditions - 1
        w = self.weights
        h_size = len(self.weights) - 1 - c - c * (self.model in ("near", "full")) - c * (self.model in ("width", "full"))
        at = 1 + c + h_size
        near = np.zeros(c)
        slope = np.zeros(c)
        if self.model in ("near", "full"):
            near, at = w[at : at + c], at + c
        if self.model in ("width", "full"):
            slope = w[at : at + c]
        return w[0], w[1 : 1 + c], w[1 + c : 1 + c + h_size], near, slope

    def _others(self):
        return [i for i in range(self.n_conditions) if i != self.reference]

    def logit(self, condition: int, gap, near) -> np.ndarray:
        gap, near = np.atleast_1d(np.asarray(gap, float)), np.atleast_1d(np.asarray(near, float))
        intercept, shifts, h, near_w, slope_w = self._split()
        eta = intercept + self.surface.columns(gap, near) @ h
        if condition != self.reference:
            j = self._others().index(condition)
            eta = eta + shifts[j] + near_w[j] * (near - self.surface.near_centre) + slope_w[j] * (gap - self.surface.gap_centre)
        return eta

    def interaction(self, condition: int) -> tuple[float, float]:
        """(condition x d_near, condition x gap) coefficients in log-odds per step; zero for the reference."""
        if condition == self.reference:
            return 0.0, 0.0
        _, _, _, near_w, slope_w = self._split()
        j = self._others().index(condition)
        return float(near_w[j]), float(slope_w[j])

    def contour(self, condition: int, nears, gaps=np.arange(-45, 45.01, 0.05)) -> np.ndarray:
        """Threshold (the 50% gap) at each ``d_near``: one iso-choice contour. NaN where none is crossed."""
        out = []
        for near in np.atleast_1d(nears):
            eta = self.logit(condition, gaps, np.full(len(gaps), near))
            out.append(_first_crossing(gaps, eta, 0.0))
        return np.array(out)

    def width(self, condition: int, near: float) -> float:
        """q75 - q25 in steps at one ``d_near``, from the model's slope through the threshold."""
        gaps = np.arange(-45, 45.01, 0.05)
        eta = self.logit(condition, gaps, np.full(len(gaps), near))
        high, low = _first_crossing(gaps, eta, np.log(3)), _first_crossing(gaps, eta, -np.log(3))
        return float(abs(low - high))


def _first_crossing(x, y, level):
    s = y - level
    idx = np.flatnonzero(np.sign(s[:-1]) * np.sign(s[1:]) <= 0)
    idx = [i for i in idx if s[i] != s[i + 1]]
    if not idx:
        return float("nan")
    i = idx[0]
    return float(x[i] + s[i] / (s[i] - s[i + 1]) * (x[i + 1] - x[i]))


def fit_model(table: CellTable, model: str, reference: int = 0, surface: Surface | None = None, ridge: float = 1e-6) -> Fit:
    if model not in MODELS:
        raise ValueError(f"model {model!r} not in {MODELS}")
    surface = surface or Surface.for_table(table)
    conditions = int(table.condition.max()) + 1
    others = [i for i in range(conditions) if i != reference]
    dummies = np.stack([(table.condition == i).astype(float) for i in others], axis=1) if others else np.zeros((len(table.n), 0))
    parts = [np.ones((len(table.n), 1)), dummies, surface.columns(table.gap, table.near)]
    if model in ("near", "full"):
        parts.append(dummies * (table.near - surface.near_centre)[:, None])
    if model in ("width", "full"):
        parts.append(dummies * (table.gap - surface.gap_centre)[:, None])
    x = np.hstack(parts)
    w, deviance = fit_binomial_logit(x, table.n, table.k, ridge=ridge)
    return Fit(model, reference, conditions, surface, w, deviance, x.shape[1])


def subset(decisions: list[Decisions], indices) -> list[Decisions]:
    return [decisions[i] for i in indices]


# --- the base's field shape under the ratio model ----------------------------


@dataclasses.dataclass(frozen=True)
class FieldShape:
    """``logit = c + psi(d_rich) - psi(d_poor)``, psi free per distance, against psi linear.

    psi is ``log phi`` in units of the decision noise; psi linear is an
    exponential field; distances past ``cap`` (the last well-populated one) share
    one psi in the free fit. ``max_departure_steps`` is how far, in steps of distance,
    the free psi strays from its best straight line over well-populated
    distances -- the scale on which "close enough to exponential" is judged.
    """

    distances: np.ndarray
    psi: np.ndarray
    counts: np.ndarray
    intercept: float
    slope: float  # psi per step under the linear fit (negative: the field decays)
    deviance_free: float
    deviance_linear: float
    df: int
    max_departure_steps: float
    ratio_vs_surface_deviance: float  # free-psi ratio model minus the h surface; NOT nested, compare by AIC
    ratio_vs_surface_df: int  # surface parameters minus ratio-model parameters


def field_shape(table: CellTable, condition: int = 0, min_count: float = 200.0, cap: int | None = None) -> FieldShape:
    keep = table.condition == condition
    dr, dp, n, k = table.d_rich[keep], table.d_poor[keep], table.n[keep], table.k[keep]
    counts = np.bincount(dr, weights=n, minlength=1) + 0.0
    counts_p = np.bincount(dp, weights=n, minlength=len(counts))
    size = max(len(counts), len(counts_p))
    total = np.zeros(size)
    total[: len(counts)] += counts
    total[: len(counts_p)] += counts_p
    if cap is None:
        populated = np.flatnonzero(total >= min_count)
        cap = int(populated.max()) if len(populated) else int(size - 1)
    dr_c, dp_c = np.minimum(dr, cap), np.minimum(dp, cap)
    distances = np.arange(0, cap + 1)
    present = np.unique(np.r_[dr_c, dp_c])
    lowest = int(present.min())
    free = [d for d in distances if d > lowest and d in present]
    x_free = np.stack([(dr_c == d).astype(float) - (dp_c == d).astype(float) for d in free], axis=1)
    x_free = np.hstack([np.ones((len(n), 1)), x_free])
    w_free, dev_free = fit_binomial_logit(x_free, n, k)
    x_lin = np.stack([np.ones(len(n)), (dr - dp).astype(float)], axis=1)  # uncapped: the line needs no pooling
    w_lin, dev_lin = fit_binomial_logit(x_lin, n, k)
    psi = np.full(len(distances), np.nan)
    psi[lowest] = 0.0
    for d, value in zip(free, w_free[1:]):
        psi[d] = value
    weight = np.array([total[d] if d < len(total) else 0.0 for d in distances])
    ok = np.isfinite(psi) & (weight >= min_count)
    # best straight line through the free psi, weighted by how often each distance occurs
    a = np.stack([np.ones(ok.sum()), distances[ok]], axis=1)
    coef = np.linalg.lstsq(a * np.sqrt(weight[ok])[:, None], psi[ok] * np.sqrt(weight[ok]), rcond=None)[0]
    departure = psi[ok] - a @ coef
    slope_for_steps = abs(coef[1]) if abs(coef[1]) > 1e-9 else np.nan
    single = CellTable(np.zeros(len(n), int), dr, dp, n, k)
    surface_fit = fit_model(single, "separable", reference=0)
    return FieldShape(
        distances=distances,
        psi=psi,
        counts=weight,
        intercept=float(w_lin[0]),
        slope=float(w_lin[1]),
        deviance_free=dev_free,
        deviance_linear=dev_lin,
        df=len(free) - 1,
        max_departure_steps=float(np.max(np.abs(departure)) / slope_for_steps),
        ratio_vs_surface_deviance=dev_free - surface_fit.deviance,
        ratio_vs_surface_df=surface_fit.n_params - x_free.shape[1],
    )


# --- nonparametric contours ---------------------------------------------------


def banded_crossings(table: CellTable, condition: int, bands: list[tuple[int, int]], min_n: float = 100.0) -> np.ndarray:
    """Per ``d_near`` band, the 50% gap from a two-parameter logistic in gap. No shared surface."""
    out = []
    for low, high in bands:
        keep = (table.condition == condition) & (table.near >= low) & (table.near <= high)
        if table.n[keep].sum() < min_n or table.k[keep].sum() in (0, table.n[keep].sum()):
            out.append(np.nan)
            continue
        x = np.stack([np.ones(keep.sum()), table.gap[keep]], axis=1)
        w, _ = fit_binomial_logit(x, table.n[keep], table.k[keep], ridge=1e-8)
        out.append(-w[0] / w[1] if abs(w[1]) > 1e-9 else np.nan)
    return np.array(out)


def bootstrap(decisions: list[Decisions], statistic, resamples: int = 200, seed: int = 0, max_distance: int = 80, paired: bool = True):
    """Resamples of row indices, recounted for every condition; returns the statistics stacked.

    ``paired``: one resample shared by all conditions, which must then have the
    same rows (the same levels in the same order). Unpaired conditions are
    resampled independently -- honest, but differences get no pairing benefit.
    """
    rows = [len(d) for d in decisions]
    if paired and len(set(rows)) != 1:
        raise ValueError("conditions are not paired: row counts differ")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(resamples):
        if paired:
            table = tabulate(decisions, rng.integers(0, rows[0], rows[0]), max_distance=max_distance)
        else:
            parts = [tabulate([d], rng.integers(0, len(d), len(d)), max_distance=max_distance) for d in decisions]
            table = CellTable(
                np.concatenate([np.full(len(p.n), i) for i, p in enumerate(parts)]),
                *(np.concatenate([getattr(p, f) for p in parts]) for f in ("d_rich", "d_poor", "n", "k")),
            )
        out.append(np.asarray(statistic(table), float))
    return np.stack(out)


def paired(decisions: list[Decisions]) -> bool:
    """Whether every condition saw the same levels row for row (distances identical)."""
    first = decisions[0]
    return all(
        len(d) == len(first) and np.array_equal(d.d_rich, first.d_rich) and np.array_equal(d.d_poor, first.d_poor)
        for d in decisions[1:]
    )
