"""The cheapest edit that retargets a network: the loss, the edit families, the optimiser.

The fine-tuned value axis ``v`` answers "which direction did SGD take when the
reward changed" -- a fact about a training process. This module answers "what is
the smallest change to *this* network that makes it choose like an agent with a
different exchange rate" -- a fact about the network. Only the edit is trained;
the base parameters stay frozen, and the edit is written at several doses at
once:

    theta_j = theta_base + shift_j * delta(phi),     target threshold tau_base + shift_j

with ``shift_j`` in steps of threshold and ``phi`` the edit's own parameters.
If one ``delta`` with linear dosing serves every shift, the network has a
one-parameter slot for the threshold; if the search cannot find one at a budget,
that is weaker evidence -- this is an upper-bound estimator, like DAS.

**Edit families** set the budget, and the budget is what keeps the search from
finding an adversarial perturbation: a dense edit restricted to some modules, a
low-rank factor per kernel, or two scalar gains on the input kernels' colour
planes (the brightness knob, :mod:`goalmisgen.analysis.input_gain`). Each family
maps its parameters to a full-length ``delta`` over the flattened network, so
everything downstream -- writing, cosines against ``v`` -- is family-agnostic.

**The metric is part of the answer.** "Cheapest" means smallest under the
penalty, and the penalty here is L2 in raw parameter coordinates, which is not
invariant to how layers are scaled. The function-space comparison in
:mod:`goalmisgen.edits.compare` is the companion that does not depend on it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from goalmisgen.analysis.route_targets import BRANCH, FORK, PREFIX, RICH

# --- the loss ------------------------------------------------------------------


def _set_log_prob(log_probs, mask):
    """log sum_{a in mask} pi(a), and 0 where the mask is empty (those steps carry no weight)."""
    masked = jnp.where(mask, log_probs, -jnp.inf)
    any_move = mask.any(-1)
    return jnp.where(any_move, jax.nn.logsumexp(jnp.where(any_move[..., None], masked, 0.0), axis=-1), 0.0)


def route_loss(logits, toward_rich, toward_poor, role, p_rich, fork_weight: float = 1.0):
    """Teacher-forced loss of a threshold agent's routes; see :mod:`goalmisgen.analysis.route_targets`.

    ``logits`` ``(N, 2, T, A)``; masks ``(N, 2, T, A)``; ``role`` ``(N, 2, T)``;
    ``p_rich`` ``(N,)``, the target agent's probability of taking the richer
    objective. Returns ``(loss, metrics)`` where the loss is the navigation mean
    plus ``fork_weight`` times the fork mean.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    lq_rich = _set_log_prob(log_probs, toward_rich)
    lq_poor = _set_log_prob(log_probs, toward_poor)
    p = p_rich[:, None, None]

    mixture = -(p * lq_rich + (1 - p) * lq_poor)
    branch_rich = -lq_rich
    branch_poor = -lq_poor
    branch_index = jnp.arange(2)[None, :, None]
    step_loss = jnp.where(
        role == BRANCH,
        jnp.where(branch_index == RICH, branch_rich, branch_poor),
        mixture,
    )
    branch_weight = jnp.where(branch_index == RICH, p, 1 - p)
    nav_weight = jnp.where(role == PREFIX, 1.0, 0.0) + jnp.where(role == BRANCH, branch_weight, 0.0)
    fork = (role == FORK).astype(logits.dtype)

    nav = (step_loss * nav_weight).sum() / jnp.maximum(nav_weight.sum(), 1e-9)
    fork_loss = (step_loss * fork).sum() / jnp.maximum(fork.sum(), 1.0)

    q_rich, q_poor = jnp.exp(lq_rich), jnp.exp(lq_poor)
    fork_share = q_rich / jnp.maximum(q_rich + q_poor, 1e-9)
    fork_error = (jnp.abs(fork_share - p) * fork).sum() / jnp.maximum(fork.sum(), 1.0)
    return nav + fork_weight * fork_loss, {"nav": nav, "fork": fork_loss, "fork_error": fork_error}


def threshold_targets(gap, tau, scale):
    """``p(rich) = sigmoid((tau - gap) / scale)``."""
    return jax.nn.sigmoid((tau - gap) / scale)


def fork_margins(logits, toward_rich, toward_poor, role):
    """``(N,)``: log q(A_rich) - log q(A_poor) at each level's fork, the greedy decision variable."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    margin = _set_log_prob(log_probs, toward_rich) - _set_log_prob(log_probs, toward_poor)
    fork = (role == FORK).reshape(role.shape[0], -1)
    return (margin.reshape(role.shape[0], -1) * fork).sum(-1) / jnp.maximum(fork.sum(-1), 1)


def margin_loss(logits, base_logits, toward_rich, toward_poor, role, offset, slope, fork_weight=1.0, nav_weight=10.0):
    """Move every fork margin to where the base's margin curve, translated along the gap, puts it.

    Behaviour is read greedily, so a level is decided by the sign of its fork
    margin. If the base's mean margin at gap ``g`` is ``f(g)``, an agent whose
    threshold is ``shift`` steps later has mean margin ``f(g - shift)``; the
    target for each level is its own base margin plus ``f(g - shift) - f(g)``,
    which the caller passes as ``offset``. That keeps the base's level-to-level
    noise and moves only the crossing.

    A single linear slope does not do this: the margin curve is steep near the
    threshold and flat in the tails, so a uniform margin shift sized by the
    global slope moved the DRC's rollout threshold by a fifth of what it asked
    for, while the fine-tuned axis -- which hits the behavioural target -- moved
    margins by several steps' worth near the boundary and by little far from it
    (``scripts/fork_margin_diagnostic.py``).

    Navigation is held to the base's own logits (a KL on every non-fork step).
    The fork term is ``((m - m*) / slope)^2``; ``slope`` only sets its units.
    """
    m = fork_margins(logits, toward_rich, toward_poor, role)
    m0 = jax.lax.stop_gradient(fork_margins(base_logits, toward_rich, toward_poor, role))
    fork_steps = (m - (m0 + offset)) / slope
    fork = jnp.mean(jnp.square(fork_steps))

    base_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(base_logits, axis=-1))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    kl = (jnp.exp(base_log_probs) * (base_log_probs - log_probs)).sum(-1)
    steps = ((role != 0) & (role != FORK)).astype(logits.dtype)
    nav = (kl * steps).sum() / jnp.maximum(steps.sum(), 1.0)
    return nav_weight * nav + fork_weight * fork, {"nav": nav, "fork": fork, "fork_error": jnp.mean(jnp.abs(fork_steps))}


def sign_loss(logits, base_logits, toward_rich, toward_poor, role, y, weight, margin=0.25, sharpness=4.0, fork_weight=1.0, nav_weight=10.0):
    """Make each fork decide ``y`` (+1 rich, -1 poor), leaving the size of the margin free.

    Greedy behaviour reads only the sign of the fork margin, and the fine-tuned
    axis -- which moves the rollout threshold exactly as asked -- changes
    individual margins by several steps' worth in both directions, so no
    per-level margin target scores it better than the unedited base. A smooth
    hinge on the sign, ``softplus(k * (margin - y * m)) / k``, does not charge
    for that. ``weight`` drops levels whose target is not defined.

    Navigation is held to the base's own logits (a KL on every non-fork step).
    ``fork_error`` is the weighted fraction of forks deciding the wrong way.
    """
    m = fork_margins(logits, toward_rich, toward_poor, role)
    hinge = jax.nn.softplus(sharpness * (margin - y * m)) / sharpness
    total = jnp.maximum(weight.sum(), 1e-9)
    fork = (weight * hinge).sum() / total
    wrong = (weight * (y * m < 0)).sum() / total

    base_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(base_logits, axis=-1))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    kl = (jnp.exp(base_log_probs) * (base_log_probs - log_probs)).sum(-1)
    steps = ((role != 0) & (role != FORK)).astype(logits.dtype)
    nav = (kl * steps).sum() / jnp.maximum(steps.sum(), 1.0)
    return nav_weight * nav + fork_weight * fork, {"nav": nav, "fork": fork, "fork_error": wrong}


def curve_loss(logits, base_logits, toward_rich, toward_poor, role, gap_bin, target_rate, n_bins, temperature=0.25, fork_weight=1.0, nav_weight=10.0, valid=None, hard=True):
    """Match the population's choice curve, not any level's choice.

    A threshold is a statistic of the psychometric curve -- P(rich) against the
    gap -- and many level-by-level patterns of decisions produce the same curve.
    The fine-tuned axis, which hits the behavioural threshold exactly, flips a
    different set of levels than any per-level rule asks for, and scores no
    better than the unedited base under a per-level sign target. So the fork
    term compares rates: within each gap bin, the mean soft decision
    ``sigmoid(m / temperature)`` against the mean target rate, squared and
    weighted by the bin's count.

    ``gap_bin`` ``(N,)`` int in ``[0, n_bins)``; ``target_rate`` ``(N,)`` is the
    rate the shifted agent should show at that level's gap. ``fork_error`` is
    the count-weighted mean absolute rate error. ``valid`` ``(N,)`` drops levels
    with no decision to read (an on-policy rollout that never committed).
    """
    m = fork_margins(logits, toward_rich, toward_poor, role)
    soft = jax.nn.sigmoid(m / temperature)
    if hard:
        # Straight-through: the value is the greedy decision, the gradient the soft one's. Without it a
        # high-capacity edit matched rates by parking borderline margins near zero, where sigmoid reads 0.5
        # and argmax reads whichever way the noise falls.
        soft = soft + jax.lax.stop_gradient((m > 0).astype(soft.dtype) - soft)
    w = jnp.ones(m.shape[0]) if valid is None else valid.astype(m.dtype)
    counts = jnp.zeros(n_bins).at[gap_bin].add(w)
    rate = jnp.zeros(n_bins).at[gap_bin].add(w * soft) / jnp.maximum(counts, 1.0)
    target = jnp.zeros(n_bins).at[gap_bin].add(w * target_rate) / jnp.maximum(counts, 1.0)
    fork = (counts * jnp.square(rate - target)).sum() / counts.sum()
    error = (counts * jnp.abs(rate - target)).sum() / counts.sum()

    base_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(base_logits, axis=-1))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    kl = (jnp.exp(base_log_probs) * (base_log_probs - log_probs)).sum(-1)
    steps = ((role != 0) & (role != FORK)).astype(logits.dtype)
    nav = (kl * steps).sum() / jnp.maximum(steps.sum(), 1.0)
    return nav_weight * nav + fork_weight * fork, {"nav": nav, "fork": fork, "fork_error": error}


def onpolicy_loss(onpolicy_fn, logits_fn, base, flat, batch, index, config: "SearchConfig"):
    """The curve loss at the commitments the edited agent actually makes, navigation held to the base along its own path."""
    trajectory = onpolicy_fn(flat, batch)
    base_logits = jax.lax.stop_gradient(
        logits_fn(base, {"static": batch["static"], "positions": jax.lax.stop_gradient(trajectory["positions"])[:, None]})
    )
    loss, metrics = curve_loss(
        trajectory["logits"][:, None], base_logits, trajectory["toward_rich"][:, None], trajectory["toward_poor"][:, None],
        trajectory["role"][:, None], batch["gap_bin"], batch["curve_rate"][:, index], config.n_bins, config.temperature,
        config.fork_weight, config.nav_weight, valid=trajectory["committed"], hard=config.hard,
    )
    # Levels that never commit drop out of the rate, so an edit could buy rate error by making the agent dither.
    # Charge the shortfall against the base's own commitment rate, measured on the same batch.
    uncommitted = 1.0 - trajectory["committed"].astype(loss.dtype).mean()
    shortfall = jnp.maximum(uncommitted - (1.0 - batch["base_committed"].astype(loss.dtype).mean()), 0.0)
    return loss + config.commit_weight * shortfall, {**metrics, "uncommitted": uncommitted}


def choice_curve(gaps: np.ndarray, decided_rich: np.ndarray, width: float = 1.0):
    """The base's fork-decision rate against gap: a monotone (non-increasing) fit on integer-gap bins, interpolated."""
    gaps = np.asarray(gaps, dtype=float)
    keys = np.round(gaps / width).astype(int)
    grid, rates, counts = [], [], []
    for k in np.unique(keys):
        rows = keys == k
        grid.append(k * width)
        rates.append(float(np.mean(decided_rich[rows])))
        counts.append(float(rows.sum()))
    blocks = [[r, c, 1] for r, c in zip(rates, counts)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            w = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [(blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / w, w, blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.concatenate([[r] * n for r, _, n in blocks])
    grid = np.array(grid)
    return lambda x: np.interp(np.asarray(x, dtype=float), grid, fitted, left=fitted[0], right=fitted[-1])


def curve_targets(curve, gaps: np.ndarray, rich_above: np.ndarray, shifts, rule: str, bin_width: float = 2.0):
    """``gap_bin (N,)``, ``n_bins``, and ``target_rate (N, J)`` = ``curve(gap - s)`` with ``s`` signed per level under the contingent rule.

    Under the contingent rule the two halves of the population need different
    curves at the same gap, so their bins are kept apart.
    """
    gaps = np.asarray(gaps, dtype=float)
    low = np.floor(gaps.min() / bin_width)
    base_bin = (np.floor(gaps / bin_width) - low).astype(np.int32)
    n_gap_bins = int(base_bin.max()) + 1
    gap_bin = base_bin if rule == "threshold" else base_bin + n_gap_bins * np.asarray(rich_above, dtype=np.int32)
    n_bins = n_gap_bins * (1 if rule == "threshold" else 2)
    rate = np.zeros((len(gaps), len(shifts)), dtype=np.float32)
    for j, shift in enumerate(shifts):
        signed = np.full(len(gaps), shift) if rule == "threshold" else np.where(rich_above, shift, -shift)
        rate[:, j] = curve(gaps - signed)
    return gap_bin, n_bins, rate


def rule_labels(gaps: np.ndarray, base_margins: np.ndarray, rich_above: np.ndarray, shifts, rule: str):
    """Fork decisions a threshold agent shifted by each dose would make, and which levels to trust.

    The base's own margin-sign threshold ``tau_m`` is the half-integer gap that
    best separates its fork decisions. Levels where the base already disagrees
    with ``sign(tau_m - gap)`` get weight zero -- that is the base's noise, and
    asking an edit to keep or to fix it would each be a different target. The
    labels for dose ``s`` are ``sign(tau_m + s - gap)``, with ``s`` signed per
    level under the contingent rule.

    Returns ``(tau_m, y (N, J), weight (N,))``.
    """
    gaps = np.asarray(gaps, dtype=float)
    decided = np.asarray(base_margins) > 0
    candidates = np.arange(np.floor(gaps.min()) - 0.5, np.ceil(gaps.max()) + 1.0, 1.0)
    agreement = [np.mean(decided == (gaps < c)) for c in candidates]
    tau_m = float(candidates[int(np.argmax(agreement))])
    weight = (decided == (gaps < tau_m)).astype(np.float32)
    y = np.zeros((len(gaps), len(shifts)), dtype=np.float32)
    for j, shift in enumerate(shifts):
        signed = np.full(len(gaps), shift) if rule == "threshold" else np.where(rich_above, shift, -shift)
        y[:, j] = np.where(gaps < tau_m + signed, 1.0, -1.0)
    return tau_m, y, weight


def margin_curve(gaps: np.ndarray, margins: np.ndarray, min_count: int = 20):
    """The base's mean fork margin as a function of gap: binned by integer gap, monotone, linearly interpolated.

    Returns ``f`` mapping gap arrays to margins. Bins with fewer than
    ``min_count`` levels are dropped; beyond the data the curve is held flat,
    since margins saturate in the tails. Isotonic (non-increasing) so the
    translation has a single crossing to move.
    """
    gaps = np.asarray(gaps, dtype=float)
    margins = np.asarray(margins, dtype=float)
    keys = np.round(gaps).astype(int)
    grid, means = [], []
    for g in np.unique(keys):
        rows = keys == g
        if rows.sum() >= min_count:
            grid.append(float(g))
            means.append(float(margins[rows].mean()))
    grid, means = np.array(grid), np.array(means)
    # Pool adjacent violators for a non-increasing fit.
    values, weights = list(means), [1.0] * len(means)
    blocks = [[v, w, 1] for v, w in zip(values, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.concatenate([[v] * n for v, _, n in blocks])
    return lambda x: np.interp(np.asarray(x, dtype=float), grid, fitted)


def margin_offsets(curve, gaps: np.ndarray, rich_above: np.ndarray, shifts, rule: str) -> np.ndarray:
    """``(N, J)``: ``f(gap - s) - f(gap)`` per level and shift, with ``s`` signed per level under the contingent rule."""
    gaps = np.asarray(gaps, dtype=float)
    out = np.zeros((len(gaps), len(shifts)), dtype=np.float32)
    for j, shift in enumerate(shifts):
        signed = np.full(len(gaps), shift) if rule == "threshold" else np.where(rich_above, shift, -shift)
        out[:, j] = curve(gaps - signed) - curve(gaps)
    return out


# --- edit families ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Leaf:
    """One parameter tensor's place in the flattened vector."""

    path: str
    start: int
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


def leaves_of(params) -> list[Leaf]:
    from jax import tree_util

    from goalmisgen.analysis.layers import path_string

    out, start = [], 0
    for path, leaf in tree_util.tree_flatten_with_path(params)[0]:
        shape = tuple(int(n) for n in np.shape(leaf))
        out.append(Leaf(path_string(path), start, shape))
        start += int(np.prod(shape))
    return out


class EditFamily:
    """Maps edit parameters ``phi`` to a full-length direction ``delta``."""

    name: str

    def init(self, key) -> dict:
        raise NotImplementedError

    def delta(self, phi) -> jnp.ndarray:
        raise NotImplementedError

    def penalty(self, phi) -> jnp.ndarray:
        return sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(phi))

    def n_parameters(self) -> int:
        return int(sum(np.prod(np.shape(x)) for x in jax.tree_util.tree_leaves(self.init(jax.random.PRNGKey(0)))))

    def rescale(self, phi, factor):
        """``phi`` whose ``delta`` is ``factor`` times as long. Linear families scale every leaf."""
        return jax.tree_util.tree_map(lambda x: x * factor, phi)


@dataclasses.dataclass
class DenseEdit(EditFamily):
    """Any direction on a fixed set of coordinates; all of them if ``indices`` is ``None``."""

    size: int
    indices: np.ndarray | None = None
    init_scale: float = 0.0
    name: str = "dense"

    def init(self, key) -> dict:
        n = self.size if self.indices is None else len(self.indices)
        return {"u": self.init_scale * jax.random.normal(key, (n,)) / np.sqrt(n)}

    def delta(self, phi) -> jnp.ndarray:
        if self.indices is None:
            return phi["u"]
        return jnp.zeros(self.size, dtype=phi["u"].dtype).at[jnp.asarray(self.indices)].set(phi["u"])


@dataclasses.dataclass
class LowRankEdit(EditFamily):
    """A rank-``r`` factor per kernel, over ``out x (everything else)``; biases and norms untouched.

    ``B`` starts at zero and ``A`` random, so ``delta`` starts at zero and both
    factors receive gradient. The penalty ``||A||^2 + ||B||^2`` is, at its
    optimum over factorisations, twice the nuclear norm of ``BA``.
    """

    size: int
    kernels: Sequence[Leaf]
    rank: int
    name: str = "lowrank"

    def init(self, key) -> dict:
        phi = {}
        for i, leaf in enumerate(self.kernels):
            out = leaf.shape[-1]
            rest = leaf.size // out
            r = min(self.rank, out, rest)
            key, sub = jax.random.split(key)
            phi[f"{i}"] = {"A": jax.random.normal(sub, (r, rest)) / np.sqrt(rest), "B": jnp.zeros((out, r))}
        return phi

    def delta(self, phi) -> jnp.ndarray:
        flat = jnp.zeros(self.size)
        for i, leaf in enumerate(self.kernels):
            factor = phi[f"{i}"]
            matrix = factor["B"] @ factor["A"]  # (out, rest)
            # Kernels are (..., out) with out last; (out, rest) reshapes to that by moving axis 0 to the end.
            tensor = jnp.moveaxis(matrix.reshape((leaf.shape[-1], *leaf.shape[:-1])), 0, -1)
            flat = flat.at[leaf.start : leaf.start + leaf.size].set(tensor.ravel())
        return flat

    def rescale(self, phi, factor):
        """``delta`` is bilinear in the factors, so only ``B`` is scaled."""
        return {i: {"A": f["A"], "B": f["B"] * factor} for i, f in phi.items()}


@dataclasses.dataclass
class GainEdit(EditFamily):
    """One scalar per input plane: ``delta = sum_c g_c * K_c``, i.e. a brightness change of ``1 + shift * g_c``."""

    size: int
    planes: dict[int, np.ndarray]  # channel -> flat indices of the input kernel reading it
    base_flat: np.ndarray
    name: str = "gain"

    def init(self, key) -> dict:
        return {"g": jnp.zeros(len(self.planes))}

    def delta(self, phi) -> jnp.ndarray:
        flat = jnp.zeros(self.size)
        for i, rows in enumerate(self.planes.values()):
            flat = flat.at[jnp.asarray(rows)].set(phi["g"][i] * jnp.asarray(self.base_flat[rows], dtype=jnp.float32))
        return flat

    def penalty(self, phi) -> jnp.ndarray:
        return jnp.sum(jnp.square(self.delta(phi)))


def module_indices(params, modules: Sequence[str]) -> np.ndarray:
    """Flat indices of the named modules (``goalmisgen.analysis.layers`` grouping).

    A name selects the module of that name, or every module it prefixes -- ``cell``
    is all six ConvLSTM groups. Underscores stand for spaces, so names survive a
    shell job file unquoted (``input_conv``).
    """
    from goalmisgen.analysis.layers import module_groups

    groups, _ = module_groups(params)
    chosen = set()
    for raw in modules:
        name = raw.replace("_", " ")
        hits = [g for g in groups if g == name or g.startswith(name + " ")]
        if not hits:
            raise ValueError(f"no module matches {raw!r}; have {sorted(groups)}")
        chosen.update(hits)
    return np.sort(np.concatenate([groups[m] for m in sorted(chosen)]))


def weight_kernels(params, modules: Sequence[str] | None = None) -> list[Leaf]:
    """Every tensor with two or more axes, optionally only inside ``modules``."""
    from goalmisgen.analysis.layers import module_of

    wanted = None if modules is None else [m.replace("_", " ") for m in modules]
    return [
        leaf
        for leaf in leaves_of(params)
        if len(leaf.shape) >= 2
        and (wanted is None or any(module_of(leaf.path) == w or module_of(leaf.path).startswith(w + " ") for w in wanted))
    ]


# --- the search --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    shifts: tuple[float, ...] = (-2.0, 2.0, 4.0)
    """Threshold shifts, in steps, the one edit is trained to hit at dose = shift."""

    tau_base: float = 7.0
    scale: float = 0.5
    rule: str = "threshold"
    """``threshold`` or ``contingent`` (the control)."""

    steps: int = 400
    batch: int = 64
    """Levels per step; each contributes two branches at every shift."""

    lr: float = 1e-3
    optimizer: str = "adam"
    l2: float = 0.0
    max_norm: float | None = None
    """Budget: after every update ``delta`` is projected back into the ball of this radius.

    The clean way to ask "what is the best edit no longer than v". A penalty
    would do it in principle, but Adam's per-coordinate normalisation makes a
    small penalty inert and a large one a second hyperparameter to sweep; plain
    SGD, whose geometry does match the L2 metric, diverges on this loss surface
    at any usable learning rate (sensitive coordinates in the input layer and
    heads sit beside a million flat ones in the dense layer).
    """

    fork_weight: float = 1.0
    target: str = "margin"
    """``margin`` (shift the base's fork margins, hold navigation to the base) or ``soft`` (solver routes, soft labels)."""

    margin_slope: float = 1.0
    """``|b|``, the base's fork margin per step of gap; the margin target needs it."""

    nav_weight: float = 10.0
    n_bins: int = 1
    temperature: float = 0.25
    hard: bool = True
    commit_weight: float = 1.0
    on_policy: bool = False
    """Read decisions where the edited agent commits in closed loop, not at the solver's fork (needs ``onpolicy_fn``)."""

    seed: int = 0
    log_every: int = 25


def signed_shifts(batch, shift, rule: str):
    if rule == "threshold":
        return jnp.full(batch["gap"].shape, shift)
    if rule == "contingent":
        return jnp.where(batch["rich_above"], shift, -shift)
    raise ValueError(f"unknown rule {rule!r}")


def shift_loss(logits, base_logits, batch, shift, index, config: "SearchConfig"):
    """The loss for dose ``shift`` (the ``index``-th of ``config.shifts``), under whichever target the config names."""
    signed = signed_shifts(batch, shift, config.rule)
    if config.target == "margin":
        return margin_loss(
            logits, base_logits, batch["toward_rich"], batch["toward_poor"], batch["role"], batch["margin_offset"][:, index],
            config.margin_slope, config.fork_weight, config.nav_weight,
        )
    if config.target == "curve":
        return curve_loss(
            logits, base_logits, batch["toward_rich"], batch["toward_poor"], batch["role"], batch["gap_bin"],
            batch["curve_rate"][:, index], config.n_bins, config.temperature, config.fork_weight, config.nav_weight,
        )
    if config.target == "sign":
        return sign_loss(
            logits, base_logits, batch["toward_rich"], batch["toward_poor"], batch["role"], batch["rule_sign"][:, index],
            batch["rule_weight"], fork_weight=config.fork_weight, nav_weight=config.nav_weight,
        )
    if config.target == "soft":
        p = threshold_targets(batch["gap"] - signed + shift, config.tau_base + shift, config.scale)
        return route_loss(logits, batch["toward_rich"], batch["toward_poor"], batch["role"], p, config.fork_weight)
    raise ValueError(f"unknown target {config.target!r}")


@dataclasses.dataclass
class SearchResult:
    phi: dict
    delta: np.ndarray
    history: list[dict]


def search(
    family: EditFamily,
    logits_fn: Callable,
    base_flat,
    targets,
    config: SearchConfig,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
    onpolicy_fn: Callable | None = None,
    init_phi=None,
) -> SearchResult:
    """Optimise ``phi`` so that ``base + shift * delta(phi)`` follows the rule at every shift.

    ``logits_fn(flat_params, batch_arrays)`` returns ``(N, 2, T, A)`` logits and
    ``targets`` is a mapping of numpy arrays with a leading level axis, including
    ``toward_rich``, ``toward_poor``, ``role``, ``gap`` and ``rich_above`` shaped
    to match those logits.
    """
    base = jnp.asarray(base_flat, dtype=jnp.float32)
    shifts = jnp.asarray(config.shifts, dtype=jnp.float32)
    key = jax.random.PRNGKey(config.seed)
    phi = family.init(key) if init_phi is None else init_phi
    schedule = optax.cosine_decay_schedule(config.lr, config.steps, alpha=0.05)
    if config.optimizer == "adam":
        optimizer = optax.adam(schedule)
    elif config.optimizer == "sgd":
        optimizer = optax.sgd(schedule, momentum=0.9)
    else:
        raise ValueError(f"unknown optimizer {config.optimizer!r}")
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optimizer)
    state = optimizer.init(phi)

    def objective(phi, batch):
        delta = family.delta(phi)
        needs_base = config.target in ("margin", "sign", "curve") and not config.on_policy
        base_logits = jax.lax.stop_gradient(logits_fn(base, batch)) if needs_base else None
        total, parts = 0.0, {"nav": 0.0, "fork": 0.0, "fork_error": 0.0}
        if config.on_policy:
            parts["uncommitted"] = 0.0
        for index, shift in enumerate(shifts):
            if config.on_policy:
                loss, metrics = onpolicy_loss(onpolicy_fn, logits_fn, base, base + shift * delta, batch, index, config)
            else:
                loss, metrics = shift_loss(logits_fn(base + shift * delta, batch), base_logits, batch, shift, index, config)
            total = total + loss
            parts = {k: parts[k] + metrics.get(k, 0.0) for k in parts}
        n = len(config.shifts)
        penalty = family.penalty(phi)
        parts = {k: v / n for k, v in parts.items()}
        return total / n + config.l2 * penalty, {**parts, "penalty": penalty, "norm": jnp.linalg.norm(delta)}

    @jax.jit
    def update(phi, state, batch):
        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(phi, batch)
        updates, state = optimizer.update(grads, state, phi)
        phi = optax.apply_updates(phi, updates)
        if config.max_norm is not None:
            norm = jnp.linalg.norm(family.delta(phi))
            phi = family.rescale(phi, jnp.minimum(1.0, config.max_norm / jnp.maximum(norm, 1e-12)))
        return phi, state, loss, metrics

    rng = np.random.default_rng(config.seed)
    history = []
    n = len(targets["gap"])
    for step in range(config.steps):
        rows = rng.choice(n, size=min(config.batch, n), replace=False)
        batch = {k: jnp.asarray(v[rows]) for k, v in targets.items()}
        phi, state, loss, metrics = update(phi, state, batch)
        if step % config.log_every == 0 or step == config.steps - 1:
            row = {"step": step, "loss": float(loss), **{k: float(v) for k, v in metrics.items()}}
            history.append(row)
            log(
                f"  step {step:>5}  loss {row['loss']:.4f}  nav {row['nav']:.4f}  fork {row['fork']:.4f}"
                f"  fork error {row['fork_error']:.3f}  |delta| {row['norm']:.4g}"
                + (f"  uncommitted {row['uncommitted']:.3f}" if "uncommitted" in row else "")
            )
    return SearchResult(phi=phi, delta=np.asarray(family.delta(phi), dtype=np.float64), history=history)


_LOSS_CACHE: dict = {}


def _jitted_loss(logits_fn, config: "SearchConfig", onpolicy_fn=None):
    key = (id(logits_fn), id(onpolicy_fn), config)
    if key not in _LOSS_CACHE:

        @jax.jit
        def loss(base, flat, batch, shift, index):
            if config.on_policy:
                return onpolicy_loss(onpolicy_fn, logits_fn, base, flat, batch, index, config)
            base_logits = logits_fn(base, batch) if config.target in ("margin", "sign", "curve") else None
            return shift_loss(logits_fn(flat, batch), base_logits, batch, shift, index, config)

        _LOSS_CACHE[key] = loss
    return _LOSS_CACHE[key]


def behaviour_loss(delta, logits_fn, base_flat, targets, config: SearchConfig, rows: np.ndarray, onpolicy_fn=None) -> dict[str, float]:
    """The search's objective without its penalty, for any direction, on fixed levels.

    The yardstick that makes "cheapest" checkable: the fine-tuned axis, rescaled
    to steps, is itself a candidate edit, so a search that ends longer than it
    *and* with a higher loss has not found the cheapest edit -- it has found an
    edit.
    """
    base = jnp.asarray(base_flat, dtype=jnp.float32)
    direction = jnp.asarray(delta, dtype=jnp.float32)
    totals = {"loss": 0.0, "nav": 0.0, "fork": 0.0, "fork_error": 0.0}
    for start in range(0, len(rows), config.batch):
        batch = {k: jnp.asarray(v[rows[start : start + config.batch]]) for k, v in targets.items()}
        for index, shift in enumerate(config.shifts):
            loss, metrics = _jitted_loss(logits_fn, config, onpolicy_fn)(base, base + shift * direction, batch, jnp.float32(shift), index)
            weight = len(batch["gap"]) / (len(rows) * len(config.shifts))
            totals["loss"] += float(loss) * weight
            for k in ("nav", "fork", "fork_error"):
                totals[k] += float(metrics[k]) * weight
    return totals


def top_k_indices(delta: np.ndarray, k: int, within: np.ndarray | None = None) -> np.ndarray:
    """Coordinates of the ``k`` largest-magnitude entries, optionally only among ``within``."""
    candidates = np.arange(len(delta)) if within is None else np.asarray(within)
    order = np.argsort(-np.abs(delta[candidates]))[:k]
    return np.sort(candidates[order])
