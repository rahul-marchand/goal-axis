"""Does writing the value axis change the plan, or only something downstream of it?

Two readings of the value axis make different predictions about the DRC's per-cell
state. If the axis is a **gain** on how loudly a colour is seen, value is carried
as the *strength* of that colour's features, so the distance field the network
builds out from the colour-``c`` objective changes with the write, from the first
tick, along the cells that field covers. If the axis is a **separate value
circuit**, the fields stay put and the write lands in a small subspace the probes
do not read, most likely late and near the agent or the fork.

Everything here is measurement on arrays; the script
``scripts/plan_invariance.py`` supplies them. Three pieces:

* **ticks** -- the DRC applies its cells ``repeats_per_step`` times per step, and
  only the last tick's state reaches the carry. A copy of the policy with one
  repeat, applied ``n`` times on the same observation, walks the same recurrence
  one tick at a time (``tests/test_plan_invariance.py`` checks the identity
  against the stock policy rather than trusting it).
* **edits to the observation** -- a colour plane scaled (the gain reference) or an
  objective removed outright (the null-object test: a pure input gain on plane
  ``c`` does nothing when plane ``c`` is empty).
* **summaries** -- how a probe's decoded field moves between two networks, where
  in the maze the activation change sits, and how much of it lies along the
  probes' own readout directions.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis import geometry
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL

# --- ticks ---------------------------------------------------------------------


def one_tick_policy(policy):
    """The same policy with one cell application per call.

    Applying it ``repeats_per_step`` times to one observation, with no episode
    start in between, is the stock step: the carry is the only thing a repeat
    hands the next, and the input encoding is recomputed identically each call.
    """
    cfg = policy.cfg
    if not hasattr(cfg, "repeats_per_step"):
        raise ValueError(f"{type(cfg).__name__} has no repeats; only a DRC has ticks to walk")
    return policy.clone(cfg=dataclasses.replace(cfg, repeats_per_step=1))


def make_tick_runner(policy, n_ticks: int, state: str = "h"):
    """A jitted ``(params, observations) -> (grids, logits)`` over ``n_ticks`` ticks from a zero carry.

    ``grids`` is ``(n_ticks, n_layers, batch, height, width, channels)`` -- the
    ConvLSTM's hidden state ``h`` (or its cell state ``c``) of every layer after every tick. ``logits`` are the action logits after
    each tick, ``(n_ticks, batch, n_actions)``; only every ``repeats``-th is one the
    agent would act on.
    """
    if state not in ("h", "c"):
        raise ValueError(f"state must be 'h' or 'c', got {state!r}")
    tick = one_tick_policy(policy)

    def run(params, observations):
        carry = tick.apply(params, jax.random.PRNGKey(0), observations.shape, method=tick.initialize_carry)
        starts = jnp.zeros(observations.shape[0], dtype=bool)
        hs, logits = [], []
        for _ in range(n_ticks):
            carry, _, logit, _ = tick.apply(
                params, carry, observations, starts, jax.random.PRNGKey(0), method=tick.get_action, temperature=0.0
            )
            hs.append(jnp.stack([getattr(layer, state) for layer in carry]))
            logits.append(logit)
        return jnp.stack(hs), jnp.stack(logits)

    return jax.jit(run)


# --- observation edits ------------------------------------------------------------


def remove_objectives(observations: np.ndarray, feature_ids, channel_axis: int = 1) -> np.ndarray:
    """A copy with the named objectives' colour planes emptied: the cell becomes floor."""
    out = np.array(observations, dtype=np.float32, copy=True)
    moved = np.moveaxis(out, channel_axis, -1)
    for feature_id in feature_ids:
        moved[..., FIRST_FEATURE_CHANNEL + feature_id] = 0.0
    return out


# --- maze geometry per level -------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LevelGeometry:
    """Per-cell masks and fields for one level, from its initial observation (HWC)."""

    free: np.ndarray
    distance: tuple[np.ndarray, ...]
    """Per objective, BFS distance to it with the other objectives blocking; NaN off the free set."""

    from_agent: np.ndarray
    masks: dict[str, np.ndarray]


LOCI = ("agent", "objective 0", "objective 1", "route 0 only", "route 1 only", "shared route", "off route")


def level_geometry(observation: np.ndarray, n_features: int = 2) -> LevelGeometry:
    """Where the agent, the objectives and the shortest routes to each are.

    A cell is on route ``f`` when it lies on *some* shortest path from the agent to
    objective ``f``: ``d_agent(x) + d_f(x) == d_f(agent)``. Cells on both routes are
    the shared corridor, whose far end is the fork.
    """
    free = geometry.free_cells(observation)
    agent = geometry.agent_cell(observation)
    distance, routes = [], []
    masks: dict[str, np.ndarray] = {}
    agent_field = geometry.bfs_field(~free, agent)
    for f in range(n_features):
        walls = geometry.blocking_walls(observation, f, n_features)
        to_f = geometry.bfs_field(walls, geometry.objective_cell(observation, f))
        from_agent = geometry.bfs_field(walls, agent)
        total = to_f[agent]
        on = np.isfinite(to_f) & np.isfinite(from_agent) & (np.abs(from_agent + to_f - total) < 1e-9)
        distance.append(np.where(free, to_f, np.nan))
        routes.append(on)
        cell = np.zeros_like(free)
        cell[geometry.objective_cell(observation, f)] = True
        masks[f"objective {f}"] = cell
    agent_mask = np.zeros_like(free)
    agent_mask[agent] = True
    special = agent_mask | masks["objective 0"] | masks["objective 1"]
    masks["agent"] = agent_mask
    masks["route 0 only"] = routes[0] & ~routes[1] & ~special
    masks["route 1 only"] = routes[1] & ~routes[0] & ~special
    masks["shared route"] = routes[0] & routes[1] & ~special
    masks["off route"] = free & ~routes[0] & ~routes[1] & ~special
    return LevelGeometry(free=free, distance=tuple(distance), from_agent=np.where(free, agent_field, np.nan), masks=masks)


# --- summaries ---------------------------------------------------------------------


def field_shift(base: np.ndarray, other: np.ndarray, label: np.ndarray, bins=(0, 4, 8, 12, np.inf)) -> dict:
    """How a probe's decoded field moved between two networks, in units of the field's own spread.

    ``base`` and ``other`` are the same probe's calibrated predictions on the same
    cells, ``label`` the true distance there. ``offset`` is the mean shift,
    ``slope`` the regression of ``other`` on ``base`` (a field stretched or
    compressed has slope away from one), ``rms`` the whole change, and ``by_distance``
    the mean shift in bands of true distance -- whether the change is near the
    objective or reaches along the field.
    """
    base, other, label = (np.asarray(a, dtype=np.float64).ravel() for a in (base, other, label))
    sd = float(label.std())
    delta = other - base
    centred = base - base.mean()
    slope = float(centred @ (other - other.mean()) / (centred @ centred)) if centred @ centred > 0 else float("nan")
    by_distance = []
    for low, high in zip(bins, bins[1:]):
        rows = (label >= low) & (label < high)
        by_distance.append(float(delta[rows].mean() / sd) if rows.any() else float("nan"))
    return {
        "offset": float(delta.mean() / sd),
        "rms": float(np.sqrt(np.mean(delta**2)) / sd),
        "slope": slope,
        "by_distance": by_distance,
        "bins": [float(b) for b in bins],
    }


def r2(label: np.ndarray, prediction: np.ndarray) -> float:
    label, prediction = np.asarray(label, np.float64).ravel(), np.asarray(prediction, np.float64).ravel()
    total = np.sum((label - label.mean()) ** 2)
    return float(1 - np.sum((label - prediction) ** 2) / total) if total > 0 else float("nan")


def locus_enrichment(change: np.ndarray, geometries) -> dict[str, float]:
    """Mean squared activation change on each locus, over its mean on free cells.

    ``change`` is ``(levels, height, width)``, the squared norm of the activation
    difference per cell. One means the change is spread evenly; the agent cell or
    one route standing well above the rest says where it was written.
    """
    sums = {name: 0.0 for name in LOCI}
    counts = {name: 0 for name in LOCI}
    free_sum = free_count = 0.0
    for level, geo in zip(change, geometries):
        free_sum += float(level[geo.free].sum())
        free_count += int(geo.free.sum())
        for name in LOCI:
            sums[name] += float(level[geo.masks[name]].sum())
            counts[name] += int(geo.masks[name].sum())
    mean = free_sum / max(free_count, 1)
    return {name: (sums[name] / counts[name] / mean if counts[name] and mean > 0 else float("nan")) for name in LOCI}


def map_correlations(change: np.ndarray, geometries) -> dict[str, float]:
    """Per level, the correlation across free cells of the change with nearness to each source; averaged.

    Nearness is minus BFS distance, so a positive number means the change is
    largest close to that source: objective 0, objective 1 or the agent.
    """
    out: dict[str, list[float]] = {"objective 0": [], "objective 1": [], "agent": []}
    for level, geo in zip(change, geometries):
        sources = {"objective 0": geo.distance[0], "objective 1": geo.distance[1], "agent": geo.from_agent}
        for name, field in sources.items():
            rows = geo.free & np.isfinite(field)
            x, y = level[rows], -field[rows]
            if x.std() > 0 and y.std() > 0:
                out[name].append(float(np.corrcoef(x, y)[0, 1]))
    return {name: float(np.mean(v)) if v else float("nan") for name, v in out.items()}


def subspace_share(delta: np.ndarray, directions: np.ndarray) -> float:
    """Share of ``||delta||^2`` inside the span of ``directions``.

    ``delta`` is ``(n, channels)``, ``directions`` ``(k, channels)``. Chance for an
    isotropic change is ``k / channels``.
    """
    q, _ = np.linalg.qr(np.asarray(directions, np.float64).T)  # (channels, k)
    delta = np.asarray(delta, np.float64)
    total = float(np.sum(delta**2))
    return float(np.sum((delta @ q) ** 2) / total) if total > 0 else float("nan")


def readout_directions(weights: np.ndarray, std: np.ndarray, floor: float = 0.1) -> np.ndarray:
    """A ridge probe's readout direction in activation space, ``w / std``, with the spread floored.

    The probe standardises its inputs, so a channel that barely varies (the DRC has
    several with ``std ~ 1e-5`` against a typical 0.3) gets an ordinary weight that,
    divided by its tiny spread, dominates the direction and points it along a unit
    no write moves -- the share along it then reads as zero whatever the write did.
    Spreads below ``floor`` times the median are raised to it.
    """
    effective = np.maximum(std, floor * np.median(std))
    return weights / effective


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def stable_levels(choices: dict[str, np.ndarray]) -> np.ndarray:
    """Levels whose reached objective is the same under every condition, and not a timeout (-1)."""
    stacked = np.stack(list(choices.values()))
    return np.all(stacked == stacked[0], axis=0) & (stacked[0] >= 0)


def module_matched_random(axis: np.ndarray, groups: dict[str, np.ndarray], seed: int = 0) -> np.ndarray:
    """A random direction with the axis's norm inside every module.

    Stricter than one matched overall norm: the input convolution is small and
    sensitive, and a globally matched random vector would put almost nothing there.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(axis, dtype=np.float64)
    for index in groups.values():
        piece = rng.normal(size=len(index))
        norm = np.linalg.norm(axis[index])
        out[index] = piece / np.linalg.norm(piece) * norm if norm > 0 else 0.0
    return out


def interpolate_gain(curve_gains, curve_taus, target_tau: float) -> float:
    """The gain whose threshold is ``target_tau``, linear in log gain between measured points.

    Clipped to the measured range rather than extrapolated; the caller reports the
    threshold actually measured at the returned gain.
    """
    order = np.argsort(curve_taus)
    taus = np.asarray(curve_taus, np.float64)[order]
    logs = np.log(np.asarray(curve_gains, np.float64))[order]
    finite = np.isfinite(taus)
    return float(np.exp(np.interp(target_tau, taus[finite], logs[finite])))
