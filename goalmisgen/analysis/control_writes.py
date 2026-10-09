"""Directions that move the threshold but are not a gain, for testing the instruments.

The study's tests all ask the same question of the fitted axis: does it look like
turning a colour's brightness up or down? They agree that it does. That is only
evidence if the tests would have said *no* for something that is not a gain, and
nothing so far has checked that -- the control network of E5 never learned to use
value, so it could not supply the negative.

The knobs here are the negative. Each moves the threshold and none of them can be
an input gain:

* **late** and **readout** -- cheapest-edit solutions confined to the last cell
  and the readout (``scripts/cheapest_edit.py --modules``). They act after the
  encoder, and the late edit does not carry the axis's route-optimality damage.
* **rotation** -- :class:`~goalmisgen.edits.rotation.RotationEdit`, which turns a
  colour's input kernel instead of scaling it, keeping its length.
* **contingent** -- ``cheapest_edit.py --rule contingent``, trained to move the
  threshold *up* where the richer objective is above the agent and *down* where
  it is below. A value knob cannot do that, so it is the cleanest distractor.

**Matching.** A knob written at an arbitrary dose is not comparable with the
axis: gain alignment and the object-independent share both move with the size of
the change. Every knob is therefore scaled so that its own antisymmetric
threshold shift equals the axis's, measured on the same held-out levels, and the
tests are run on the two writes that shift is measured from. :func:`match_scale`
does one measurement and one linear correction, which is enough because each
knob is close to linear in its dose over this range; what it actually achieved is
reported rather than assumed.

The rotation is the exception that shapes the interface: ``delta`` is not linear
in its angle, so a knob is stored as an explicit pair of parameter vectors rather
than as a direction and a dose, and :func:`save_pairs` writes that pair.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Matched:
    """One knob, scaled so its threshold shift matches a target."""

    name: str
    plus: np.ndarray
    minus: np.ndarray
    scale: float
    shift: float
    """The threshold shift the pair actually produced, ``tau(plus) - tau(minus)``."""
    target: float
    history: list[tuple[float, float]]
    """``(scale, shift)`` for every round, in order."""

    @property
    def error(self) -> float:
        return abs(self.shift - self.target)


def linear_pair(base: np.ndarray, direction: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``base +- scale * direction``, for a knob stored as a direction."""
    step = scale * direction
    return base + step, base - step


def rotation_pair(family, base: np.ndarray, turns: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``base + delta(+scale * turns)`` and ``base + delta(-scale * turns)``.

    A rotation is not linear in its angle, so the two signs are built by turning
    each way rather than by negating one write's change.
    """
    turns = np.asarray(turns, dtype=np.float64)
    plus = np.asarray(family.delta({"t": scale * turns}), dtype=np.float64)
    minus = np.asarray(family.delta({"t": -scale * turns}), dtype=np.float64)
    return base + plus, base + minus


def match_scale(
    make_pair,
    shift_fn,
    target: float,
    scale0: float,
    name: str = "",
    rounds: int = 2,
    clip: tuple[float, float] = (0.15, 8.0),
) -> Matched:
    """Scale a knob until its threshold shift is ``target``; keep the closest round.

    ``make_pair(scale)`` returns the two parameter vectors and ``shift_fn(plus,
    minus)`` their threshold shift. The first round uses ``scale0``; each later
    round multiplies by ``target / shift``, the exact correction for a knob whose
    shift is proportional to its dose, clipped so that a round which barely moved
    the threshold cannot send the next one somewhere absurd.
    """
    if target == 0:
        raise ValueError("a target shift of zero does not identify a scale")
    history: list[tuple[float, float]] = []
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    scale = float(scale0)
    for _ in range(max(1, rounds)):
        plus, minus = make_pair(scale)
        shift = float(shift_fn(plus, minus))
        history.append((scale, shift))
        if best is None or abs(shift - target) < abs(best[1] - target):
            best = (scale, shift, plus, minus)
        if not np.isfinite(shift) or shift == 0:
            break
        # Clip the size of the correction, never its sign: a knob whose saved
        # direction points the other way needs a negative factor, and clipping
        # that to a positive floor would leave it pointing the wrong way forever.
        raw = target / shift
        factor = float(np.sign(raw) * np.clip(abs(raw), *clip))
        if abs(factor - 1.0) < 1e-3:
            break
        scale *= factor
    assert best is not None
    return Matched(name=name, plus=best[2], minus=best[3], scale=best[0], shift=best[1], target=target, history=history)


def save_pairs(path: Path, matched: list[Matched], meta: dict | None = None) -> None:
    """Write the pairs as float32 plus a json sidecar of what the matching did."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for m in matched:
        arrays[f"{m.name}|plus"] = m.plus.astype(np.float32)
        arrays[f"{m.name}|minus"] = m.minus.astype(np.float32)
    np.savez(path, names=np.array([m.name for m in matched]), **arrays)
    sidecar = {
        "names": [m.name for m in matched],
        "matching": {m.name: {"scale": m.scale, "shift": m.shift, "target": m.target, "history": m.history} for m in matched},
    }
    if meta:
        sidecar.update(meta)
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=float))


def load_pairs(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{name: (plus, minus)}`` as written by :func:`save_pairs`."""
    with np.load(Path(path)) as z:
        names = [str(n) for n in z["names"]]
        return {
            n: (np.asarray(z[f"{n}|plus"], dtype=np.float64), np.asarray(z[f"{n}|minus"], dtype=np.float64)) for n in names
        }
