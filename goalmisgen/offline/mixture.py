"""Several demonstration sets drawn from as one.

Task-agnostic: it reads only what the trainer reads (observations, routes,
lengths and the header), so the maze's goal mixture and Craftax's DAgger pool
both build on it. It moved here from ``craftax/dagger.py``, which still
re-exports it.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class MixedDemoSet:
    """Several demonstration sets drawn from as one, each repeated ``repeats[k]`` times.

    The trainer samples items uniformly, so a small set mixed into a large one
    would barely be seen: 40k visited states in a 30M-item pool are a few
    thousand samples over a whole run. ``repeats`` tiles a part virtually so
    that it makes up the share of samples the caller wants; :func:`with_fraction`
    computes the repeats for a target share.
    """

    parts: tuple
    repeats: tuple = ()

    def __post_init__(self) -> None:
        sizes = {p.size for p in self.parts}
        channels = {p.n_channels for p in self.parts}
        widths = {p.max_actions for p in self.parts}
        if len(sizes) != 1 or len(channels) != 1 or len(widths) != 1:
            raise ValueError(f"parts disagree in shape: sizes {sizes}, channels {channels}, max_actions {widths}")
        repeats = tuple(self.repeats) or (1,) * len(self.parts)
        if len(repeats) != len(self.parts) or any(r < 1 for r in repeats):
            raise ValueError(f"one positive repeat per part, got {repeats}")
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "_offsets", np.cumsum([0] + [len(p) * r for p, r in zip(self.parts, repeats)]))

    @classmethod
    def with_fraction(cls, base, extra, fraction: float) -> "MixedDemoSet":
        """``base`` and ``extra`` mixed so that ``extra`` is about ``fraction`` of the samples."""
        if not 0 < fraction < 1:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        repeat = max(1, round(fraction * len(base) / ((1 - fraction) * max(1, len(extra)))))
        return cls((base, extra), (1, repeat))

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _split(self, indices):
        indices = np.asarray(indices)
        part = np.searchsorted(self._offsets, indices, side="right") - 1
        out = []
        for k in range(len(self.parts)):
            rows = np.nonzero(part == k)[0]
            local = (indices[rows] - self._offsets[k]) % len(self.parts[k])
            out.append((k, rows, local))
        return out

    def _gather(self, indices, method: str) -> np.ndarray:
        indices = np.asarray(indices)
        pieces = []
        order = []
        for k, rows, local in self._split(indices):
            if len(local):
                pieces.append(getattr(self.parts[k], method)(local))
                order.append(rows)
        out = np.concatenate(pieces)
        result = np.empty_like(out)
        result[np.concatenate(order)] = out
        return result

    def observations(self, indices) -> np.ndarray:
        return self._gather(indices, "observations")

    def routes(self, indices) -> np.ndarray:
        return self._gather(indices, "routes")

    @property
    def lengths(self) -> np.ndarray:
        return np.concatenate([np.tile(np.asarray(p.lengths), r) for p, r in zip(self.parts, self.repeats)])

    @property
    def level_index(self) -> np.ndarray:
        return np.concatenate([np.tile(np.asarray(p.level_index), r) for p, r in zip(self.parts, self.repeats)])

    def __getattr__(self, name):  # shape and header attributes: the first part's
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.parts[0], name)

    @property
    def meta(self) -> dict:
        return {
            **self.parts[0].meta,
            "mixed": [{"n": len(p), "repeats": r, "task": p.meta.get("task")} for p, r in zip(self.parts, self.repeats)],
        }

    @property
    def path(self):
        return self.parts[0].path
