"""Comparing edits: in weight space, per module, and by what they do to the policy.

A raw weight-space cosine between a searched edit ``u`` and the fine-tuned axis
``v`` is capped by whatever in ``v`` is noise: the DRC's flatten dense holds a
tenth of the axis's length and none of its effect, and a minimum-norm ``u`` will
put nothing there. Two companions guard against reading that cap as
disagreement.

* **Per module**: the cosine inside each module and each module's share of
  ``||u||^2``, so a match confined to the input layer shows as one.
* **Function space**: ``J u`` against ``J v``, the first-order change in the
  policy's logits along each direction over a batch of real states, centred
  across actions because a softmax ignores a common shift. It does not depend
  on how parameters are scaled, and a direction's noise that moves no logit
  contributes nothing to it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.route_targets import PAD
from goalmisgen.analysis.weights import cosine


def per_module(u: np.ndarray, v: np.ndarray, groups: dict[str, np.ndarray], order) -> list[dict]:
    """Per module: share of ``||u||^2``, share of ``||v||^2``, and ``cos(u, v)`` inside it."""
    uu, vv = float(u @ u), float(v @ v)
    rows = []
    for name in order:
        a, b = u[groups[name]], v[groups[name]]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        rows.append(
            {
                "module": name,
                "u_share": float(a @ a) / uu if uu else float("nan"),
                "v_share": float(b @ b) / vv if vv else float("nan"),
                "cos": float(a @ b / denominator) if denominator > 0 else float("nan"),
            }
        )
    return rows


def function_tangents(logits_fn, base_flat, batch, directions: dict[str, np.ndarray], roles=None) -> dict[str, np.ndarray]:
    """For each direction, the centred change in logits at every non-padding step (or only ``roles``), flattened.

    ``roles=(FORK,)`` keeps the decision states alone: navigation steps outnumber
    forks ten to one, so an all-steps cosine mostly compares what two edits do to
    walking, not to choosing.
    """
    base = jnp.asarray(base_flat, dtype=jnp.float32)
    role = np.asarray(batch["role"])
    valid = role != PAD if roles is None else np.isin(role, roles)

    def centred(flat):
        logits = logits_fn(flat, batch)
        return logits - logits.mean(-1, keepdims=True)

    out = {}
    for name, direction in directions.items():
        _, tangent = jax.jvp(centred, (base,), (jnp.asarray(direction, dtype=jnp.float32),))
        out[name] = np.asarray(tangent, dtype=np.float64)[valid].ravel()
    return out


def cosine_matrix(vectors: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = list(vectors)
    return {a: {b: float(cosine(vectors[a], vectors[b])) for b in names} for a in names}
