"""The route model under the cheapest-edit search: teacher-forced logits over both branches.

The prefix LM reads the maze once and emits moves, so teacher forcing is exact:
feed a branch's moves as the action tokens and read the head at SEP and at each
action position. Position ``t`` predicts the move from the ``t``-th cell of the
branch; position ``length`` predicts EOS, which is added to the targets here as
a fifth class so a route that fails to stop is penalised as a navigation error.

Behaviour is still read the way ``027`` reads it -- greedy decode, replayed
under the environment's rules -- so only the logits are new.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.analysis.route_targets import BRANCH, POOR, RICH, RouteTargets


def with_eos(targets: RouteTargets, max_actions: int) -> dict[str, np.ndarray]:
    """Arrays for :func:`goalmisgen.edits.core.route_loss` over five classes, one step longer."""
    n, _, steps, _ = targets.toward_rich.shape
    if steps > max_actions:
        raise ValueError(f"routes of {steps} moves do not fit a model with max_actions={max_actions}")
    length = steps + 1
    shape = (n, 2, length, 5)
    toward_rich = np.zeros(shape, dtype=bool)
    toward_poor = np.zeros(shape, dtype=bool)
    toward_rich[:, :, :steps, :4] = targets.toward_rich
    toward_poor[:, :, :steps, :4] = targets.toward_poor
    role = np.zeros((n, 2, length), dtype=np.int8)
    role[:, :, :steps] = targets.role
    rows = np.arange(n)
    for branch, mask in ((RICH, toward_rich), (POOR, toward_poor)):
        end = targets.lengths[:, branch]
        mask[rows, branch, end] = False
        mask[rows, branch, end, 4] = True
        role[rows, branch, end] = BRANCH
    actions = np.full((n, 2, max_actions), -1, dtype=np.int32)
    actions[:, :, :steps] = targets.moves
    return {
        "static": targets.static,
        "start": targets.positions[:, RICH, 0],
        "actions": actions,
        "toward_rich": toward_rich,
        "toward_poor": toward_poor,
        "role": role,
        "gap": targets.gap,
        "rich_above": targets.rich_above,
    }


def make_logits_fn(model, unravel):
    """``(flat, batch) -> (N, 2, T + 1, 5)`` logits for both branches."""
    from goalmisgen.analysis.closed_loop import draw_agent

    def logits_fn(flat, batch):
        params = unravel(flat)
        obs = draw_agent(batch["static"], batch["start"])
        n, _, max_actions = batch["actions"].shape
        length = batch["role"].shape[-1]
        obs = jnp.repeat(obs[:, None], 2, axis=1).reshape(n * 2, *obs.shape[1:])
        logits, _ = model.apply(params, obs, batch["actions"].reshape(n * 2, max_actions))
        return logits[:, :length].reshape(n, 2, length, -1)

    return jax.checkpoint(logits_fn)
