"""Fine-tune a cleanba agent with some modules frozen, without touching cleanba.

Asking whether a network *can* carry a value change downstream of its input
layer, when SGD is not allowed to use the input layer, needs an ordinary arm
(``013``) with those parameters held fixed. cleanba's only freezing option is
for its noop-head fine-tune, so this wraps its optimiser from our side instead:
:func:`frozen_modules` replaces ``cleanba.cleanba_impala.make_optimizer`` with
one whose updates are zeroed on the named modules
(:func:`goalmisgen.analysis.layers.module_of` names).

The wrapper zeroes the *final* updates rather than the gradients, so momentum,
weight decay and anything else inside the chain cannot leak into a frozen
parameter. It also returns the inner optimiser's state object unchanged, because
the training loop reads ``opt_state.hyperparams`` for logging and a
``multi_transform`` or ``chain`` wrapper would hide it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import jax
import optax

from goalmisgen.analysis.layers import module_of, path_string


def frozen_leaves(tree, modules: Sequence[str]) -> list[str]:
    """Paths of the leaves of ``tree`` that belong to ``modules``."""
    return [
        path_string(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]
        if module_of(path_string(path)) in modules
    ]


def freeze(tx: optax.GradientTransformation, modules: Sequence[str]) -> optax.GradientTransformation:
    """``tx`` with its updates zeroed on every leaf inside ``modules``.

    Raises at ``init`` if no parameter matches, since a misspelt module name
    would otherwise train an ordinary arm and label it frozen.
    """
    modules = tuple(modules)

    def init(params):
        if not frozen_leaves(params, modules):
            raise ValueError(f"no parameter belongs to {modules}")
        return tx.init(params)

    def update(updates, state, params=None, **extra):
        updates, state = tx.update(updates, state, params, **extra)
        updates = jax.tree_util.tree_map_with_path(
            lambda path, u: u * 0 if module_of(path_string(path)) in modules else u, updates
        )
        return updates, state

    return optax.GradientTransformationExtraArgs(init, update)


@contextlib.contextmanager
def frozen_modules(modules: Sequence[str]):
    """Within the block, every optimiser cleanba builds leaves ``modules`` untouched."""
    import cleanba.cleanba_impala as impala

    original = impala.make_optimizer

    def make_optimizer(*args, **kwargs):
        return freeze(original(*args, **kwargs), modules)

    impala.make_optimizer = make_optimizer
    try:
        yield
    finally:
        impala.make_optimizer = original
