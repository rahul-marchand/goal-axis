"""How much of a route model's cell token can an edit at ``cell_in`` reach?

    uv run python scripts/studies/gain-vs-factorised/token_budget.py /workspace/data/offline/runs/scaling/sc11.d512l4.s1 \
        --demos /workspace/data/offline/demos/scaling --json figures/data/token_budget/d512l4.json

The dilution account of the route model's weak input knobs says a cell token is
mostly position, so a gain or a rotation on ``cell_in`` turns it hardly at all
(:mod:`goalmisgen.analysis.token_budget`). This prints the colour channels', the
remaining content's and the position embeddings' share of the token, of its
variation across cells, and of what survives the model's own first LayerNorm.

The share after the norm is the one to read: it is the fraction of the post-norm
token an input-layer edit can move at all. Run it across the width campaign to
see whether it falls as ``d_model`` grows, which is what dilution predicts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.token_budget import budget, cell_token_parts
from goalmisgen.envs.observation import FIRST_FEATURE_CHANNEL
from goalmisgen.provenance import header

COLOUR = (FIRST_FEATURE_CHANNEL, FIRST_FEATURE_CHANNEL + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True, help="Directory holding test.rho100.")
    parser.add_argument("--n-levels", type=int, default=256)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())

    import jax.numpy as jnp

    from goalmisgen.analysis.closed_loop import draw_agent
    from goalmisgen.analysis.route_targets import route_targets, usable
    from goalmisgen.offline.axis import load_base
    from goalmisgen.offline.demos import DemoSet

    base = load_base(args.run)
    config = base.model.config
    print(f"base   {args.run.name} @ step {base.step:,}  ({base.flat.size:,} parameters)")
    print(f"       d_model {config.d_model}, {config.n_layers} blocks, {config.n_channels} input channels, size {config.size}")

    demos = DemoSet.load(args.demos / "test.rho100", hide_values=base.hide_values)
    levels = []
    for index in range(len(demos)):
        level = demos.level(index)
        if usable(level):
            levels.append(level)
        if len(levels) == args.n_levels:
            break
    targets = route_targets(levels, size=config.size, n_channels=config.n_channels)
    observations = np.asarray(draw_agent(jnp.asarray(targets.static), jnp.asarray(targets.positions[:, 0, 0])))
    cells = observations.reshape(len(levels), config.n_cells, config.n_channels)
    print(f"levels {len(levels):,} of {args.demos / 'test.rho100'}")

    tree = base.params["params"] if "params" in base.params else base.params
    norm = tree["block_0"]["ln_attention"]
    parts = cell_token_parts(base.params, cells, size=config.size, colour=COLOUR)
    shares = budget(parts, np.asarray(norm["scale"]), np.asarray(norm["bias"]))

    print("\n=== what a cell token is made of; shares of ||x||^2 and of its variation across cells ===\n")
    print(f"  {'part':<16}{'of token':>10}{'of variation':>14}{'reach after LN':>16}")
    for name, share in shares.items():
        print(f"  {name:<16}{share.norm:>10.4f}{share.variation:>14.4f}{share.after_norm:>16.4f}")
    print("\n  reach after LN = relative change in the post-LayerNorm token when that part is removed;")
    print("  it bounds what an edit confined to that part of the input can do.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "run": str(args.run),
                    "step": base.step,
                    "parameters": int(base.flat.size),
                    "d_model": int(config.d_model),
                    "n_layers": int(config.n_layers),
                    "n_levels": len(levels),
                    "shares": {name: vars(share) for name, share in shares.items()},
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
