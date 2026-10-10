#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/studies/utility-rule/margin_swap_bc.py /workspace/data/offline/runs/bcnv11.s1 --read fork \
    --eval-demos /workspace/data/offline/demos/test.rho100 --choices figures/data/h1/bcnv11.s1.npz \
    > results/utility-rule/margin-fork-bcnv11.s1.txt 2> results/utility-rule/margin-fork-bcnv11.s1.log \
    && echo "bcnv11 ok" >> results/utility-rule/margin-fork.status || echo "bcnv11 FAILED" >> results/utility-rule/margin-fork.status
uv run python scripts/studies/utility-rule/margin_swap_bc.py /workspace/data/offline/runs/scaling/sc11.d512l4.s1 --read fork \
    --eval-demos /workspace/data/offline/demos/scaling/test.rho100 --choices figures/data/scaling/d512l4.npz \
    > results/utility-rule/margin-fork-d512l4.txt 2> results/utility-rule/margin-fork-d512l4.log \
    && echo "d512l4 ok" >> results/utility-rule/margin-fork.status || echo "d512l4 FAILED" >> results/utility-rule/margin-fork.status
echo done >> results/utility-rule/margin-fork.status
