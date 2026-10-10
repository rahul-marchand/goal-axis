#!/usr/bin/env bash
# Run the distance-gap probe on all three bcnv11 seeds, sequentially.
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
for s in 1 2 3; do
    uv run python scripts/studies/utility-rule/probe_bc_distance.py /workspace/data/offline/runs/bcnv11.s$s \
        > results/utility-rule/gap-probe-bcnv11.s$s.txt 2> results/utility-rule/gap-probe-bcnv11.s$s.log \
        && echo "s$s ok" >> results/utility-rule/gap-probe.status \
        || echo "s$s FAILED" >> results/utility-rule/gap-probe.status
done
echo done >> results/utility-rule/gap-probe.status
