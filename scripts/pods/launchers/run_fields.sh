#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
for s in 1 2 3; do
    uv run python scripts/studies/utility-rule/distance_field_bc.py /workspace/data/offline/runs/bcnv11.s$s \
        > results/utility-rule/distance-field-bcnv11.s$s.txt 2> results/utility-rule/distance-field-bcnv11.s$s.log \
        && echo "s$s ok" >> results/utility-rule/distance-field.status \
        || echo "s$s FAILED" >> results/utility-rule/distance-field.status
done
echo done >> results/utility-rule/distance-field.status
