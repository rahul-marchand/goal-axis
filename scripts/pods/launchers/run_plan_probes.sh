#!/bin/bash
# Will-visit plan probes + heatmap dumps across the four blog-post models.
# DRC / ResNet / ViT probed on the same dataset test split (levels are
# byte-identical across them given identical env settings); the BC model reads
# its own demo pool's test set — different levels, said in the caption.
set -uo pipefail
cd /workspace/GoalMisgeneralisation
UV=/root/.local/bin/uv
LEVELS=/workspace/data/levels/values/1.00-0.50@1M
DEMOS=/workspace/data/offline/demos/scaling/test.rho100
BC_RUN=/workspace/data/offline/runs/scaling/sc11.d512l4.s1
OUT=/workspace/data/figures-data/plan-probes
mkdir -p "$OUT" results
STATUS=results/plan-probes.status
: > "$STATUS"

cp_of() { python3 -c "import json;print(json.load(open('$1/BASE.json'))['checkpoint'])"; }
DRC=/workspace/data/runs/novalue11.s1234/$(cp_of /workspace/data/runs/novalue11.s1234)
RESNET=/workspace/data/runs/resnet11novalue.s1234-cp130M/$(cp_of /workspace/data/runs/resnet11novalue.s1234-cp130M)
VIT=/workspace/data/runs/vit11novalue.s1234/$(cp_of /workspace/data/runs/vit11novalue.s1234)

run() { # name, command...
  local name=$1; shift
  echo "=== $name: $*"
  if "$@" > "results/$name.txt" 2> "results/$name.err"; then
    echo "$name ok" >> "$STATUS"
  else
    echo "$name FAILED" >> "$STATUS"; tail -5 "results/$name.err"
  fi
}

# --- smoke the two patched load paths with tiny counts before spending time
run smoke-resnet $UV run python experiments/003_probe_plan.py "$RESNET" \
    --levels "$LEVELS" --split test --train-episodes 8 --test-episodes 8 --num-envs 8 --no-untrained
run smoke-bc $UV run python experiments/025_bc_probe_plan.py "$BC_RUN" \
    --demos "$DEMOS" --train-episodes 16 --test-episodes 16 --layers 4 --no-untrained
grep -q FAILED "$STATUS" && { echo "SMOKE FAILED"; exit 1; }

# --- probe tables (headline AUC + untrained + observation controls, think 0;
# --- the DRC also gets think 4 for the refinement clause)
run probe-plan-novalue11.s1234 $UV run python experiments/003_probe_plan.py "$DRC" \
    --levels "$LEVELS" --split test --think 0 4 --test-correlations 1.0
run probe-plan-resnet11novalue.cp130M $UV run python experiments/003_probe_plan.py "$RESNET" \
    --levels "$LEVELS" --split test --think 0 --test-correlations 1.0
run probe-plan-vit11novalue.s1234 $UV run python experiments/003_probe_plan.py "$VIT" \
    --levels "$LEVELS" --split test --think 0 --test-correlations 1.0
run probe-plan-sc11.d512l4 $UV run python experiments/025_bc_probe_plan.py "$BC_RUN" \
    --demos "$DEMOS" --train-episodes 512 --test-episodes 256

# --- heatmap dumps (same seeds -> same 12 example levels across the RL trio)
run heatmap-novalue11.s1234 $UV run python scripts/measure/plan_heatmap.py "$DRC" \
    --levels "$LEVELS" --split test --out "$OUT/plan.novalue11.s1234.npz"
run heatmap-resnet11novalue.cp130M $UV run python scripts/measure/plan_heatmap.py "$RESNET" \
    --levels "$LEVELS" --split test --out "$OUT/plan.resnet11novalue.cp130M.npz"
run heatmap-vit11novalue.s1234 $UV run python scripts/measure/plan_heatmap.py "$VIT" \
    --levels "$LEVELS" --split test --out "$OUT/plan.vit11novalue.s1234.npz"
run heatmap-sc11.d512l4 $UV run python scripts/measure/bc_plan_heatmap.py "$BC_RUN" \
    --demos "$DEMOS" --out "$OUT/plan.sc11.d512l4.npz"

echo "PLAN_PROBES_DONE" >> "$STATUS"
echo PLAN_PROBES_DONE
