#!/usr/bin/env bash
# The retargeting study's positive control, trained on as many mazes as bcnv11 (THREADS.md RC.11).
#
#   bash scripts/pods/goal_vector_random.sh [seed]     # -> offline/runs/bcgvr11.s<seed>
#
# bcgv11 (goal_vector_base.sh) saw the o1 sweep's 50k mazes at all 25 of its
# exchange rates, and navigates fresh mazes worse than bcnv11 (1.03 bits per
# route on the retargeting pool against 0.33), so its null arm keeps improving
# and contaminates the null-arm subtraction. Here every level of bcnv11's own
# training set (train.rho100, ~900k) is re-demonstrated at ONE of the same 25
# rates, drawn uniformly at random with a fixed seed (scripts/data/make_goal_split.py),
# and the model is otherwise the bcgv11 recipe: hidden values, 30k steps, batch
# 256, base goal tau 10, the same held-out evaluation. The goal sets are named
# like the o1 sweep's, so the goal ids and taus are bcgv11.s1's.
#
# The retargeting pool is test.rho100 minus the arm pool's training split, so it
# is held out from these levels too; the build checks it.

set -euo pipefail
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
cd "$(dirname "$0")/../.."
ulimit -n "$(ulimit -Hn)"

DATA="${DATA:-/workspace/data}"
DEMOS="${DATA}/offline/demos"
SEED="${1:-1}"
SPLIT_SEED="${SPLIT_SEED:-0}"
NAME="bcgvr11.s${SEED}"
OUT="${DATA}/offline/runs/${NAME}"
GOALS="${DEMOS}/retarget/goals-random.s${SPLIT_SEED}"
PYTHON="${PYTHON:-uv run --no-sync python}"
mkdir -p "${DATA}/logs/retarget"

if [ -f "${OUT}/done.json" ]; then echo "have ${NAME}"; exit 0; fi
like=$(ls -d "${DEMOS}"/arms/1.00-*.train.rho100 | sort)
[ "$(echo "${like}" | wc -l)" -eq 25 ] || { echo "expected the o1 sweep's 25 sets" >&2; exit 1; }

if [ ! -f "${GOALS}/BUILT" ]; then
    rm -rf "${GOALS}"
    ${PYTHON} scripts/data/make_goal_split.py --demos "${DEMOS}/train.rho100" --goals-like ${like} \
        --seed "${SPLIT_SEED}" --out "${GOALS}" > "${DATA}/logs/retarget/goals-random.s${SPLIT_SEED}.log" 2>&1
    ${PYTHON} - "${GOALS}" "${DEMOS}/retarget/pool.rho100" <<'PY'
import pathlib, sys
from goalmisgen.offline.demos import DemoSet, shared_levels
goals, pool = pathlib.Path(sys.argv[1]), DemoSet.load(sys.argv[2])
parts = [DemoSet.load(p) for p in sorted(goals.iterdir()) if p.is_dir()]
# shared_levels is zero across pools, so first make sure the question can be asked by index.
assert all(p.meta["source_fingerprint"] == pool.meta["source_fingerprint"] for p in parts), "different level pools"
shared = sum(shared_levels(p, pool) for p in parts)
if shared:
    sys.exit(f"{shared} training levels are in the retargeting pool")
print(f"{sum(len(p) for p in parts):,} levels, disjoint from the retargeting pool")
PY
    touch "${GOALS}/BUILT"
fi

rm -rf "${OUT}"
sets=$(ls -d "${GOALS}"/1.00-*.train.rho100 | sort)
${PYTHON} experiments/023_train_bc.py --goal-demos ${sets} --goal-levels disjoint --base-tau 10 --hide-values \
    --eval holdout="${DEMOS}/overfit/holdout.rho100" \
    --out "${OUT}" --seed "${SEED}" \
    --note "Retargeting positive control on bcnv11's ~900k training mazes: each re-solved at one of bcgv11's 25 exchange rates drawn at random (goals-random.s${SPLIT_SEED}), a learned goal vector per rate, base goal tau 10. THREADS.md RC.11." \
    > "${DATA}/logs/retarget/${NAME}.train.log" 2>&1
echo "trained ${NAME}"
