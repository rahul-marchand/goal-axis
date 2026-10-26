#!/usr/bin/env bash
# The retargeting study's positive control: a route model with a goal vector per exchange rate.
#
#   bash scripts/pods/goal_vector_base.sh [seed]          # -> offline/runs/bcgv11.s<seed>
#   N=1k bash scripts/pods/goal_vector_base.sh [seed]     # -> offline/runs/bcgvm11.s<seed>
#
# The bcnv11 recipe (hidden values, rho=1.0, 30k steps, lr 3e-4, batch 256),
# trained on the o1 arm sweep's 25 demonstration sets at once: one level pool,
# the poorer objective's value moved from 0.05 to 0.95, so tau runs 1-19 steps.
# Each demonstration is tagged with its set, and the model learns one vector per
# set at the SEP token (ModelConfig.n_goals). With the values hidden that vector
# is the only way to tell the goals apart, so the model has a dial by
# construction. Its own goal is tau = 10 (values 1.0, 0.5). None of the pilot's
# half-integer rates is a training rate, so every arm is a held-out goal.
#
# The sets are the arm pool's training split, which the retargeting pool
# excludes, so retargeting never sees a level the base was trained on.
#
# N (1k, 3k or 10k) trains the "factorised but memorised" control instead
# (THREADS.md RC.13): the same recipe on only the first N mazes of each set, as
# the memoriser bcov11.n1k used 1k. The slices are offline/demos/overfit/
# <tag>.n<N>.train.rho100, the overfitting study's naming; the o1 tags that
# study never built are made here with make_demo_subset.py, checked level for
# level against the null tag's slice.

set -euo pipefail
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
cd "$(dirname "$0")/../.."
ulimit -n "$(ulimit -Hn)"

DATA="${DATA:-/workspace/data}"
DEMOS="${DATA}/offline/demos"
SEED="${1:-1}"
N="${N:-}"
if [ -n "${N}" ]; then NAME="bcgvm11.s${SEED}"; else NAME="bcgv11.s${SEED}"; fi
OUT="${DATA}/offline/runs/${NAME}"
PYTHON="${PYTHON:-uv run --no-sync python}"
mkdir -p "${DATA}/logs/retarget"

if [ -f "${OUT}/done.json" ]; then echo "have ${NAME}"; exit 0; fi
rm -rf "${OUT}"
sets=$(ls -d "${DEMOS}"/arms/1.00-*.train.rho100 | sort)
[ "$(echo "${sets}" | wc -l)" -eq 25 ] || { echo "expected the o1 sweep's 25 sets" >&2; exit 1; }
NOTE="Retargeting positive control: the bcnv11 recipe with a learned goal vector per exchange rate, trained on the o1 sweep's 25 sets (tau 1-19), base goal tau 10. THREADS.md RC.6."
if [ -n "${N}" ]; then
    count="${N%k}000"
    null_slice="${DEMOS}/overfit/1.00-0.50.n${N}.train.rho100"
    [ -d "${null_slice}" ] || ${PYTHON} scripts/data/make_demo_subset.py --demos "${DEMOS}/arms/1.00-0.50.train.rho100" \
        --n "${count}" --out "${null_slice}"
    slices=""
    for set in ${sets}; do
        tag=$(basename "${set}" .train.rho100)
        slice="${DEMOS}/overfit/${tag}.n${N}.train.rho100"
        [ -d "${slice}" ] || ${PYTHON} scripts/data/make_demo_subset.py --demos "${set}" --n "${count}" \
            --check "${null_slice}" --out "${slice}" >> "${DATA}/logs/retarget/${NAME}.subsets.log" 2>&1
        slices="${slices} ${slice}"
    done
    sets="${slices}"
    NOTE="Retargeting control, factorised but memorised: the bcgv11 recipe (a learned goal vector per exchange rate, the o1 sweep's 25 sets, tau 1-19, base goal tau 10) on only the first ${N} mazes of each set, as bcov11.n${N} memorised. THREADS.md RC.13."
fi
${PYTHON} experiments/023_train_bc.py --goal-demos ${sets} --base-tau 10 --hide-values \
    --eval holdout="${DEMOS}/overfit/holdout.rho100" \
    --out "${OUT}" --seed "${SEED}" \
    --note "${NOTE}" \
    > "${DATA}/logs/retarget/${NAME}.train.log" 2>&1
echo "trained ${NAME}"
