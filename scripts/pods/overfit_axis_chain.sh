#!/usr/bin/env bash
# The whole overfitting study, unattended and resumable, in one tmux:
#
#   TAGS="1k 10k 50k" tmux new-session -d -s overfit \
#       "bash scripts/pods/overfit_axis_chain.sh > /workspace/data/logs/offline-overfit/chain.log 2>&1"
#
# Once, before the rungs: the held-out set every condition is scored on, and the
# published campaign's own sweep re-read on it, so the three conditions differ
# in what they were trained on and in nothing else.
#
# Per rung, in the order that gets an answer soonest:
#   1 subsets   the N levels the base and all 25 arms share          (CPU, seconds)
#   2 base      bcov11.n<N>.s1, the bcnv11 recipe on N levels        (~15 min)
#   3 gap       how much of its competence is memory                 (~1 min)
#   4 arms      the o0 sweep, each arm on those same N levels        (~45 min)
#   5 analysis  027 on held-out levels and on the memorised ones
#   6 control   bcnv11.s1's own parameters with N-level arms, and the same analysis
#
# Each stage skips what is already on the volume, so re-running resumes. The
# rungs run in the order given: put the one that answers the question first.
# Progress: grep -E "rung|stage|FAILED|DONE" <the log>

set -uo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/../.."
POD=scripts/pods/overfit_axis_pod.sh
DATA="${DATA:-/workspace/data}"
RUNS="${DATA}/offline/runs"
LOGS="${DATA}/logs/offline-overfit"
TAGS="${TAGS:-1k}"
SEED="${SEED:-1}"
CONTROL_BASE="${CONTROL_BASE:-bcnv11.s1}"
mkdir -p "${LOGS}"
RESULTS="${DATA}/offline/results"


stamp() { date -u +%FT%TZ; }
stage() { echo "$(stamp) stage $*"; }

stage "holdout"
bash ${POD} holdout || { echo "$(stamp) holdout FAILED"; exit 1; }

# The campaign's published sweep (50k fresh levels per arm, generalising base),
# re-measured on the same held-out levels as everything below.
stage "reference ${CONTROL_BASE}"
if [ ! -f "${RESULTS}/overfit.value_axis.${CONTROL_BASE}.${SWEEP:-o0}.json" ]; then
    uv run python experiments/027_bc_value_axis.py "${RUNS}/${CONTROL_BASE}" --sweep "${SWEEP:-o0}" \
        --steps 1000 --demos "${DATA}/offline/demos/overfit/holdout.rho100" \
        --json "${RESULTS}/overfit.value_axis.${CONTROL_BASE}.${SWEEP:-o0}.json" \
        > "${RESULTS}/overfit.value_axis.${CONTROL_BASE}.${SWEEP:-o0}.txt" 2>&1 \
        || echo "$(stamp) reference ${CONTROL_BASE} FAILED"
fi

for tag in ${TAGS}; do
    echo "$(stamp) rung ${tag}"
    run="bcov11.n${tag}.s${SEED}"
    control="${CONTROL_BASE}.n${tag}"

    stage "subsets ${tag}"
    bash ${POD} subsets "${tag}" || { echo "$(stamp) subsets ${tag} FAILED"; continue; }

    stage "base ${run}"
    bash ${POD} base "${tag}" "${SEED}" || { echo "$(stamp) base ${run} FAILED"; continue; }
    [ -f "${RUNS}/${run}/done.json" ] || { echo "$(stamp) base ${run} did not finish"; continue; }

    stage "gap ${run}"
    uv run python scripts/studies/overfit/overfit_gap.py "${RUNS}/${run}" \
        own="${DATA}/offline/demos/overfit/1.00-0.50.n${tag}.train.rho100" \
        holdout="${DATA}/offline/demos/overfit/holdout.rho100" \
        --json "${DATA}/offline/results/overfit.gap.${run}.json" \
        2>&1 | tee "${DATA}/offline/results/overfit.gap.${run}.txt" | tail -6

    stage "arms ${run}"
    bash ${POD} arms "${run}" "${tag}"
    stage "analysis ${run}"
    bash ${POD} analysis "${run}" "${tag}"

    stage "control ${control}"
    bash ${POD} sibling "${CONTROL_BASE}" "${control}" || echo "$(stamp) sibling ${control} FAILED"
    bash ${POD} arms "${control}" "${tag}"
    bash ${POD} analysis "${control}" "${tag}"

    echo "$(stamp) rung ${tag} DONE"
done
echo "$(stamp) OVERFIT_CHAIN_DONE"
