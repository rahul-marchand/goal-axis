#!/usr/bin/env bash
# The retargeting-cost pilot on the route model, unattended and resumable:
#
#   tmux new-session -d -s retarget \
#       "bash scripts/pods/retarget_chain.sh > /workspace/data/logs/retarget/chain.log 2>&1"
#
# One job is one (agent, exchange rate, replicate) prequential code, written to
# offline/results/retarget/<agent>/<arm>.json; a job whose JSON exists is
# skipped, so re-running resumes. Jobs run JOBS at a time on the one GPU.
#
# The grid, in the order that answers soonest:
#   1 main      the learner (bcnv11.s1) and the memoriser (bcov11.n1k.s1) at every rate
#   2 noise     both of them at tau 2.5 and the null arm, varying one stage at a time:
#               data ordering, learner seed, the size of V, the learning rate
#   3 seeds     bcnv11.s2 and .s3 at every rate: between-base spread
#   4 scratch   a fresh initialisation of bcnv11's architecture, the "learn everything" anchor
#   5 goalvec   the positive control bcgv11.s1 (goal_vector_base.sh; or GOALVEC_AGENTS): every rate with the
#               full-weight learner, and the dial-only learner (goal vector alone, lr 3e-3)
#   6 subspace  SUBSPACE_AGENTS (default bcnv11.s1) with the learner restricted to the coefficients of
#               each direction set in SUBSPACES (make_subspace.py's names under offline/subspaces/<agent>/;
#               a directory such as axis-heldout holds one file per rate, tau<tau>.npz), at SUBSPACE_LR
#               (3e-2: the fastest of 3e-3..1e-1 to the same V along the axis, pick_lr.bcnv11.s1.axis.txt; THREADS.md RC.12)
#
# PHASES picks which of these to run (default all four, in that order),
# NOISE_AGENTS which agents get the noise grid, GOALVEC_AGENTS which goal-vector
# agents the goalvec phase runs (bcgvr11.s1 too, RC.11), GOALVEC_LEARNERS which of its learners
# ("full dial"; naming one lets a second chain take the other), and TAUS the rates. The 2026-10-10 pilot dropped
# the memoriser's noise grid and the scratch anchor once both turned out to fail
# the competence condition (THREADS.md RC.1), which halved the remaining compute.
#
# Progress: grep -E "start|done|FAILED|CHAIN" <the log>

set -uo pipefail
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$(dirname "$0")/../.."
ulimit -n "$(ulimit -Hn)"

DATA="${DATA:-/workspace/data}"
RUNS="${DATA}/offline/runs"
POOL="${DATA}/offline/demos/retarget/pool.rho100"
OUT="${DATA}/offline/results/retarget"
LOGS="${DATA}/logs/retarget"
JOBS="${JOBS:-3}"
PYTHON="${PYTHON:-uv run --no-sync python}"
TAUS="${TAUS:-0.5 2.5 6.5 8.5 10 12.5 14.5 18.5}"
PHASES="${PHASES:-main noise seeds scratch}"
NOISE_AGENTS="${NOISE_AGENTS:-bcnv11.s1 bcov11.n1k.s1}"
GOALVEC_AGENTS="${GOALVEC_AGENTS:-bcgv11.s1}"
GOALVEC_LEARNERS="${GOALVEC_LEARNERS:-full dial}"
SUBSPACE_AGENTS="${SUBSPACE_AGENTS:-bcnv11.s1}"
SUBSPACES="${SUBSPACES:-axis axis-heldout axis-drift pca1 pca2 pca4 random1 random4}"
SUBSPACE_LR="${SUBSPACE_LR:-3e-2}"
SUBSPACE_DIR="${DATA}/offline/subspaces"
mkdir -p "${OUT}" "${LOGS}"

stamp() { date -u +%FT%TZ; }

# agent tau ordering seed n_valid lr [restriction: a top-level tensor, or sub:<direction set>]
jobs() {
    for phase in ${PHASES}; do
        case "${phase}" in
            main)
                for agent in bcnv11.s1 bcov11.n1k.s1; do
                    for tau in ${TAUS}; do echo "${agent} ${tau} 0 0 1024 3e-5"; done
                done ;;
            noise)
                for agent in ${NOISE_AGENTS}; do
                    for tau in 2.5 10; do
                        echo "${agent} ${tau} 1 0 1024 3e-5"
                        echo "${agent} ${tau} 2 0 1024 3e-5"
                        echo "${agent} ${tau} 0 1 1024 3e-5"
                        echo "${agent} ${tau} 0 0 256 3e-5"
                        echo "${agent} ${tau} 0 0 1024 1e-5"
                    done
                done ;;
            seeds)
                for agent in bcnv11.s2 bcnv11.s3; do
                    for tau in ${TAUS}; do echo "${agent} ${tau} 0 0 1024 3e-5"; done
                done ;;
            scratch)
                for tau in 10 2.5; do echo "scratch:bcnv11.s1 ${tau} 0 0 1024 3e-5"; done ;;
            goalvec)
                for agent in ${GOALVEC_AGENTS}; do
                    case " ${GOALVEC_LEARNERS} " in *" full "*)
                        for tau in ${TAUS}; do echo "${agent} ${tau} 0 0 1024 3e-5"; done ;; esac
                    case " ${GOALVEC_LEARNERS} " in *" dial "*)
                        for tau in ${TAUS}; do echo "${agent} ${tau} 0 0 1024 3e-3 goal_embedding"; done ;; esac
                done ;;
            subspace)
                for agent in ${SUBSPACE_AGENTS}; do
                    for name in ${SUBSPACES}; do
                        for tau in ${TAUS}; do echo "${agent} ${tau} 0 0 1024 ${SUBSPACE_LR} sub:${name}"; done
                    done
                done ;;
            *) echo "unknown phase ${phase}" >&2; exit 2 ;;
        esac
    done
}

run_job() {
    agent="$1" tau="$2" ordering="$3" seed="$4" n_valid="$5" lr="$6" restrict="${7:-}"
    only="" sub="" subspace=""
    case "${restrict}" in
        sub:*) sub="${restrict#sub:}" ;;
        *) only="${restrict}" ;;
    esac
    arm="tau${tau}.o${ordering}.s${seed}"
    [ "${n_valid}" != 1024 ] && arm="${arm}.v${n_valid}"
    [ "${lr}" != 3e-5 ] && arm="${arm}.lr${lr}"
    [ -n "${only}" ] && arm="${arm}.only-${only}"
    if [ -n "${sub}" ]; then
        arm="${arm}.sub-${sub}"
        subspace="${SUBSPACE_DIR}/${agent}/${sub}.npz"
        [ -d "${SUBSPACE_DIR}/${agent}/${sub}" ] && subspace="${SUBSPACE_DIR}/${agent}/${sub}/tau${tau}.npz"
        [ -f "${subspace}" ] || { echo "$(stamp) FAILED ${agent}/${arm} (no ${subspace})"; return 0; }
    fi
    case "${agent}" in
        scratch:*) base="random:${RUNS}/${agent#scratch:}" ;;
        *) base="${RUNS}/${agent}" ;;
    esac
    name="${agent/:/.}"  # scratch:bcnv11.s1 -> scratch.bcnv11.s1
    out="${OUT}/${name}/${arm}.json"
    [ -f "${out}" ] && { echo "$(stamp) have ${name}/${arm}"; return 0; }
    echo "$(stamp) start ${name}/${arm}"
    if ${PYTHON} scripts/studies/retarget/retarget_cost.py "${base}" --pool "${POOL}" --tau "${tau}" \
        --ordering "${ordering}" --seed "${seed}" --n-valid "${n_valid}" --lr "${lr}" ${only:+--only "${only}"} ${subspace:+--subspace "${subspace}"} --out "${out}" \
        > "${LOGS}/${name}.${arm}.log" 2>&1; then
        echo "$(stamp) done ${name}/${arm}"
    else
        echo "$(stamp) FAILED ${name}/${arm} (see ${LOGS}/${name}.${arm}.log)"
    fi
}
export -f run_job stamp
export RUNS POOL OUT LOGS PYTHON SUBSPACE_DIR

jobs | xargs -P "${JOBS}" -L 1 bash -c 'run_job "$@"' _
echo "$(stamp) RETARGET_CHAIN_DONE"
