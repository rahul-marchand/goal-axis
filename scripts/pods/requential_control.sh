#!/usr/bin/env bash
# RC.14: requential coding of the positive control, unattended and resumable:
#
#   tmux new-session -d -s requential \
#       "bash scripts/pods/requential_control.sh > /workspace/data/logs/retarget-requential/chain.log 2>&1"
#
# The goal-slot agent (AGENT, default bcgvr11.s1) is coded at every goal in
# TAUS by requential coding, with the student training every weight at the
# pilot learner's rate (3e-5) and one of three teachers:
#
#   dial      the goal vector alone, at the dial-only learner's rate (3e-3)
#   full      every weight, at 3e-5, the student's own recipe
#   full-ema  the same, taught from its moving average (teacher smoothing)
#
# If the goal slot is a dial, the dial teacher should cost least and the full
# teacher should drift (THREADS.md RC.3, RC.7). CALLS REC calls of CALL_BATCH
# routes each; the large batch amortises the per-call overhead. One job is one
# (teacher, goal); a job whose JSON exists is skipped, so re-running resumes.
#
# Progress: grep -E "start|done|FAILED|DONE" <the log>

set -uo pipefail
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$(dirname "$0")/../.."
ulimit -n "$(ulimit -Hn)"

DATA="${DATA:-/workspace/data}"
AGENT="${AGENT:-bcgvr11.s1}"
POOL="${DATA}/offline/demos/retarget/pool.rho100"
OUT="${DATA}/offline/results/retarget-requential"
LOGS="${DATA}/logs/retarget-requential"
PYTHON="${PYTHON:-uv run --no-sync python}"
JOBS="${JOBS:-3}"
TAUS="${TAUS:-0.5 2.5 6.5 8.5 10 12.5 14.5 18.5}"
TEACHERS="${TEACHERS:-dial full full-ema}"
CALLS="${CALLS:-64}"
CALL_BATCH="${CALL_BATCH:-4096}"
mkdir -p "${OUT}/${AGENT}" "${LOGS}"

stamp() { date -u +%FT%TZ; }

jobs() {
    for teacher in ${TEACHERS}; do
        for tau in ${TAUS}; do echo "${teacher} ${tau}"; done
    done
}

run_job() {
    teacher="$1" tau="$2"
    case "${teacher}" in
        dial) flags="--teacher-lr 3e-3 --teacher-only goal_embedding" ;;
        full) flags="--teacher-lr 3e-5" ;;
        full-ema) flags="--teacher-lr 3e-5 --teacher-ema" ;;
        *) echo "$(stamp) FAILED unknown teacher ${teacher}"; return 0 ;;
    esac
    arm="tau${tau}.req-${teacher}"
    out="${OUT}/${AGENT}/${arm}.json"
    [ -f "${out}" ] && { echo "$(stamp) have ${arm}"; return 0; }
    echo "$(stamp) start ${arm}"
    # shellcheck disable=SC2086
    if ${PYTHON} scripts/studies/retarget/retarget_cost.py "${DATA}/offline/runs/${AGENT}" --pool "${POOL}" --tau "${tau}" \
        --lr 3e-5 --coding requential --calls "${CALLS}" --call-batch "${CALL_BATCH}" ${flags} --out "${out}" \
        > "${LOGS}/${AGENT}.${arm}.log" 2>&1; then
        echo "$(stamp) done ${arm}"
    else
        echo "$(stamp) FAILED ${arm} (see ${LOGS}/${AGENT}.${arm}.log)"
    fi
}
export -f run_job stamp
export DATA AGENT POOL OUT LOGS PYTHON CALLS CALL_BATCH

jobs | xargs -P "${JOBS}" -L 1 bash -c 'run_job "$@"' _
echo "$(stamp) REQUENTIAL_CONTROL_DONE"
