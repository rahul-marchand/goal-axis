#!/usr/bin/env bash
# Is the value axis there in a route model that has *memorised* its data?
#
#   bash scripts/overfit_axis_pod.sh holdout              # levels held out from both pools at once
#   bash scripts/overfit_axis_pod.sh subsets  1k          # the N levels every run in the rung shares
#   bash scripts/overfit_axis_pod.sh base     1k 1        # bcov11.n1k.s1: the bcnv11 recipe on N levels
#   bash scripts/overfit_axis_pod.sh sibling  bcnv11.s1 bcnv11.s1.n1k   # same base, arms trained on N levels
#   bash scripts/overfit_axis_pod.sh arms     bcov11.n1k.s1 1k          # the 25-arm o0 sweep, on N levels
#   bash scripts/overfit_axis_pod.sh analysis bcov11.n1k.s1 1k          # 027 held-out and memorised, plus the gap
#
# The campaign this forks (offline_value_axis_pod.sh) trains its base on 900k
# demonstrations and each arm on 50k fresh ones, and finds an axis: arms on
# opposite sides of the base oppose, an axis fitted without an arm reads that
# arm's offset back, and writing offset * axis into the base moves its exchange
# rate. The question here is whether any of that needs the base to have learned
# the task, or whether fine-tuning alone manufactures it.
#
# So a rung holds everything at one training-set size N:
#
#   bcov11.n<N>.s<seed>   the base, same recipe as bcnv11 but on N levels
#   its arms             fine-tuned on *the same N levels* at shifted values,
#                        so an arm's diff from the base is what changing what an
#                        objective is worth does, with the data held fixed
#   bcnv11.s1.n<N>       the control: the generalising base, arms on those same
#                        N levels. It separates "the base is a memoriser" from
#                        "the arms had little data", which the rung otherwise
#                        confounds.
#
# Every arm of every rung trains on the same mazes; only the values differ.
# Fixed values consume no randomness, so the layouts at every values tag are
# identical and the split is decided by hashing the maze, which makes "the
# first N of each tag" the same N levels. ``make_demo_subset.py --check``
# asserts it rather than trusting it.
#
# Arms use the campaign's own budget and rate (1000 steps, constant 1e-4,
# warmup 50, batch 256) so that a rung's axis is measured in the units the
# published one was. What changes between rungs is data, and nothing else.
#
# Behaviour is read on ``demos/overfit/holdout.rho100``, not on ``test.rho100``,
# because the two bases being compared were trained from different pools and
# nothing else is held out from both. ``test.rho100`` is clean for bcnv11 and
# holds 53 of every rung's 1k levels; the arms' own test set is clean for the
# rungs and sits 89% inside bcnv11's training set. The held-out set is
# ``test.rho100`` minus the arm pool's training split: clean for every
# condition, and still a subset of the levels the published numbers were
# measured on.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd "$(dirname "$0")/.."

DATA="${DATA:-/workspace/data}"
DEMOS="${DATA}/offline/demos"
SUBSETS="${DEMOS}/overfit"
RUNS="${DATA}/offline/runs"
LOGS="${DATA}/logs/offline-overfit"
RESULTS="${DATA}/offline/results"
BASE_STEPS="${BASE_STEPS:-30000}"
BASE_LR="${BASE_LR:-3e-4}"
FT_STEPS="${FT_STEPS:-1000}"
FT_LR="${FT_LR:-1e-4}"
FT_WARMUP="${FT_WARMUP:-50}"
SWEEP="${SWEEP:-o0}"
BASE_TAG="1.00-0.50"
HOLDOUT="${SUBSETS}/holdout.rho100"
HOLDOUT_LEVELS="${HOLDOUT_LEVELS:-2048}"
mkdir -p "${SUBSETS}" "${RUNS}" "${LOGS}" "${RESULTS}"

stamp() { date -u +%FT%TZ; }

# ``1k`` -> 1000. The tag is what appears in run and subset names, so it has to
# be a round number for the same reason arm budgets do: two things claiming the
# same size should have it.
levels_of() {
    case "$1" in
        *k) echo $(( ${1%k} * 1000 )) ;;
        *M) echo $(( ${1%M} * 1000000 )) ;;
        *) echo "level tag $1 should look like 1k or 50k" >&2; exit 2 ;;
    esac
}

tags_of_sweep() {  # every values tag the sweep touches, base tag included
    { echo "${BASE_TAG}"; uv run python scripts/value_axis_arms.py --steps "${FT_STEPS}" --objective "${SWEEP#o}" \
        | awk '{print $5}'; } | sort -u
}

holdout() {
    if [ -f "${HOLDOUT}/meta.json" ]; then echo "have ${HOLDOUT}"; return 0; fi
    uv run python scripts/make_demo_subset.py --demos "${DEMOS}/test.rho100" --n "${HOLDOUT_LEVELS}" \
        --exclude "${DEMOS}/arms/${BASE_TAG}.train.rho100" --out "${HOLDOUT}"
}

subsets() {
    tag="$1"; n=$(levels_of "${tag}")
    for values in $(tags_of_sweep); do
        src="${DEMOS}/arms/${values}.train.rho100"
        out="${SUBSETS}/${values}.n${tag}.train.rho100"
        [ -d "${src}" ] || { echo "MISSING ${src}"; exit 1; }
        if [ -f "${out}/meta.json" ]; then echo "have ${out}"; continue; fi
        uv run python scripts/make_demo_subset.py --demos "${src}" --n "${n}" --out "${out}" \
            --check "${DEMOS}/arms/${BASE_TAG}.train.rho100"
    done
}

base() {
    tag="$1"; seed="$2"; n=$(levels_of "${tag}")
    name="bcov11.n${tag}.s${seed}"
    out="${RUNS}/${name}"
    if [ -f "${out}/done.json" ]; then echo "have ${name}"; return 0; fi
    rm -rf "${out}"
    uv run python experiments/023_train_bc.py \
        --demos "${SUBSETS}/${BASE_TAG}.n${tag}.train.rho100" --hide-values \
        --eval holdout="${HOLDOUT}" \
        --out "${out}" --seed "${seed}" --steps "${BASE_STEPS}" --lr "${BASE_LR}" \
        --note "Overfitting rung of the offline value-axis campaign: the bcnv11 recipe (hidden values, rho=1.0, ${BASE_STEPS} steps) on ${n} demonstrations instead of 900k, so the model memorises rather than learns the task. Its arms fine-tune on those same ${n} levels at shifted values." \
        > "${LOGS}/${name}.log" 2>&1
    echo "$(stamp) trained ${name}"
}

# A run directory that shares another's parameters, so the same base can carry a
# second set of arms. The ladder does the same thing for a checkpoint part way
# through training; here it is the *arms* that differ and the base that is held
# fixed, which is the control the rung needs.
sibling() {
    src="${RUNS}/$1"; dst="${RUNS}/$2"
    [ -f "${src}/done.json" ] || { echo "no finished run ${src}" >&2; exit 1; }
    mkdir -p "${dst}"
    cp "${src}/config.json" "${src}/done.json" "${dst}/"
    [ -e "${dst}/checkpoints" ] || ln -s "${src}/checkpoints" "${dst}/checkpoints"
    echo "Arms of $1 trained on a subset of the levels, not on 50k fresh ones. The parameters are $1's own; only ${dst}/arms is new." > "${dst}/note.txt"
    echo "$(stamp) ${dst} shares the parameters of ${src}"
}

arm() {
    run="$1"; tag="$2"; sweep="$3"; offset="$4"; seed="$5"; dirname="$6"; values="$7"
    out="${RUNS}/${run}/arms/${dirname}"
    init=$(ls -d "${RUNS}/${run}"/checkpoints/step_* | sort | tail -n1)
    uv run python experiments/023_train_bc.py \
        --demos "${SUBSETS}/${values}.n${tag}.train.rho100" --init-from "${init}" --schedule constant \
        --eval base="${HOLDOUT}" own="${DEMOS}/arms/${values}.test.rho100" --eval-levels 1024 \
        --out "${out}" --seed "${seed}" --steps "${FT_STEPS}" --lr "${FT_LR}" --warmup "${FT_WARMUP}" \
        --checkpoint-first 100000000 \
        --note "Value-axis arm ${dirname} of ${run}: fine-tuned for ${FT_STEPS} steps at constant lr ${FT_LR} on the *same* $(levels_of "${tag}") levels the base trained on, demonstrated at values ${values}. Data held fixed; only what the objectives are worth changed."
}

arms() {
    run="$1"; tag="$2"
    uv run python scripts/value_axis_arms.py --steps "${FT_STEPS}" --objective "${SWEEP#o}" \
        | while read -r sweep offset seed dirname values; do
        if [ -f "${RUNS}/${run}/arms/${dirname}/done.json" ]; then echo "done ${run}/${dirname}"; continue; fi
        echo "$(stamp) starting ${run}/${dirname}"
        arm "${run}" "${tag}" "${sweep}" "${offset}" "${seed}" "${dirname}" "${values}" \
            > "${LOGS}/${run}.${dirname}.log" 2>&1 || echo "${run}/${dirname} FAILED"
        echo "$(stamp) finished ${run}/${dirname}"
    done
}

analysis() {
    run="$1"; tag="$2"
    own="${SUBSETS}/${BASE_TAG}.n${tag}.train.rho100"
    # The axis, read on held-out levels exactly as the campaign reads it ...
    uv run python experiments/027_bc_value_axis.py "${RUNS}/${run}" --sweep "${SWEEP}" --steps "${FT_STEPS}" \
        --demos "${HOLDOUT}" --json "${RESULTS}/overfit.value_axis.${run}.${SWEEP}.json" \
        > "${RESULTS}/overfit.value_axis.${run}.${SWEEP}.txt" 2>&1 || echo "027 ${run} ${SWEEP} held-out FAILED"
    # ... and again on the levels the base was trained on, where a memoriser has
    # an exchange rate at all. A write that moves one and not the other is an
    # axis that exists only where the routes are stored.
    uv run python experiments/027_bc_value_axis.py "${RUNS}/${run}" --sweep "${SWEEP}" --steps "${FT_STEPS}" \
        --demos "${own}" --json "${RESULTS}/overfit.value_axis.${run}.${SWEEP}.mem.json" \
        > "${RESULTS}/overfit.value_axis.${run}.${SWEEP}.mem.txt" 2>&1 || echo "027 ${run} ${SWEEP} memorised FAILED"
    uv run python scripts/overfit_gap.py "${RUNS}/${run}" \
        own="${own}" holdout="${HOLDOUT}" --json "${RESULTS}/overfit.gap.${run}.json" \
        > "${RESULTS}/overfit.gap.${run}.txt" 2>&1 || echo "gap ${run} FAILED"
    echo "$(stamp) analysis ${run} -> ${RESULTS}/overfit.*.${run}.*"
}

case "${1:-}" in
    holdout) holdout ;;
    subsets) subsets "$2" ;;
    base) base "$2" "$3" ;;
    sibling) sibling "$2" "$3" ;;
    arms) arms "$2" "$3" ;;
    analysis) analysis "$2" "$3" ;;
    *) echo "usage: $0 holdout | subsets TAG | base TAG SEED | sibling SRC DST | arms RUN TAG | analysis RUN TAG" >&2; exit 2 ;;
esac
