#!/bin/bash
# E5, the factorised-value positive control: one gamma=1.0 DRC base trained on the
# legend observation (goalmisgen/envs/legend.py), its o-family value sweep, the
# 014 measures, and the two signature checks the colour bases were read with.
#   SEED=1234 STEPS=150000000 NAME=legend11g1.s1234 bash scripts/legend_chain.sh
# Mirrors /workspace/base_chain_g1.sh stage for stage, minus the x-family
# extensions. Every stage skips what is on disk, so it can be re-run.
set -uo pipefail
ulimit -n "$(ulimit -Hn)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export UV_NO_SYNC=1  # the .venv is shared across pods; never let a stage re-sync it
export JAX_COMPILATION_CACHE_DIR=/workspace/data/jax-cache-legend
cd /workspace/GoalMisgeneralisation
# The switch is the base's own config: 001 --observation legend records it in cfg.json,
# and every later stage reads it back with the weights (arms inherit it through 013).
# Runs trained before the field existed are read through the OBSERVATION file below.
UV="$(command -v uv)"; [ -n "$UV" ] || { echo "NO_UV"; exit 1; }
DATA=/workspace/data
LEVELS_TRAIN=$DATA/levels/values/1.00-0.50@1M
LEVELS_MEASURE=$DATA/levels/values/1.00-0.50@500k
SEED=${SEED:-1234}; STEPS=${STEPS:-150000000}; NAME=${NAME:-legend11g1.s$SEED}
RUN=$DATA/runs/$NAME
# A base that needed the tail-LR extension lives under NAME + "e"; resume there.
[ -f "$DATA/runs/${NAME}e/BASE.json" ] && { NAME=${NAME}e; RUN=$DATA/runs/$NAME; }
OUT=results/gain-vs-factorised
mkdir -p "$OUT" "$DATA/figures-data/gain-vs-factorised"

$UV run python -c "
from goalmisgen.configs.env import MazeConfig
c = MazeConfig(max_episode_steps=120, num_envs=1, min_size=11, max_size=11, value_encoding='none', colour_is_the_only_value_cue=True, observation='legend')
e = c.encoder(); assert type(e).__name__ == 'LegendEncoder', e; print('encoder', e, e.shape)" || { echo "LEGEND_CONFIG_FAILED"; exit 1; }

echo "=== $NAME: train (legend, seed=$SEED gamma=1.0 steps=$STEPS)"
reached_steps() { ls "$1/local-files" 2>/dev/null | sed -n 's/^cp_0*//p' | sort -n | tail -1; }
if [ ! -f "$RUN/BASE.json" ]; then
  # A training process for this run may already be going (the chain was restarted
  # around it); wait for it rather than starting a second.
  while [ "$(pgrep -fc "001_maze_repro.py.*$NAME" 2>/dev/null | head -1)" -gt 0 ]; do sleep 120; done
  r=$(reached_steps "$RUN")
  if [ -z "$r" ] || [ "$r" -lt $((STEPS * 99 / 100)) ]; then
    rm -rf "$RUN"
    $UV run python experiments/001_maze_repro.py --net drc33 --correlation 1.0 \
      --min-size 11 --max-size 11 --levels "$LEVELS_TRAIN" --hide-values --observation legend --gamma 1.0 \
      --total-timesteps "$STEPS" --seed "$SEED" --run-dir "$RUN" \
      --note "E5 gain-vs-factorised control: novalue11g1 recipe on the LEGEND observation (objectives uncoloured on one plane, the richer one named by row/column marks in a border; goalmisgen/envs/legend.py). Recorded in cfg.json as observation=legend. 2026-09-15." \
      || { echo "TRAIN_FAILED $NAME"; exit 1; }
  fi
  echo legend > "$RUN/OBSERVATION"
  FINAL=$(ls "$RUN/local-files" | tail -1)
  echo "=== $NAME: convergence decode of $FINAL (g1 bar: argmax reaches >= 95%)"
  decode() { $UV run python scripts/decode_h1_drc.py "$1" --levels "$LEVELS_MEASURE" --split test \
      --episodes 2048 --num-envs 128 --out "/tmp/e5_decode.npz" 2>/dev/null | grep -oE "reached [0-9.]+%" | grep -oE "[0-9.]+" ; }
  pct=$(decode "$RUN/local-files/$FINAL"); echo "DECODE $NAME $FINAL reached ${pct:-?}%"
  BASE_RUN=$RUN; BASE_CP=$FINAL; PICKED=final
  if [ -z "$pct" ] || [ "${pct%.*}" -lt 95 ]; then
    # The g1 protocol extension: tail-LR continuation (4e-6, the anneal's floor) for up
    # to 100M, taking the first checkpoint that decodes >= 95% (as s3456e did).
    EXT=$DATA/runs/${NAME}e
    echo "=== ${NAME}e: extend +100M at tail LR 4e-6"
    r=$(reached_steps "$EXT")
    if [ -z "$r" ] || [ "$r" -lt 99000000 ]; then
      rm -rf "$EXT"
      $UV run python experiments/013_value_axis.py "$RUN/local-files/$FINAL" \
        --objective-values 1.0 0.5 --levels "$LEVELS_TRAIN" \
        --steps 100000000 --lr 4e-6 --checkpoints 8 --seed 42 --run-dir "$EXT" \
        --note "Tail-LR continuation of $NAME (unconverged at 150M): the g1 protocol extension. LEGEND observation." \
        || echo "EXT_TRAIN_FAILED ${NAME}e"
    fi
    echo legend > "$EXT/OBSERVATION"
    BASE_RUN=$EXT; BASE_CP=$(ls "$EXT/local-files" | tail -1); PICKED=extension-final-unconverged
    for cp in $(ls "$EXT/local-files"); do
      pct=$(decode "$EXT/local-files/$cp"); echo "DECODE ${NAME}e $cp reached ${pct:-?}%"
      if [ -n "$pct" ] && [ "${pct%.*}" -ge 95 ]; then BASE_CP=$cp; PICKED=extension-converged-decode; break; fi
    done
    NAME=${NAME}e; RUN=$EXT
  fi
  $UV run python - "$RUN" "$BASE_CP" "$PICKED" <<'PY' || { echo "BASEPICK_FAILED $NAME"; exit 1; }
import json, sys
from pathlib import Path
from goalmisgen.volume import parse_checkpoint_dirname
run, cp, picked = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {"checkpoint": f"local-files/{cp}", "values": [1.0, 0.5], "objectives": 2,
           "steps": parse_checkpoint_dirname(cp), "picked": picked, "observation": "legend"}
(run / "BASE.json").write_text(json.dumps(payload, indent=1))
print("BASE.json ->", payload)
PY
  case "$PICKED" in *unconverged*) echo "BASE_UNCONVERGED $NAME -- stopping before the sweep"; exit 1;; esac
fi

echo "=== $NAME: arms (o-family, both sweeps)"
$UV run python scripts/value_axis_sweep.py --agent "$NAME" --objectives 0 1 \
  --offsets 0.1 0.2 0.3 0.4 --steps 400000 || echo "SWEEP_FAILED $NAME o"

echo "=== $NAME: measures"
CP=$(python3 -c "import json;print(json.load(open('$RUN/BASE.json'))['checkpoint'])")
m() { local prefix=$1 bv=$2; shift 2
  [ -s "$OUT/$NAME-$prefix.txt" ] && grep -q . "$OUT/$NAME-$prefix.txt" && return
  $UV run python experiments/014_value_axis_analysis.py \
    --base "$RUN/$CP" --arms "$RUN/arms" --levels "$LEVELS_MEASURE" \
    --prefix "$prefix" --base-value "$bv" --arm-steps 400000 --at -1 "$@" \
    > "$OUT/$NAME-$prefix.txt" 2> "$OUT/$NAME-$prefix.err" || echo "MEASURE_FAILED $NAME $prefix"
}
m o0 1.0
m o1 0.5

echo "=== $NAME: restricted writes (o1)"
$UV run python scripts/restricted_writes.py "$RUN" --levels "$LEVELS_MEASURE" --sweep o1 \
  --json "$DATA/figures-data/gain-vs-factorised/restrict-$NAME.o1.json" \
  > "$OUT/restrict-$NAME.o1.txt" 2> "$OUT/restrict-$NAME.o1.err" || echo "RESTRICT_FAILED $NAME"

echo "=== $NAME: input gain (o1); planes 2 = objectives, 3 = legend"
$UV run python scripts/input_gain.py "$RUN" --levels "$LEVELS_MEASURE" --sweep o1 \
  --json "$DATA/figures-data/gain-vs-factorised/input-gain-$NAME.o1.json" \
  > "$OUT/input-gain-$NAME.o1.txt" 2> "$OUT/input-gain-$NAME.o1.err" || echo "GAIN_FAILED $NAME"

echo "E5_CHAIN_DONE $NAME"
