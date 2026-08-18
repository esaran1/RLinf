#!/usr/bin/env bash
# Independent training seeds for the SAC/RLPD comparison.
#
# The n=25 held-out conditions measure ENVIRONMENT-condition variability. Independent
# training seeds measure ALGORITHMIC variability. A conference claim needs both, so the
# target is 3 seeds per method reported as mean +/- spread across seeds.
#
# Seed 1 of each method is the first matched pair (run by run_experiments.sh). This
# script adds seeds 2 and 3. Everything except the seed is identical, including the
# evaluation suite -- every seed is scored on the SAME held-out conditions, so the
# reset-suite seed is deliberately NOT varied with the training seed.
#
# Usage: run_replication.sh <scratch_dir> [seeds...]      (default seeds: 2 3)
set -uo pipefail

SCRATCH="${1:?usage: run_replication.sh <scratch_dir> [seeds...]}"
shift || true
SEEDS=("${@:-2 3}")
[ $# -eq 0 ] && SEEDS=(2 3)

PY="$HOME/miniconda3/envs/env_isaaclab/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$HERE/train_sac.py"
RUNS="$SCRATCH/runs"
mkdir -p "$RUNS"

export MUJOCO_GL=egl
export PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab

STEPS="${STEPS:-690000}"        # 500 online episodes, matching seed 1
EVAL_EVERY="${EVAL_EVERY:-138000}"

wait_for_gpu() {
  local need_mib="${1:-12000}"
  for _ in $(seq 1 180); do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] \
       && [ "$free" -ge "$need_mib" ]; then
      echo "GPU ready: ${free} MiB free"
      return 0
    fi
    sleep 10
  done
  echo "WARNING: GPU still busy after 30 min; launching anyway"
}

run_arm() {
  local algo="$1" seed="$2"
  local name="${algo}_s${seed}"
  wait_for_gpu 12000
  echo "=== $name: launching (seed $seed, $STEPS env steps) ==="
  OUTF="$RUNS/${name}.json" \
  ALGO="$algo" \
  RUN_DIR="$RUNS/$name" \
  MAX_ENV_STEPS="$STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  N_EVAL_CONDITIONS=50 \
  N_EVAL_PERIODIC=25 \
  EP_CHUNKS=23 \
  UTD=0.5 \
  BATCH=8 \
  SEED="$seed" \
  ALPHA_INIT=0.05 \
  ALPHA_LR=1e-3 \
  ACTOR_LR=3e-6 \
  DEMO_FRAC=0.5 \
  EVAL_STOCHASTIC="${EVAL_STOCHASTIC:-0}" \
  RESET_SUITE_SEED=20260817 \
  "$PY" "$TRAIN" > "$RUNS/${name}.log" 2>&1
  local rc=$?
  echo "=== $name: exit $rc ==="
  if [ "$rc" -ne 0 ]; then echo "=== $name FAILED; aborting ==="; return "$rc"; fi
  return 0
}

for s in "${SEEDS[@]}"; do
  run_arm sac "$s"  || exit 1
  run_arm rlpd "$s" || exit 1
done

echo "=== replication seeds ${SEEDS[*]} complete ==="
