#!/usr/bin/env bash
# Matched SAC / RLPD experiment for the G1 piston task.
#
# The two arms are identical except for the demonstration prior: same SFT
# initialization, same TRAIN reset distribution, same reward, same 20-D action mask,
# same held-out EVAL suite, same interaction budget, same hyperparameters. SAC gets no
# demonstration transitions; RLPD mixes the frozen verified-executable buffer at
# DEMO_FRAC. Demonstration samples are counted separately from online interactions.
#
# Runs serially: each arm needs ~9.5 GB and the card has 16 GB.
#
# Usage: run_experiments.sh <scratch_dir> [max_env_steps]
set -uo pipefail

SCRATCH="${1:?usage: run_experiments.sh <scratch_dir> [max_env_steps]}"
STEPS="${2:-690000}"          # 690000 / 1380 = 500 online episodes per arm
PY="$HOME/miniconda3/envs/env_isaaclab/bin/python"
TRAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_sac.py"
RUNS="$SCRATCH/runs"
mkdir -p "$RUNS"

export MUJOCO_GL=egl
export PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab

# Evaluate every 100 episodes (100 * 1380 env steps).
EVAL_EVERY=138000

# Wait until the GPU actually has room, not merely until nvidia-smi lists no compute
# apps. A finished process disappears from the compute-app list before the driver frees
# its allocation, and an arm launched in that window dies with a CUDA OOM inside the VLM
# forward -- which is exactly how the first SAC launch failed.
wait_for_gpu() {
  local need_mib="${1:-12000}"
  for _ in $(seq 1 120); do
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
  echo "WARNING: GPU still busy after 20 min (free=${free} MiB); launching anyway"
}

run_arm() {
  local algo="$1"; shift
  wait_for_gpu 12000
  echo "=== $algo: launching ($STEPS env steps, eval every $EVAL_EVERY) ==="
  OUTF="$RUNS/${algo}.json" \
  ALGO="$algo" \
  RUN_DIR="$RUNS/$algo" \
  MAX_ENV_STEPS="$STEPS" \
  EVAL_EVERY="$EVAL_EVERY" \
  N_EVAL_CONDITIONS=50 \
  N_EVAL_PERIODIC=25 \
  EP_CHUNKS=23 \
  UTD=0.5 \
  BATCH=8 \
  SEED=0 \
  ALPHA_INIT=0.05 \
  ALPHA_LR=1e-3 \
  ACTOR_LR=3e-6 \
  DEMO_FRAC=0.5 \
  EVAL_STOCHASTIC="${EVAL_STOCHASTIC:-0}" \
  "$PY" "$TRAIN" > "$RUNS/${algo}.log" 2>&1
  local rc=$?
  echo "=== $algo: exit $rc ==="
  # Do not cascade: a failed arm means the next one would very likely fail the same way
  # (and would overwrite the GPU state needed to diagnose the first).
  if [ "$rc" -ne 0 ]; then
    echo "=== $algo FAILED; aborting the chain so the cause can be diagnosed ==="
    return "$rc"
  fi
}

run_arm sac || exit 1
run_arm rlpd || exit 1

"$PY" "$(dirname "$TRAIN")/analyze_runs.py" "$RUNS/analysis" \
    "$RUNS/sft_v2.json" "$RUNS/sac.json" "$RUNS/rlpd.json" || true
echo "=== all arms complete ==="
