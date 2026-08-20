#!/usr/bin/env bash
# The three training arms still missing from the 3-seed comparison:
#   SAC seed 3, RLPD seed 2, RLPD seed 3
#
# Everything is the frozen configuration used by the completed arms. Nothing here is
# tuned on the results so far; the only variable is the training seed.
#
# Hardened the same way as run_eval_queue.sh:
#  - helpers end in an explicit `return 0`, and the loop never uses `|| exit`, so a
#    SUCCESSFUL arm cannot abort the queue (the bug that stopped the first attempt);
#  - the script signals no PIDs; it owns its child directly (a stale-PID kill of mine
#    previously killed freshly launched jobs);
#  - an arm counts as complete only if _status == OK AND the run reached the full
#    interaction budget; anything else is quarantined rather than left looking valid;
#  - idempotent: complete arms are skipped, so a re-run resumes where it stopped.
set -uo pipefail

S="${SCRATCH:-/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad}"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
RUNS="$S/runs"
STEPS=690000
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab

echo "$$" > "$S/missing_seeds.pid"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

is_complete(){
  local json="$1"
  [ -f "$json" ] || return 1
  "$PY" - "$json" "$STEPS" <<'PYEOF' >/dev/null 2>&1
import json,sys
d=json.load(open(sys.argv[1])); budget=int(sys.argv[2])
eps=d.get("episodes") or []
reached = bool(eps) and eps[-1]["env_steps"] >= budget*0.98
sys.exit(0 if d.get("_status")=="OK" and reached else 1)
PYEOF
}

wait_gpu(){
  for _ in $(seq 1 360); do
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
       && [ $((total-used)) -ge 12000 ]; then return 0; fi
    sleep 20
  done
  log "WARNING: gpu still busy after 2h; proceeding"
  return 0
}

run_arm(){
  local algo="$1"
  local seed="$2"
  local name="${algo}_s${seed}"
  local out="$RUNS/${name}.json"

  if is_complete "$out"; then log "SKIP  $name (already complete)"; return 0; fi
  [ -f "$out" ] && mv -f "$out" "${out}.partial.$(date +%s)"

  wait_gpu
  log "RUN   $name (seed $seed, $STEPS env steps)"
  OUTF="$out" \
  ALGO="$algo" \
  RUN_DIR="$RUNS/$name" \
  MAX_ENV_STEPS="$STEPS" \
  EVAL_EVERY=138000 \
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
  EVAL_STOCHASTIC=0 \
  RESET_SUITE_SEED=20260817 \
  "$PY" "$T/train_sac.py" > "$RUNS/${name}.log" 2>&1
  local rc=$?

  if is_complete "$out"; then
    log "DONE  $name"
  else
    log "FAIL  $name (exit $rc) -- continuing with the remaining arms"
    [ -f "$out" ] && mv -f "$out" "${out}.failed.$(date +%s)"
  fi
  return 0
}

# SAC seed 3 first: it completes the SAC seed distribution, which is what currently
# blocks any claim (the two existing SAC seeds disagree more than SAC differs from RLPD).
run_arm sac  3
run_arm rlpd 2
run_arm rlpd 3

log "MISSING_SEEDS_DONE"
rm -f "$S/missing_seeds.pid"
