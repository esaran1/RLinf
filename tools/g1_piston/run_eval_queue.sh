#!/usr/bin/env bash
# Robust post-hoc n=50 evaluation queue for saved G1 piston checkpoints.
#
# Hardened against the three orchestration failures this experiment actually hit:
#
#  1. LAUNCHER BUG: a trailing `[ cond ] && {...}` made a helper return 1 on success,
#     so a `|| exit` caller aborted after a SUCCESSFUL arm. Here every helper ends in
#     an explicit `return 0` and the loop never uses `|| exit`.
#  2. STALE-PID KILL: PIDs were read from an earlier listing and re-signalled after the
#     OS had recycled them, killing freshly launched jobs. This script signals nothing;
#     it owns its child directly and writes its own PID file.
#  3. SILENT PARTIAL OUTPUT: an interrupted sweep left a json that looked valid. Here a
#     result is only accepted if _status == OK and the per-condition count matches
#     N_EVAL; anything else is moved aside as .partial.
#
# Idempotent: already-complete outputs are skipped, so it can be re-run after any
# interruption and will resume where it stopped.
set -uo pipefail

S="${SCRATCH:-/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad}"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
RUNS="$S/runs"
N_EVAL="${N_EVAL:-50}"
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab

echo "$$" > "$S/eval_queue.pid"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# Accept a result only if it is genuinely complete.
is_complete(){
  local json="$1"
  [ -f "$json" ] || return 1
  "$PY" - "$json" "$N_EVAL" <<'PYEOF' >/dev/null 2>&1
import json,sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[2])
det=d.get("modes",{}).get("deterministic",{})
rows=det.get("per_condition") or []
sys.exit(0 if d.get("_status")=="OK" and len(rows)==n else 1)
PYEOF
}

wait_gpu(){
  local need=12000
  for _ in $(seq 1 180); do
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
       && [ $((total-used)) -ge "$need" ]; then return 0; fi
    sleep 10
  done
  log "WARNING: gpu still busy; proceeding anyway"
  return 0
}

run_one(){
  local arm="$1" step="$2" ck="$3"
  local out="$RUNS/n50_${arm}_${step}.json"

  if is_complete "$out"; then log "SKIP  $arm@$step (already complete)"; return 0; fi
  [ -f "$ck" ] || { log "MISS  $arm@$step checkpoint absent: $ck"; return 0; }
  # Quarantine any incomplete leftover so it can never be mistaken for a result.
  [ -f "$out" ] && mv -f "$out" "${out}.partial.$(date +%s)"

  wait_gpu
  log "RUN   $arm@$step (n=$N_EVAL, deterministic)"
  OUTF="$out" CKPT="$ck" RUN_DIR="$RUNS/n50_${arm}_${step}" \
    N_EVAL="$N_EVAL" RESET_SUITE_SEED=20260817 EP_CHUNKS=23 MODES=det \
    "$PY" "$T/eval_checkpoint.py" > "$RUNS/n50_${arm}_${step}.log" 2>&1
  local rc=$?

  if is_complete "$out"; then
    log "DONE  $arm@$step"
  else
    log "FAIL  $arm@$step (exit $rc) -- continuing with the rest of the queue"
    [ -f "$out" ] && mv -f "$out" "${out}.failed.$(date +%s)"
  fi
  return 0
}

# Decisive 414k pair first, then the trajectory points.
run_one rlpd   415140 "$RUNS/rlpd/rlpd_ckpt_step415140.pt"
run_one sac_s2 414000 "$RUNS/sac_s2/sac_ckpt_step414000.pt"
run_one rlpd   277320 "$RUNS/rlpd/rlpd_ckpt_step277320.pt"
run_one sac_s2 276000 "$RUNS/sac_s2/sac_ckpt_step276000.pt"
run_one rlpd   552000 "$RUNS/rlpd/rlpd_ckpt_step552000.pt"
run_one sac_s2 552240 "$RUNS/sac_s2/sac_ckpt_step552240.pt"

log "EVAL_QUEUE_DONE"
rm -f "$S/eval_queue.pid"
